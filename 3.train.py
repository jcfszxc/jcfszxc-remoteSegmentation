import tqdm
import datetime
import time
import torch
import torch.nn.functional as F
from utils.dice_raw import dice_raw
from utils.dice_score import dice_loss
from sklearn.metrics import confusion_matrix
from math import cos, pi
from torch.utils.data import WeightedRandomSampler, DataLoader
from models.modeling import deeplabv3plus_resnet101
from dataset import DataGenerator
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def calculate_uncertainty(hardness):
    """
    Calculate uncertainty values based on hardness scores

    Parameters:
    hardness (torch.Tensor): The hardness tensor returned by the model

    Returns:
    torch.Tensor: Per-sample uncertainty scores
    float: Batch average uncertainty score
    dict: Additional uncertainty statistics
    """
    # Normalize hardness to [0,1] range for better interpretability
    epsilon = 1e-10
    hardness_max = hardness.max()
    hardness_min = hardness.min()
    normalized_hardness = (hardness - hardness_min) / \
        (hardness_max - hardness_min + epsilon)

    # If hardness is [batch_size, height, width]
    if len(hardness.shape) == 3:
        batch_uncertainty = torch.mean(normalized_hardness, dim=(1, 2))
    # If hardness is [batch_size, channels, height, width]
    elif len(hardness.shape) == 4:
        batch_uncertainty = torch.mean(normalized_hardness, dim=(1, 2, 3))
    # If hardness is already [batch_size, features]
    elif len(hardness.shape) == 2:
        batch_uncertainty = torch.mean(normalized_hardness, dim=1)
    else:
        batch_uncertainty = torch.mean(
            normalized_hardness.view(hardness.shape[0], -1), dim=1)

    # Batch average uncertainty
    avg_uncertainty = torch.mean(batch_uncertainty).item()

    # Calculate additional statistics
    uncertainty_stats = {
        'max': hardness_max.item(),
        'min': hardness_min.item(),
        'mean': torch.mean(hardness).item(),
        'std': torch.std(hardness).item(),
        'median': torch.median(hardness).item(),
        'normalized_mean': torch.mean(normalized_hardness).item()
    }

    return batch_uncertainty, avg_uncertainty, uncertainty_stats


def metrice(y_true, y_pred, epsilon=1e-7):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(
        y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))

    y_true = (y_true == 0)
    y_pred = (y_pred == 0)

    tp = ((y_true == y_pred) & (y_true == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)

    return dice


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
            1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class UncertaintySamplingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Initialize all samples with equal weights
        self.sample_weights = torch.ones(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def update_weights(self, indices, uncertainties, uncertainty_factor=2.0):
        """
        Update sampling weights based on uncertainty scores

        Parameters:
        indices (list): Indices of samples to update
        uncertainties (torch.Tensor): Uncertainty scores for samples
        uncertainty_factor (float): Factor to increase weight for high uncertainty samples
        """
        # Normalize uncertainties to [0, 1] for consistent scaling
        if len(uncertainties) > 0:
            min_uncertainty = uncertainties.min()
            max_uncertainty = uncertainties.max()
            if max_uncertainty > min_uncertainty:
                normalized_uncertainties = (
                    uncertainties - min_uncertainty) / (max_uncertainty - min_uncertainty)

                # Update weights - higher uncertainty means higher weight
                for i, idx in enumerate(indices):
                    if idx < len(self.sample_weights):
                        # Scale weights based on normalized uncertainty
                        # We add 1.0 to ensure all samples have at least the original weight
                        self.sample_weights[idx] = 1.0 + \
                            normalized_uncertainties[i] * uncertainty_factor

        return self.sample_weights


if __name__ == '__main__':
    epochs = 200
    BATCH_SIZE = 8
    alpha = 0.01

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Parameters for uncertainty sampling
    uncertainty_factor = 2.0  # Higher factor increases weight of uncertain samples
    # Minimum samples to collect before updating weights
    min_samples_before_update = 100

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = deeplabv3plus_resnet101(num_classes=2, pretrained_backbone=True)
    model.backbone.conv1 = torch.nn.Conv2d(
        22, 64, kernel_size=7, stride=2, padding=3, bias=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # Create base dataset
    train_dataset_base = DataGenerator(
        'crop_data/train.txt', batchsize=BATCH_SIZE)

    # Wrap with uncertainty sampling dataset
    train_dataset = UncertaintySamplingDataset(train_dataset_base)

    # Track all seen samples and their uncertainties
    sample_indices = []
    sample_uncertainties = []

    # Initialize with uniform sampling for the first epoch
    train_generator = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=True  # Start with shuffle for first epoch
    )

    test_dataset = DataGenerator('crop_data/test.txt', valid=True)
    valid_generator = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7)
    loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(
        np.array([0.1, 1.0])).float()).to(DEVICE)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    with open('train.log', 'w+') as f:
        f.write('epoch,train_loss,test_loss,train_dice,test_dice,train_dice_mean,test_dice_mean,train_uncertainty,test_uncertainty')

    best_dice = 0
    print('{} begin train!'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    for epoch in range(epochs):
        model.to(DEVICE)
        model.train()
        train_loss = 0
        begin = time.time()
        num = 0
        train_dice, train_dice_mean = 0, 0
        train_uncertainty = 0

        # Collect batch indices and uncertainty scores for this epoch
        epoch_indices = []
        epoch_uncertainties = []

        # If not first epoch and we have enough samples, update sampling weights
        if epoch > 0 and len(sample_indices) >= min_samples_before_update:
            # Convert to tensors for processing
            indices_tensor = torch.tensor(sample_indices)
            uncertainties_tensor = torch.tensor(sample_uncertainties)

            # Update weights in dataset
            weights = train_dataset.update_weights(
                indices_tensor,
                uncertainties_tensor,
                uncertainty_factor=uncertainty_factor
            )

            # Create sampler with updated weights
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )

            # Recreate dataloader with weighted sampler
            train_generator = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=0,
                pin_memory=True
            )

            print(
                f"Epoch {epoch}: Using uncertainty-based sampling with {len(weights)} samples")

        batch_idx = 0
        for x, y in train_generator:
            # Calculate current batch indices
            current_indices = list(range(
                batch_idx * BATCH_SIZE, min((batch_idx + 1) * BATCH_SIZE, len(train_dataset))))
            batch_idx += 1

            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred, hardness = model(x.float())

            ls, l_h = dice_loss(F.softmax(pred, dim=1).float(),
                                F.one_hot(y, 2).permute(0, 3, 1, 2).float(),
                                hardness,
                                multiclass=True)
            ls += dice_raw(F.softmax(pred, dim=1).float(),
                           F.one_hot(y, 2).permute(0, 3, 1, 2).float(),
                           multiclass=True)

            # Calculate uncertainties for current batch
            batch_uncertainty, avg_uncertainty, uncertainty_stats = calculate_uncertainty(
                hardness)
            train_uncertainty += avg_uncertainty

            # Store indices and uncertainties
            epoch_indices.extend(current_indices[:len(batch_uncertainty)])
            epoch_uncertainties.extend(batch_uncertainty.cpu().tolist())

            # Save high uncertainty samples for debugging
            if uncertainty_stats['normalized_mean'] > 0.7:
                print(
                    f"High uncertainty detected in batch {num} of epoch {epoch}")

            L = ls + l_h * alpha

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(L).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            train_loss += float(ls.data)
            train_dice_ = metrice(y, pred)
            train_dice_mean_ = (metrice(y, pred) + metrice(1-y, 1-pred)) / 2
            train_dice += train_dice_
            train_dice_mean += train_dice_mean_
            num += 1

        # Update master list of samples and uncertainties
        sample_indices = epoch_indices
        sample_uncertainties = epoch_uncertainties

        train_loss /= num
        train_dice, train_dice_mean = train_dice / num, train_dice_mean / num
        train_uncertainty /= num

        # Print summary of uncertainty distribution
        if len(epoch_uncertainties) > 0:
            uncertainty_arr = np.array(epoch_uncertainties)
            print(f"Epoch {epoch} uncertainty stats: min={uncertainty_arr.min():.4f}, max={uncertainty_arr.max():.4f}, "
                  f"mean={uncertainty_arr.mean():.4f}, median={np.median(uncertainty_arr):.4f}")

        num = 0
        test_loss = 0
        model.eval()
        test_dice, test_dice_mean = 0, 0
        test_uncertainty = 0

        with torch.no_grad():
            for x, y in valid_generator:
                x, y = x.to(DEVICE), y.to(DEVICE).long()

                pred, hardness = model(x.float())
                ls, l_h = dice_loss(F.softmax(pred, dim=1).float(),
                                    F.one_hot(y, 2).permute(
                                        0, 3, 1, 2).float(),
                                    hardness,
                                    multiclass=True)

                batch_uncertainty, avg_uncertainty, uncertainty_stats = calculate_uncertainty(
                    hardness)
                test_uncertainty += avg_uncertainty

                num += 1
                test_loss += float(ls.data)

                test_dice_ = metrice(y, pred)
                test_dice_mean_ = (metrice(y, pred) + metrice(1-y, 1-pred)) / 2
                test_dice += test_dice_
                test_dice_mean += test_dice_mean_

        scheduler.step(test_dice)

        test_loss /= num
        test_dice, test_dice_mean = test_dice / num, test_dice_mean / num
        test_uncertainty /= num

        if test_dice > best_dice:
            best_dice = test_dice
            model.to('cpu')
            torch.save(model, 'CA_NDVI_model.pkl')
            torch.save(model, f'checkpoints/{epoch}.pkl')
            print('BestModel Save Success!')

        print(
            '{} epoch:{}, time:{:.2f}s, lr:{:.6f}, \n train_loss:{:.4f}, val_loss:{:.4f}, train_dice:{:.4f}, val_dice:{:.4f}, train_mean_dice:{:.4f}, test_mean_dice:{:.4f}, train_uncertainty:{:.4f}, val_uncertainty:{:.4f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1, time.time() -
                begin, optimizer.state_dict()['param_groups'][0]['lr'],
                train_loss, test_loss, train_dice, test_dice, train_dice_mean, test_dice_mean, train_uncertainty, test_uncertainty
            ))
        with open('train.log', 'a+') as f:
            f.write('\n{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                epoch, train_loss, test_loss, train_dice, test_dice, train_dice_mean, test_dice_mean, train_uncertainty, test_uncertainty
            ))
