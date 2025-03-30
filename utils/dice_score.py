import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, hardness:Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        hardness_level_map = hardness.detach().reshape(-1)
        hardness_level_map = hardness_level_map * hardness_level_map.shape[0]

        input_detach = input.detach()
        hardness = hardness.detach().reshape(-1)
        hardness = hardness * hardness.shape[0]

        inter_h = torch.dot(input_detach.reshape(-1) * hardness, target.reshape(-1))

        inter = torch.dot(input.reshape(-1) * hardness_level_map, target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon), (2 * inter_h + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, hardness:Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    dice_h = 0
    for channel in range(input.shape[1]-1):
    # for channel in range(input.shape[1]):
        a, b = dice_coeff(input[:, channel, ...], target[:, channel, ...], hardness[:, 0, ...], reduce_batch_first, epsilon)
        dice += a
        dice_h += b

    return dice / input.shape[1], dice_h / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, hardness:Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    a, b = fn(input, target, hardness, reduce_batch_first=True)
    return 1 - a, b - 1
