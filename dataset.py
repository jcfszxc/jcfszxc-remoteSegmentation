import cv2
import os
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, Path, target_size=(256, 256), batchsize=1, valid=False):
        with open(Path) as f:
            imgPath = list(map(lambda x: x.strip(), f.readlines()))

        if not valid:
            imgPath += imgPath[:batchsize-len(imgPath) % batchsize]
        self.imgPath_list = np.array(imgPath)
        print(f"Loaded {len(self.imgPath_list)} image paths")
        self.target_size = target_size
        self.indexes = np.arange(len(self.imgPath_list))
        self.valid = valid

    def __len__(self):
        return len(self.imgPath_list)

    def __getitem__(self, index):
        global errors
        try:
            # Get the image path at the current index
            img_path = self.imgPath_list[index]

            # Generate the data for this image
            x, y = self.__data_generation(img_path, index)
            x = np.transpose(x, axes=[2, 0, 1])

            return x, y
        except Exception as e:
            # Increment error counter
            if 'errors' in globals():
                globals()['errors'] += 1

            print(
                f"Error processing index {index}, path: {self.imgPath_list[index]}: {str(e)}")
            # Return a dummy sample in case of error to prevent dataset failure
            x = np.zeros((22, *self.target_size))
            y = np.zeros(self.target_size, dtype=np.uint8)
            return x, y

    def __data_generation(self, img_path, index=0):
        # Split the path into image and mask paths
        parts = img_path.split('\t')
        if len(parts) != 2:
            print(
                f"Invalid path format: {img_path}. Expected format: 'image_path\\tmask_path'")
            # Create dummy data as fallback
            img = np.zeros((256, 256, 22), dtype=np.float32)
            mask = np.zeros((256, 256), dtype=np.uint8)
            return img.copy(), mask.copy()

        img_file = parts[0]
        mask_file = parts[1]

        # Load the image and mask with better error handling
        try:
            # Check if files exist
            if not os.path.exists(img_file):
                raise FileNotFoundError(f"Image file not found: {img_file}")
            if not os.path.exists(mask_file):
                raise FileNotFoundError(f"Mask file not found: {mask_file}")

            # Load files
            img = np.load(img_file, allow_pickle=True)
            mask = np.load(mask_file, allow_pickle=True)

            # Check if image and mask are valid
            if img is None or mask is None:
                raise ValueError(
                    f"Failed to load image or mask (None returned)")

            # Verify dimensions
            if img.size == 0 or mask.size == 0:
                raise ValueError(f"Empty image or mask (zero size)")

            # Check if paths contain backslashes that need escaping
            if '\\' in img_file and not img_file.startswith('./'):
                print(
                    f"Warning: Path contains unescaped backslashes: {img_file}")
        except Exception as e:
            print(
                f"File loading error: {str(e)} - img_file: {img_file}, mask_file: {mask_file}")
            # Create dummy data as fallback
            img = np.zeros((256, 256, 22), dtype=np.float32)
            mask = np.zeros((256, 256), dtype=np.uint8)
            print(f"Created dummy data for problematic files")

        # Only print shapes for first few samples to avoid console spam
        if index < 5:
            print(f"Loaded image shape: {img.shape}, mask shape: {mask.shape}")

        # Ensure mask has proper values
        mask[mask > 1] = 1

        # Apply data augmentation
        random_int = random.randint(1, 5)
        if self.valid:
            random_int = 3

        if random_int == 1:
            img, mask = self.random_crop(img, mask)
        elif random_int == 2:
            img, mask = self.flip(img, mask)

        # Resize image
        img = cv2.resize(img, self.target_size)

        # Handle 3D mask (H, W, C) by squeezing or taking first channel
        if len(mask.shape) == 3:
            if mask.shape[2] == 1:
                # Squeeze single-channel masks to 2D
                mask = np.squeeze(mask, axis=2)
            else:
                # Take first channel for multi-channel masks
                mask = mask[:, :, 0]

        # Convert mask to uint8 and resize
        mask = mask.astype(np.uint8)

        # Make sure mask is not empty before resizing
        if mask.size > 0:
            mask = cv2.resize(mask, self.target_size,
                              interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError("Mask is empty after processing")

        # Add random noise for data augmentation
        img += np.random.normal(loc=0, scale=0.1, size=img.shape)

        return img.copy(), mask.copy()

    def random_crop(self, img, mask, scale=[0.8, 1.0], ratio=[3. / 4., 4. / 3.]):
        """
        Random crop with aspect ratio constraints
        """
        # Check dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Cannot crop image with zero dimensions")
            return img, mask

        aspect_ratio = math.sqrt(np.random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio
        src_h, src_w = img.shape[:2]

        bound = min((float(src_w) / src_h) / (w ** 2),
                    (float(src_h) / src_w) / (h ** 2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = src_h * src_w * np.random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        # Make sure crop dimensions don't exceed image dimensions
        w = min(w, src_w)
        h = min(h, src_h)

        if w <= 0 or h <= 0:
            return img, mask

        i = np.random.randint(0, max(1, src_w - w + 1))
        j = np.random.randint(0, max(1, src_h - h + 1))

        img = img[j:j + h, i:i + w]
        mask = mask[j:j + h, i:i + w]
        return img, mask

    def flip(self, img, mask):
        """
        Flip the image and mask
        :param img: input image
        :param mask: input mask
        :return: flipped image and mask
        """
        mode = np.random.choice([0, 1])
        return np.flip(img, axis=mode), np.flip(mask, axis=mode)


if __name__ == '__main__':
    import tqdm
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data generator test script')
    parser.add_argument('--data_path', type=str, default='crop_data/train.txt',
                        help='Path to the dataset file')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                        help='Target size for resizing (height, width)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for the dataloader')
    parser.add_argument('--valid', action='store_true',
                        help='Set to use validation mode (no random augmentations)')
    parser.add_argument('--test_samples', type=int, default=5,
                        help='Number of samples to test before full iteration')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')

    args = parser.parse_args()

    # Initialize the data generator
    print(f"Loading data from: {args.data_path}")

    try:
        with open(args.data_path, 'r') as f:
            lines = f.readlines()
            print(f"File contains {len(lines)} lines")
            if len(lines) > 0:
                print(f"First line in file: {lines[0].strip()}")

                # Check for potential path issues in the dataset file
                for i, line in enumerate(lines[:5]):  # Check first 5 lines
                    if '\\' in line and not line.startswith('./'):
                        print(
                            f"Warning: Line {i} contains backslashes which might cause issues: {line.strip()}")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

    # Create the data generator
    try:
        a = DataGenerator(
            args.data_path,
            target_size=tuple(args.target_size),
            batchsize=args.batch_size,
            valid=args.valid
        )

        print(f"Created DataGenerator with {len(a)} samples")

        # Try to load a few samples to verify
        print("\nTesting individual samples:")
        success_count = 0
        for i in range(min(args.test_samples, len(a))):
            try:
                x, y = a[i]
                print(f"✓ Sample {i}: x.shape={x.shape}, y.shape={y.shape}")
                success_count += 1
            except Exception as e:
                print(f"✗ Error on sample {i}: {str(e)}")

        print(
            f"\nSuccessfully loaded {success_count}/{min(args.test_samples, len(a))} test samples")

        # Now iterate through the entire dataset with progress bar
        print("\nIterating through all samples:")
        errors = 0
        try:
            for i, (x, y) in enumerate(tqdm.tqdm(a)):
                if i > 0 and i % 100 == 0:
                    print(f"Processed {i} samples ({errors} errors)")
        except KeyboardInterrupt:
            print(f"\nProcess interrupted by user after {i} samples")
        except Exception as e:
            print(f"\nError during iteration at sample {i}: {str(e)}")

        print(
            f"\nCompleted iteration with {errors} errors out of {len(a)} samples")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
