#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 13:54
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : 2.crop.py
# @Description   : Optimized script for cropping and preprocessing image data

import cv2
from sklearn.model_selection import train_test_split
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from libtiff import TIFF
from pathlib import Path
import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")


def read_image(path):
    try:
        with rasterio.open(path) as src:
            image = src.read()  # 所有波段
            # 重新排列维度从(bands, height, width)到(height, width, bands)
            image = image.transpose((1, 2, 0))
            return image
    except Exception as e:
        print(f"Error loading image file {path}: {e}")
        return None


def create_directories(paths):
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_tiff_image(file_path):
    """Load a TIFF image with error handling."""
    try:
        return TIFF.open(file_path, mode='r').read_image()
    except Exception as e:
        print(f"Error loading TIFF file {file_path}: {e}")
        return None


def load_label(file_path):
    """Load a label image with error handling."""
    try:
        return np.array(Image.open(file_path))
    except Exception as e:
        print(f"Error loading label file {file_path}: {e}")
        return None


def crop_and_save_patches(image, label, crop_size, output_img_dir, output_label_dir, start_idx, num_crops=150):
    """Crop image and label into patches and save them."""
    idx = start_idx
    a, b = image.shape[:2]

    if a < crop_size or b < crop_size:
        print(
            f"Warning: Image dimensions ({a}x{b}) smaller than crop size ({crop_size})")
        return idx

    # Check if both image and label have valid dimensions
    if image is None or label is None:
        print("Error: Invalid image or label data")
        return idx

    # Verify label dimensions match image dimensions
    if image.shape[:2] != label.shape[:2]:
        print(
            f"Error: Image shape {image.shape[:2]} doesn't match label shape {label.shape[:2]}")
        return idx

    for _ in tqdm.tqdm(range(num_crops), desc=f"Generating crops for image"):
        j = np.random.randint(0, a - crop_size)
        k = np.random.randint(0, b - crop_size)

        img_patch = image[j:j + crop_size, k:k + crop_size]
        lab_patch = label[j:j + crop_size, k:k + crop_size]

        # Verify patches are valid
        if img_patch.size == 0 or lab_patch.size == 0:
            print(f"Warning: Generated empty patch at position ({j}, {k})")
            continue

        # Save the patches
        img_file = os.path.join(output_img_dir, f"{idx}.npy")
        lab_file = os.path.join(output_label_dir, f"{idx}.npy")

        np.save(img_file, img_patch)
        np.save(lab_file, lab_patch)

        # Verify files were created with non-zero size
        if not os.path.exists(img_file) or os.path.getsize(img_file) == 0:
            print(f"Warning: Empty image file created: {img_file}")
            continue

        if not os.path.exists(lab_file) or os.path.getsize(lab_file) == 0:
            print(f"Warning: Empty label file created: {lab_file}")
            continue

        idx += 1

    return idx


def visualize_npy_as_color_image(npy_file, output_path=None):
    """Visualize .npy file as a color image and optionally save it."""
    try:
        data = np.load(npy_file)

        # Handle different channel configurations
        if data.ndim == 3 and data.shape[2] >= 3:
            # Use first three channels for RGB display
            display_data = data[:, :, :3]
        elif data.ndim == 3 and data.shape[2] == 1:
            # Single channel - display as grayscale
            display_data = data[:, :, 0]
        else:
            # Just use the data as is
            display_data = data

        # Normalize for display if needed
        if display_data.max() > 1.0 and display_data.dtype != np.uint8:
            display_data = display_data / display_data.max()

        plt.figure(figsize=(8, 8))
        if len(display_data.shape) == 2 or display_data.shape[2] == 1:
            plt.imshow(display_data, cmap='gray')
        else:
            plt.imshow(display_data)
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        print(f"Error visualizing {npy_file}: {e}")


def main():
    # Define directories
    # 图像对列表
    image_pairs_train = [
        ('2jineiya2017.tif', 'jineiya2_label.tif'),
        ('jineiya2017.tif', 'jineiya_label.tif'),
        ('linzhi2016.tif', 'linzhi_label.tif')
    ]
    image_pairs_test = [
        ('yuenan2019.tif', 'yuenan_label.tif')
    ]

    crop_size = 256
    input_image_dir_train = './preprocess_data/train/images/'
    input_label_dir_train = './preprocess_data/train/labels/'
    input_image_dir_test = './preprocess_data/test/images/'
    input_label_dir_test = './preprocess_data/test/labels/'
    output_img_dir = './crop_data/seg_img/'
    output_label_dir = './crop_data/seg_label/'
    num_crops_per_image = 150

    # Create output directories
    create_directories([output_img_dir, output_label_dir])

    # Process each image
    idx = 0

    # Process training images
    for image_name, label_file in image_pairs_train:
        image_path = os.path.join(input_image_dir_train, image_name)
        label_path = os.path.join(input_label_dir_train, label_file)

        image = read_image(image_path)
        label = read_image(label_path)

        if image is None or label is None:
            print(
                f"Skipping pair due to loading error: {image_name}, {label_file}")
            continue

        # Log image statistics
        print(
            f"Image: {image_name}, Shape: {image.shape}, Min: {np.min(image)}, Max: {np.max(image)}")
        print(
            f"Label: {label_file}, Shape: {label.shape}, Min: {np.min(label)}, Max: {np.max(label)}")

        # Crop and save patches
        idx = crop_and_save_patches(
            image, label, crop_size,
            output_img_dir, output_label_dir,
            idx, num_crops_per_image
        )

    # Process test images
    for image_name, label_file in image_pairs_test:
        image_path = os.path.join(input_image_dir_test, image_name)
        label_path = os.path.join(input_label_dir_test, label_file)

        image = read_image(image_path)
        label = read_image(label_path)

        if image is None or label is None:
            print(
                f"Skipping pair due to loading error: {image_name}, {label_file}")
            continue

        # Log image statistics
        print(
            f"Image: {image_name}, Shape: {image.shape}, Min: {np.min(image)}, Max: {np.max(image)}")
        print(
            f"Label: {label_file}, Shape: {label.shape}, Min: {np.min(label)}, Max: {np.max(label)}")

        # Crop and save patches
        idx = crop_and_save_patches(
            image, label, crop_size,
            output_img_dir, output_label_dir,
            idx, num_crops_per_image
        )

    print(f"Total patches generated: {idx}")

    # Create train/test split text files with correct path format
    temp = []
    valid_files = 0

    for i in tqdm.tqdm(os.listdir(output_img_dir), desc="Verifying files"):
        img_file = os.path.join(output_img_dir, i)
        label_file = os.path.join(output_label_dir, i)

        # Check that both files exist and have non-zero size
        if os.path.exists(img_file) and os.path.getsize(img_file) > 0 and \
           os.path.exists(label_file) and os.path.getsize(label_file) > 0:
            # Use proper path format with forward slashes
            temp.append('{}\t{}'.format(
                img_file.replace('\\', '/'),
                label_file.replace('\\', '/')
            ))
            valid_files += 1

    print(f"Valid files found: {valid_files} out of {idx}")

    # Split into train and test sets
    train, test = train_test_split(
        temp, test_size=0.2, random_state=42, shuffle=True)

    with open('./crop_data/train.txt', 'w+') as f:
        f.write('\n'.join(train))

    with open('./crop_data/test.txt', 'w+') as f:
        f.write('\n'.join(test))

    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")


if __name__ == "__main__":
    main()
