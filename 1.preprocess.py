#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/03/30 13:54
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : 1.preprocess.py
# @Description   :


import numpy as np
from osgeo import gdal
import rasterio
import os
from pathlib import Path


def normalize(image, min_val=None, max_val=None):
    """标准化图像到0-1范围"""
    if min_val is None:
        min_val = np.min(image)
    if max_val is None:
        max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def calculate_ndvi(image_path):
    """计算NDVI (归一化差分植被指数)"""
    with rasterio.open(image_path) as src:
        # 根据文件中的具体波段选择红外和近红外波段
        red_band = src.read(7).astype(np.float32)
        nir_band = src.read(3).astype(np.float32)

        # 安全处理除法操作
        mask = (nir_band + red_band) != 0
        ndvi = np.zeros_like(red_band, dtype=np.float32)
        ndvi[mask] = (nir_band[mask] - red_band[mask]) / \
            (nir_band[mask] + red_band[mask])

        # 将值限制在有效范围内
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi, src.transform, src.crs


def process_image_pair(image_path_a, image_path_b, output_path, label_path=None, dem_path=None):
    """处理一对图像并生成差异结果"""
    # 打开源图像
    ds_a = gdal.Open(image_path_a, gdal.GA_ReadOnly)
    ds_b = gdal.Open(image_path_b, gdal.GA_ReadOnly)

    # 验证图像兼容性
    bands_a, rows_a, cols_a = ds_a.RasterCount, ds_a.RasterYSize, ds_a.RasterXSize
    bands_b, rows_b, cols_b = ds_b.RasterCount, ds_b.RasterYSize, ds_b.RasterXSize

    if bands_a != bands_b or rows_a != rows_b or cols_a != cols_b:
        raise ValueError(f"图像维度不匹配: {image_path_a}与{image_path_b}")

    # 创建输出图像
    driver = gdal.GetDriverByName('GTiff')
    output_bands = bands_a * 2 + 2  # 原始波段 + 差异波段 + NDVI差异 + DEM
    output_ds = driver.Create(
        output_path, cols_a, rows_a, output_bands, gdal.GDT_Float32)

    # 设置地理参考信息
    output_ds.SetGeoTransform(ds_a.GetGeoTransform())
    output_ds.SetProjection(ds_a.GetProjection())

    # 处理原始波段
    for band in range(1, bands_a + 1):
        # 读取并归一化原始波段
        band_a_data = ds_a.GetRasterBand(band).ReadAsArray().astype(np.float32)
        normalized_band = normalize(np.clip(band_a_data, 0, 4000))
        output_ds.GetRasterBand(band).WriteArray(normalized_band)
        print(
            f"波段 {band}: min={normalized_band.min():.4f}, max={normalized_band.max():.4f}")

    # 处理差异波段
    for band in range(1, bands_a + 1):
        band_a_data = ds_a.GetRasterBand(band).ReadAsArray().astype(np.float32)
        band_b_data = ds_b.GetRasterBand(band).ReadAsArray().astype(np.float32)

        diff_band = np.abs(band_a_data - band_b_data)
        normalized_diff = normalize(np.clip(diff_band, 0, 2000))

        output_band_idx = bands_a + band
        output_ds.GetRasterBand(output_band_idx).WriteArray(normalized_diff)
        print(
            f"差异波段 {output_band_idx}: min={normalized_diff.min():.4f}, max={normalized_diff.max():.4f}")

    # 计算NDVI差异
    ndvi1, transform1, crs1 = calculate_ndvi(image_path_a)
    ndvi2, transform2, crs2 = calculate_ndvi(image_path_b)

    ndvi_difference = ndvi2 - ndvi1
    normalized_ndvi_diff = normalize(np.clip(ndvi_difference, -0.8, 0))

    ndvi_band_idx = bands_a * 2 + 1
    output_ds.GetRasterBand(ndvi_band_idx).WriteArray(normalized_ndvi_diff)
    print(
        f"NDVI差异 {ndvi_band_idx}: min={normalized_ndvi_diff.min():.4f}, max={normalized_ndvi_diff.max():.4f}")

    # 处理DEM数据（如果提供）
    if dem_path and os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1).astype(np.float32)
            normalized_dem = normalize(np.clip(dem_data, 500, 2000))

            dem_band_idx = bands_a * 2 + 2
            output_ds.GetRasterBand(dem_band_idx).WriteArray(normalized_dem)
            print(
                f"DEM {dem_band_idx}: min={normalized_dem.min():.4f}, max={normalized_dem.max():.4f}")

    # 复制标签文件（如果提供）
    if label_path and os.path.exists(label_path):
        print(f"复制标签文件: {label_path}")

    # 清理资源
    ds_a = None
    ds_b = None
    output_ds = None


def main():
    """主函数"""
    # 设置路径
    raw_data_path = Path('./raw_data')
    preprocess_train_images_path = Path('./preprocess_data/train/images/')
    preprocess_train_labels_path = Path('./preprocess_data/train/labels/')
    preprocess_test_images_path = Path('./preprocess_data/test/images/')
    preprocess_test_labels_path = Path('./preprocess_data/test/labels/')
    dem_base_path = Path('./DEM')

    # 确保目录存在
    raw_data_path.mkdir(exist_ok=True)
    preprocess_train_images_path.mkdir(parents=True, exist_ok=True)
    preprocess_train_labels_path.mkdir(parents=True, exist_ok=True)
    preprocess_test_images_path.mkdir(parents=True, exist_ok=True)
    preprocess_test_labels_path.mkdir(parents=True, exist_ok=True)

    # 图像对列表
    image_pairs_train = [
        ('2jineiya2017.tif', '2jineiya2018.tif', 'jineiya2_label.tif'),
        ('jineiya2017.tif', 'jineiya2018.tif', 'jineiya_label.tif'),
        ('linzhi2016.tif', 'linzhi2018.tif', 'linzhi_label.tif')
    ]
    image_pairs_test = [
        ('yuenan2019.tif', 'yuenan2021.tif', 'yuenan_label.tif')
    ]

    # 处理每一对训练图像
    for image_a, image_b, label_file in image_pairs_train:
        image_path_a = raw_data_path / image_a
        image_path_b = raw_data_path / image_b
        label_path = raw_data_path / label_file
        output_path = preprocess_train_images_path / image_a
        output_label_path = preprocess_train_labels_path / label_file
        dem_path = dem_base_path / image_a if dem_base_path.exists() else None

        print(f"处理: {image_path_a} 和 {image_path_b}")

        try:
            # 处理图像对
            process_image_pair(str(image_path_a), str(image_path_b), str(output_path),
                               str(label_path) if label_path.exists() else None,
                               str(dem_path) if dem_path else None)

            # 复制标签文件
            if label_path.exists():
                # 使用gdal打开标签文件
                label_ds = gdal.Open(str(label_path), gdal.GA_ReadOnly)
                if label_ds:
                    # 创建新的标签文件
                    label_driver = gdal.GetDriverByName('GTiff')
                    new_label_ds = label_driver.Create(
                        str(output_label_path),
                        label_ds.RasterXSize,
                        label_ds.RasterYSize,
                        label_ds.RasterCount,
                        label_ds.GetRasterBand(1).DataType
                    )

                    # 设置地理参考信息
                    new_label_ds.SetGeoTransform(label_ds.GetGeoTransform())
                    new_label_ds.SetProjection(label_ds.GetProjection())

                    # 复制数据
                    for band in range(1, label_ds.RasterCount + 1):
                        data = label_ds.GetRasterBand(band).ReadAsArray()
                        new_label_ds.GetRasterBand(band).WriteArray(data)

                    # 清理资源
                    label_ds = None
                    new_label_ds = None

                    print(f"标签文件已保存到: {output_label_path}")
                else:
                    print(f"无法打开标签文件: {label_path}")

            # 验证输出
            with rasterio.open(output_path) as src:
                output_data = src.read()
                print(
                    f'输出校验: min={output_data.min():.4f}, max={output_data.max():.4f}')

        except Exception as e:
            print(f"处理 {image_a} 和 {image_b} 时出错: {e}")

    # 处理每一对测试图像
    for image_a, image_b, label_file in image_pairs_test:
        image_path_a = raw_data_path / image_a
        image_path_b = raw_data_path / image_b
        label_path = raw_data_path / label_file
        output_path = preprocess_test_images_path / image_a
        output_label_path = preprocess_test_labels_path / label_file
        dem_path = dem_base_path / image_a if dem_base_path.exists() else None

        print(f"处理: {image_path_a} 和 {image_path_b}")

        try:
            # 处理图像对
            process_image_pair(str(image_path_a), str(image_path_b), str(output_path),
                               str(label_path) if label_path.exists() else None,
                               str(dem_path) if dem_path else None)

            # 复制标签文件
            if label_path.exists():
                # 使用gdal打开标签文件
                label_ds = gdal.Open(str(label_path), gdal.GA_ReadOnly)
                if label_ds:
                    # 创建新的标签文件
                    label_driver = gdal.GetDriverByName('GTiff')
                    new_label_ds = label_driver.Create(
                        str(output_label_path),
                        label_ds.RasterXSize,
                        label_ds.RasterYSize,
                        label_ds.RasterCount,
                        label_ds.GetRasterBand(1).DataType
                    )

                    # 设置地理参考信息
                    new_label_ds.SetGeoTransform(label_ds.GetGeoTransform())
                    new_label_ds.SetProjection(label_ds.GetProjection())

                    # 复制数据
                    for band in range(1, label_ds.RasterCount + 1):
                        data = label_ds.GetRasterBand(band).ReadAsArray()
                        new_label_ds.GetRasterBand(band).WriteArray(data)

                    # 清理资源
                    label_ds = None
                    new_label_ds = None

                    print(f"标签文件已保存到: {output_label_path}")
                else:
                    print(f"无法打开标签文件: {label_path}")

            # 验证输出
            with rasterio.open(output_path) as src:
                output_data = src.read()
                print(
                    f'输出校验: min={output_data.min():.4f}, max={output_data.max():.4f}')

        except Exception as e:
            print(f"处理 {image_a} 和 {image_b} 时出错: {e}")


if __name__ == "__main__":
    main()
