# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import cv2
import numpy as np
import torch


def transform_img_plane_to_cartesian(x, img_size=128, r_min=-1, r_max=4, reverse=False):
    if reverse:
        x = img_size - x
    return (r_max - r_min) * (x / img_size) + r_min


def project_3d_to_2d(points_3d, fx, fy, ppx, ppy):
    X, Y, Z = points_3d[:, :, 0], points_3d[:, :, 1], points_3d[:, :, 2]
    x_distorted = fx * (X / Z) + ppx
    y_distorted = fy * (Y / Z) + ppy
    return torch.cat((x_distorted.unsqueeze(1), y_distorted.unsqueeze(1)), dim=1)


def project_3d_to_2d_original(points_3d, fx, fy, ppx, ppy, k1, k2, k3, p1, p2, h, w):
    # Step 1: Projection
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    x_distorted = fx * (X / Z) + ppx
    y_distorted = fy * (Y / Z) + ppy

    # Step 2: Distortion Correction using Inverse Brown-Conrady Model
    distorted_points = np.column_stack((x_distorted, y_distorted))
    distorted_points = np.expand_dims(distorted_points, axis=1)

    # Camera matrix and distortion coefficients
    camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # Step 2: Distortion Correction using cv2.undistortPoints
    undistorted_points = cv2.undistortPoints(distorted_points, camera_matrix, dist_coeffs, P=camera_matrix)
    x_undistorted, y_undistorted = (
        undistorted_points[:, 0, 0],
        undistorted_points[:, 0, 1],
    )

    # step 3: generate 2D boolean array for segmentation masks
    pixel_raw = np.zeros((h, w), dtype=bool)
    x_pixel_raw = np.round(x_distorted).astype(int)
    y_pixel_raw = np.round(y_distorted).astype(int)
    pixel_raw[
        y_pixel_raw[np.logical_and(y_pixel_raw < h, x_pixel_raw < w)],
        x_pixel_raw[np.logical_and(y_pixel_raw < h, x_pixel_raw < w)],
    ] = True

    pixel_corr = np.zeros((h, w), dtype=bool)
    x_pixel_corr = np.round(x_undistorted).astype(int)
    y_pixel_corr = np.round(y_undistorted).astype(int)
    pixel_corr[
        y_pixel_corr[np.logical_and(y_pixel_corr < h, x_pixel_corr < w)],
        x_pixel_corr[np.logical_and(y_pixel_corr < h, x_pixel_corr < w)],
    ] = True

    return np.array((x_distorted, y_distorted)), np.array((x_undistorted, y_undistorted))
