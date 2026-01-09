import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch
import random
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # required for 3D projection
from matplotlib import cm
from matplotlib.colors import Normalize
from .data_loader_camera_calibration import pc3d_to_tensor_idx

from smplx import SMPLX
from PIL import Image



gb_matrix = np.array([[375.66860062, 0., 319.99508973],
                      [0., 375.66347079, 239.41364796],
                      [0., 0., 1.]])

radar2rgb_tvec = np.array([-0.03981857, 1.35834002, -0.05225502])
radar2rgb_rotmatrix = np.array([[9.99458797e-01, 3.28646073e-02, 1.42475954e-03],
                                [4.78233954e-04, 2.87906567e-02, -9.99585349e-01],
                                [-3.28919997e-02, 9.99045052e-01, 2.87593582e-02]])

CAM_EXT = np.eye(4)
CAM_EXT[:3, :3] = radar2rgb_rotmatrix
CAM_EXT[:3, 3] = radar2rgb_tvec

CAM_INS = gb_matrix


def project_3d_to_2d(points):
    mk3d_cam = CAM_EXT[:3, :3] @ points.T + np.expand_dims(CAM_EXT[:3, 3], axis=1)
    uvw = CAM_INS.dot(mk3d_cam)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    return uvs

def project_3d_to_2d1(points):
    uvw = CAM_INS.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    return uvs

# project the 3D points to 2D image
def project_to_2d(points_3d, camera_matrix, camera_extrinsics):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_camera = np.dot(camera_extrinsics, points_3d_homogeneous.T).T
    points_image_homogeneous = np.dot(camera_matrix, points_camera[:, :3].T).T
    points_2d = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2, np.newaxis]
    return points_2d



#plot the 3D points on 2D images
def plot_projected_image(image, projected_vertices, sample_idx):
    try:
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            # image = image.transpose(1, 2, 0)  # CHW to HWC
            print(image.shape)
        sample_idx_names = sample_idx.split("actions")[-1].split(".")[0].split("/")
        print(sample_idx_names)
        sample_idx = sample_idx_names[1] + "," + sample_idx_names[-1]

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
        plt.title(f'Projected Vertices on Image {sample_idx}')
        plt.axis('off')
        plt.savefig(f"projected_image_{sample_idx}.png")
        print(f"projected_image_{sample_idx}.png saved")
        plt.close()
    except Exception as e:
        print(f"Error during plotting projected image for sample {sample_idx}: {e}")

# randomly select 5 images to plot
def plot_random_samples(dataset, num_samples=5):
    print("start plotting")
    try:
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            sample = dataset[idx]
            if sample is not None:
                plot_projected_image(sample['image'], sample['projected_vertices'], sample['name_path'])
    except Exception as e:
        print(f"Error during random sample plotting: {e}")



def plot_frames_for_gif(dataset, start_idx=0, max_frames=30, combined_gif_path='combined_output.gif'):
    print(f"Generating {max_frames} frames starting from index {start_idx} for combined projection+radar GIF...")
    combined_frames = []

    end_idx = min(start_idx + max_frames, len(dataset))

    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        if sample is None:
            continue

        image = sample.get('image', None)
        projected_vertices = sample['projected_vertices']
        indicator = sample['indicator']
        radar = sample['rawImage_XYZ']

        # print_nonzero_proportion(radar)

        print(indicator)
        # Filter projected_vertices to only keep points within image bounds
        if image is not None:
            img_h, img_w = image.shape[:2]
            mask = (
                (projected_vertices[:, 0] >= -500) & #(projected_vertices[:, 0] < img_w) &
                (projected_vertices[:, 1] >= -500) #& (projected_vertices[:, 1] < img_h)
            )
            projected_vertices = projected_vertices[mask]

        frame_proj = None
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            canvas1 = FigureCanvas(fig1)
            ax1.imshow(image)
            ax1.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
            ax1.set_title(f"p{indicator[0]}-a{indicator[1]}-f{indicator[2]}")
            ax1.axis('off')
            canvas1.draw()
            frame_proj = np.frombuffer(canvas1.buffer_rgba(), dtype='uint8').reshape(canvas1.get_width_height()[::-1] + (4,))
            plt.close(fig1)

        radar_points = sample['radar_points'].reshape(-1, 4)  # Ensure radar points are reshaped correctly
        radar_mesh = sample['vertices'] / 1000

        # print(radar_points[:, 0].min(), radar_points[:, 1].min(), radar_points[:, 2].min(),radar_points[:, 3].min())
        # print(radar_points[:, 0].max(), radar_points[:, 1].max(), radar_points[:, 2].max(),radar_points[:, 3].max())
        # print(radar_mesh[:, 0].min(), radar_mesh[:, 1].min(), radar_mesh[:, 2].min())
        # print(radar_mesh[:, 0].max(), radar_mesh[:, 1].max(), radar_mesh[:, 2].max())
        
        # print(np.abs(radar_points[:, 0]).max(), np.abs(radar_points[:, 1]).max(), np.abs(radar_points[:, 2]).max())
        # print(np.abs(radar_mesh[:, 0]).max(), np.abs(radar_mesh[:, 1]).max(), np.abs(radar_mesh[:, 2]).max())

        xlim = (-1.5, 1.5)
        ylim = (1.5, 3.5)
        zlim = (0., 2.5)

        fig2 = plt.figure(figsize=(8, 8))
        canvas2 = FigureCanvas(fig2)
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(radar_points[:, 0], radar_points[:, 1], radar_points[:, 2], c='black', s=1, label='Radar Points')
        ax.scatter(radar_mesh[:, 0], radar_mesh[:, 1], radar_mesh[:, 2], c='red', s=2, label='Mesh Vertices')
        ax.set_title(f"Radar Point Cloud + Mesh p{indicator[0]}-a{indicator[1]}-f{indicator[2]}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        x_len = xlim[1] - xlim[0]
        y_len = ylim[1] - ylim[0]
        z_len = zlim[1] - zlim[0]
        stretch_factor = max(x_len, y_len) / z_len
        ax.set_box_aspect([x_len, y_len, z_len * stretch_factor])
        ax.legend(loc='upper right')
        ax.view_init(elev=15, azim=-35)

        canvas2.draw()
        frame_radar = np.frombuffer(canvas2.buffer_rgba(), dtype='uint8').reshape(canvas2.get_width_height()[::-1] + (4,))
        plt.close(fig2)

        # Combine frames
        if frame_proj is not None:
            h_proj, w_proj = frame_proj.shape[:2]
            h_radar, w_radar = frame_radar.shape[:2]
            if h_proj != h_radar:
                frame_radar = np.array(Image.fromarray(frame_radar).resize((w_radar, h_proj)))
            combined_frame = np.concatenate([frame_proj[..., :3], frame_radar[..., :3]], axis=1)
        else:
            combined_frame = frame_radar[..., :3]

        combined_frames.append(combined_frame)

    if combined_frames:
        imageio.mimsave(combined_gif_path, combined_frames, duration=0.1)
        print(f"Saved combined GIF: {combined_gif_path}")
    else:
        print("No valid combined frames to save.")

def print_nonzero_proportion(radar_image):
    print(radar_image.shape)
    total_elements = 121*111*31  # total number of elements
    nonzero_elements = np.count_nonzero(radar_image)
    proportion = nonzero_elements / total_elements
    print(f"Non-zero elements: {nonzero_elements}")
    print(f"Total elements: {total_elements}")
    print(f"Proportion of non-zero values: {proportion:.4f}")
    
    

def _edges_from_corners8(c):
    return [(0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)]

def draw_bbox_world_3d(ax, corners8, color='g', lw=1.5, alpha=0.8):
    """Draw 3D bbox in world coordinates on a 3D axis."""
    for i,j in _edges_from_corners8(corners8):
        xs = [corners8[i,0], corners8[j,0]]
        ys = [corners8[i,1], corners8[j,1]]
        zs = [corners8[i,2], corners8[j,2]]
        ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)

def draw_bbox_xy(ax, corners_idx, color='g', lw=1.5, alpha=0.9):
    """Draw bbox edges projected on XY plane (tensor index space)."""
    for i,j in _edges_from_corners8(corners_idx):
        xs = [corners_idx[i,0], corners_idx[j,0]]
        ys = [corners_idx[i,1], corners_idx[j,1]]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha)

def draw_bbox_xz(ax, corners_idx, color='g', lw=1.5, alpha=0.9):
    """Draw bbox edges projected on XZ plane (tensor index space)."""
    for i,j in _edges_from_corners8(corners_idx):
        xs = [corners_idx[i,0], corners_idx[j,0]]
        zs = [corners_idx[i,2], corners_idx[j,2]]
        ax.plot(xs, zs, color=color, linewidth=lw, alpha=alpha)

def plot_frames_for_gif_new(dataset, start_idx=0, max_frames=20, combined_gif_path='combined_output.gif'):
    """
    Renders 4 panels per frame in one row:
      Col1 : Image with projected vertices (if available)
      Col2 : 3D radar point cloud + mesh vertices (+ world bbox if available)
      Col3 : 2D heatmap (X–Y) with PC projection (+ tensor bbox if available)
      Col4 : 2D heatmap (X–Z) with PC projection (+ tensor bbox if available)
    """
    print(f"Generating {max_frames} frames starting from index {start_idx} for combined GIF with 4 columns...")
    combined_frames = []

    end_idx = min(start_idx + max_frames, len(dataset))

    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        if sample is None:
            continue

        image = sample.get('image', None)
        projected_vertices = sample['projected_vertices']
        indicator = sample['indicator']
        radar_tensor = sample.get('rawImage_XYZ', None)[-1]  # (X,Y,Z) = (121,111,31)
        radar_points = sample['radar_points'][-1].reshape(-1, 4)
        bbbox = sample.get("bbbox", None)

        print(indicator)
        print(bbbox)
        # print(radar_tensor.shape)
        # print(radar_points.shape)

        # filter out all-zero rows
        if isinstance(radar_points, torch.Tensor):
            mask = (radar_points != 0).any(dim=1)
            radar_points = radar_points[mask]
            rp_np = radar_points.detach().cpu().numpy()
        else:
            radar_points = radar_points[np.any(radar_points != 0, axis=1)]
            rp_np = radar_points
        # print("Radar points range:",
        #       rp_np[:,0].min(), rp_np[:,0].max(),
        #       rp_np[:,1].min(), rp_np[:,1].max(),
        #       rp_np[:,2].min(), rp_np[:,2].max())

        # tensor threshold stats
        rt = radar_tensor.detach().cpu().numpy() if isinstance(radar_tensor, torch.Tensor) else radar_tensor
        nz = np.nonzero(rt >= 0.3)
        if len(nz[0]) > 0:
            xmin, xmax = nz[0].min(), nz[0].max()
            ymin, ymax = nz[1].min(), nz[1].max()
            zmin, zmax = nz[2].min(), nz[2].max()
            # print(f"Radar tensor >=0.3 bounds: X[{xmin},{xmax}] Y[{ymin},{ymax}] Z[{zmin},{zmax}] I[{rt.min()},{rt.max()}]")

        # project PC into tensor index space
        idx_all = pc3d_to_tensor_idx(rp_np[:,:3], clip=True)
        x_idx_all, y_idx_all, z_idx_all = idx_all[:,0], idx_all[:,1], idx_all[:,2]

        # ===== Col1: 2D image + projected vertices =====
        frame_proj = None
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            mask = (projected_vertices[:, 0] >= -500) & (projected_vertices[:, 1] >= -500)
            projected_vertices = projected_vertices[mask]

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            canvas1 = FigureCanvas(fig1)
            ax1.imshow(image)
            ax1.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
            ax1.set_title("Image+Proj")
            ax1.axis('off')
            canvas1.draw()
            frame_proj = np.frombuffer(canvas1.buffer_rgba(), dtype='uint8').reshape(canvas1.get_width_height()[::-1] + (4,))
            plt.close(fig1)

        # ===== Col2: 3D radar point cloud + mesh =====
        radar_mesh = sample['vertices'] / 1000.0
        xlim, ylim, zlim = (-1.5, 1.5), (1.5, 3.5), (0., 2.5)

        fig2 = plt.figure(figsize=(6, 6))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(rp_np[:, 0], rp_np[:, 1], rp_np[:, 2], c='black', s=1)
        ax2.scatter(radar_mesh[:, 0], radar_mesh[:, 1], radar_mesh[:, 2], c='red', s=2)
        # draw bbox in world coords
        if bbbox is not None and "pc" in bbbox:
            draw_bbox_world_3d(ax2, np.asarray(bbbox["pc"]["corners"]))
        ax2.set_title("Radar PC + Mesh")
        ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_zlim(zlim)
        ax2.view_init(elev=20, azim=-50)
        canvas2.draw()
        frame_radar = np.frombuffer(canvas2.buffer_rgba(), dtype='uint8').reshape(canvas2.get_width_height()[::-1] + (4,))
        plt.close(fig2)

        # ===== Col3 & Col4: 2D projections from tensor =====
        frame_xy = None
        frame_xz = None
        if rt is not None and np.count_nonzero(rt) > 0:
            nz_vals = rt[rt > 0]
            if nz_vals.size > 0:
                vmin = np.percentile(nz_vals, 5)
                vmax = np.percentile(nz_vals, 99.5)
                if vmin == vmax:
                    vmin, vmax = 0.0, float(nz_vals.mean()) if nz_vals.size else 1.0
            else:
                vmin, vmax = 0.0, 1.0
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap = cm.get_cmap("jet")

            # --- Col3: XY projection
            sum_xy = rt.sum(axis=2); cnt_xy = (rt > 0).sum(axis=2)
            img_xy = np.divide(sum_xy, cnt_xy, out=np.zeros_like(sum_xy, dtype=float), where=cnt_xy > 0)
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            canvas3 = FigureCanvas(fig3)
            im3 = ax3.imshow(img_xy.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
            ax3.scatter(x_idx_all, y_idx_all, s=6, c='r', alpha=0.9)
            if bbbox is not None and "tensor" in bbbox:
                draw_bbox_xy(ax3, np.asarray(bbbox["tensor"]["corners"]))
            ax3.set_title("XY Heatmap")
            ax3.set_xlabel("X bins"); ax3.set_ylabel("Y bins")
            fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            canvas3.draw()
            frame_xy = np.frombuffer(canvas3.buffer_rgba(), dtype='uint8').reshape(canvas3.get_width_height()[::-1] + (4,))
            plt.close(fig3)

            # --- Col4: XZ projection
            sum_xz = rt.sum(axis=1); cnt_xz = (rt > 0).sum(axis=1)
            img_xz = np.divide(sum_xz, cnt_xz, out=np.zeros_like(sum_xz, dtype=float), where=cnt_xz > 0)
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            canvas4 = FigureCanvas(fig4)
            im4 = ax4.imshow(img_xz.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
            ax4.scatter(x_idx_all, z_idx_all, s=6, c='r', alpha=0.9)
            if bbbox is not None and "tensor" in bbbox:
                draw_bbox_xz(ax4, np.asarray(bbbox["tensor"]["corners"]))
            ax4.set_title("XZ Heatmap")
            ax4.set_xlabel("X bins"); ax4.set_ylabel("Z bins")
            fig4.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            canvas4.draw()
            frame_xz = np.frombuffer(canvas4.buffer_rgba(), dtype='uint8').reshape(canvas4.get_width_height()[::-1] + (4,))
            plt.close(fig4)

        # ===== Combine horizontally =====
        panels = [f[..., :3] for f in [frame_proj, frame_radar, frame_xy, frame_xz] if f is not None]
        if panels:
            target_h = panels[0].shape[0]
            for i in range(len(panels)):
                if panels[i].shape[0] != target_h:
                    w = panels[i].shape[1]
                    panels[i] = np.array(Image.fromarray(panels[i]).resize((w, target_h)))
            combined_frame = np.concatenate(panels, axis=1)
            combined_frames.append(combined_frame)

    if combined_frames:
        imageio.mimsave(combined_gif_path, combined_frames, duration=0.1)
        print(f"Saved combined GIF with 4 columns: {combined_gif_path}")
    else:
        print("No valid frames to save.")




from typing import Dict
import cv2
import numpy as np
import torch

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD  = 255. * np.array([0.229, 0.224, 0.225])



def normalize_img_patch(img_patch: np.ndarray, mean: np.ndarray = DEFAULT_MEAN, std: np.ndarray = DEFAULT_STD) -> np.ndarray:

    out = img_patch.copy()
    for c in range(min(img_patch.shape[0], 3)):
        out[c, :, :] = (out[c, :, :] - mean[c]) / std[c]
    return out


def denormalize_img_patch(img_patch: np.ndarray, mean: np.ndarray = DEFAULT_MEAN, std: np.ndarray = DEFAULT_STD) -> np.ndarray:
    """
    Reverse the normalization applied to a CHW image patch.
    Input: img_patch (C, H, W) normalized as (x - mean) / std
    Output: uint8 image in HWC order with values in [0,255]
    """
    out = img_patch.copy().astype(np.float32)
    for c in range(min(out.shape[0], len(mean))):
        out[c, :, :] = out[c, :, :] * std[c] + mean[c]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = np.transpose(out, (1, 2, 0))
    out = out[:,:,::-1].astype(np.float32)/255.0
    return out









import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch
import random
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # required for 3D projection
from matplotlib import cm
from matplotlib.colors import Normalize
from .data_loader_camera_calibration import pc3d_to_tensor_idx

from smplx import SMPLX
from PIL import Image
import cv2
def plot_frames_for_gif_depth(dataset, start_idx=0, max_frames=50*100, combined_gif_path='combined_output.gif', act_id = 0, is_plot="save"):
    """
    Renders 4 panels per frame in one row:
      Col1 : Image with projected vertices (if available)
      Col2 : 3D radar point cloud + mesh vertices (+ world bbox if available)
      Col3 : 2D heatmap (X–Y) with PC projection (+ tensor bbox if available)
      Col4 : 2D heatmap (X–Z) with PC projection (+ tensor bbox if available)
    """
    print(f"Generating {max_frames} frames starting from index {start_idx} for combined GIF with 4 columns...")
    combined_frames = []

    end_idx = min(start_idx + max_frames, len(dataset))
    
    font_size = 24
    font_name = 'Times New Roman'

    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        if sample is None:
            continue

        image = sample.get('image', None)[[2,1,0],...]/255.
        depth = sample.get('depth', None)
        item_image = sample.get('item_image', None)
        item_depth = sample.get('item_depth', None)
        projected_vertices = sample['projected_vertices']
        indicator = sample['indicator']
        if indicator[1] != act_id:
            continue
        if indicator[1]<36 and len(combined_frames)>=30:
            break
        # image = denormalize_img_patch(item_image["img"])
        # depth = item_depth["img"]
        radar_tensor = sample.get('rawImage_XYZ', None)[-1]  # (X,Y,Z) = (121,111,31)
        radar_points = sample['radar_points'][-1].reshape(-1, 4)
        bbbox = sample.get("bbbox", None)

        print(indicator, f"image: {image.shape if image is not None else None}, depth: {depth.shape if depth is not None else None}, radar_tensor: {radar_tensor.shape if radar_tensor is not None else None}, radar_points: {radar_points.shape if radar_points is not None else None}")
        # print(image.shape, image.min(), image.max())
        # print(depth.shape, depth.min(), depth.max())
        # print(bbbox)
        # # print(radar_tensor.shape)
        # print(radar_points.shape)

        # filter out all-zero rows
        if isinstance(radar_points, torch.Tensor):
            mask = (radar_points != 0).any(dim=1)
            radar_points = radar_points[mask]
            rp_np = radar_points.detach().cpu().numpy()
        else:
            radar_points = radar_points[np.any(radar_points != 0, axis=1)]
            rp_np = radar_points
        # print("Radar points range:",
        #       rp_np[:,0].min(), rp_np[:,0].max(),
        #       rp_np[:,1].min(), rp_np[:,1].max(),
        #       rp_np[:,2].min(), rp_np[:,2].max())

        # tensor threshold stats
        rt = radar_tensor.detach().cpu().numpy() if isinstance(radar_tensor, torch.Tensor) else radar_tensor
        nz = np.nonzero(rt >= 0.3)
        if len(nz[0]) > 0:
            xmin, xmax = nz[0].min(), nz[0].max()
            ymin, ymax = nz[1].min(), nz[1].max()
            zmin, zmax = nz[2].min(), nz[2].max()
            # print(f"Radar tensor >=0.3 bounds: X[{xmin},{xmax}] Y[{ymin},{ymax}] Z[{zmin},{zmax}] I[{rt.min()},{rt.max()}]")

        # project PC into tensor index space
        idx_all = pc3d_to_tensor_idx(rp_np[:,:3], clip=True)
        x_idx_all, y_idx_all, z_idx_all = idx_all[:,0], idx_all[:,1], idx_all[:,2]
        
        # Tensor threshold stats
        rt = radar_tensor.detach().cpu().numpy() if isinstance(radar_tensor, torch.Tensor) else radar_tensor
        nz_vals = rt[rt > 0]
        if nz_vals.size > 0:
            vmin = np.percentile(nz_vals, 5)
            vmax = np.percentile(nz_vals, 99.5)
            if vmin == vmax:
                vmin, vmax = 0.0, float(nz_vals.mean()) if nz_vals.size else 1.0
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = cm.get_cmap("jet")

        # ===== Col1: 2D image + projected vertices =====
        frame_rgb = None
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            mask = (projected_vertices[:, 0] >= -500) & (projected_vertices[:, 1] >= -500)
            projected_vertices = projected_vertices[mask]

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            canvas1 = FigureCanvas(fig1)
            ax1.imshow(image)
            # ax1.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
            ax1.set_title("RGB Image", fontdict={'fontsize': font_size, 'fontname': font_name})
            ax1.axis('off')
            canvas1.draw()
            frame_rgb = np.frombuffer(canvas1.buffer_rgba(), dtype='uint8').reshape(canvas1.get_width_height()[::-1] + (4,))
            plt.close(fig1)


        # ===== Col1: 2D depth + projected vertices =====
        frame_depth = None
        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.numpy()
            if depth.ndim == 3 and depth.shape[0] in [1, 3]:
                depth = depth.transpose(1, 2, 0)
            mask = (projected_vertices[:, 0] >= -500) & (projected_vertices[:, 1] >= -500)
            projected_vertices = projected_vertices[mask]

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            canvas1 = FigureCanvas(fig1)
            ax1.imshow(depth)
            # ax1.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
            ax1.set_title("Depth Image", fontdict={'fontsize': font_size, 'fontname': font_name})
            ax1.axis('off')
            canvas1.draw()
            frame_depth = np.frombuffer(canvas1.buffer_rgba(), dtype='uint8').reshape(canvas1.get_width_height()[::-1] + (4,))
            plt.close(fig1)
            
        # ===== Col1: 2D image + projected vertices =====
        frame_proj = None
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            mask = (projected_vertices[:, 0] >= -500) & (projected_vertices[:, 1] >= -500)
            projected_vertices = projected_vertices[mask]

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            canvas1 = FigureCanvas(fig1)
            ax1.imshow(image)
            ax1.scatter(projected_vertices[:, 0], projected_vertices[:, 1], c='r', s=2)
            ax1.set_title("RGB Image + 2D Mesh Proj.", fontdict={'fontsize': font_size, 'fontname': font_name})
            ax1.axis('off')
            canvas1.draw()
            frame_proj = np.frombuffer(canvas1.buffer_rgba(), dtype='uint8').reshape(canvas1.get_width_height()[::-1] + (4,))
            plt.close(fig1)
        
        # ===== Col2: 3D radar point cloud + mesh =====
        radar_mesh = sample['vertices'] / 1000.0
        xlim, ylim, zlim = (-0., 6.), (-4., 2.), (0., 3.)

        fig2 = plt.figure(figsize=(6, 6))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(rp_np[:, 0], rp_np[:, 1], rp_np[:, 2], c='cyan', s=2)
        ax2.scatter(radar_mesh[:, 0], radar_mesh[:, 1], radar_mesh[:, 2], c='red', s=2)
        # if bbbox is not None and "pc" in bbbox:
        #     draw_bbox_world_3d(ax2, np.asarray(bbbox["pc"]["corners"]))
        ax2.set_title("Radar Point Clouds + 3D Mesh", fontdict={'fontsize': font_size, 'fontname': font_name})
        ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_zlim(zlim)
        ax2.view_init(elev=20, azim=-45)
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.set_zlabel("")
        canvas2.draw()
        frame_radar = np.frombuffer(canvas2.buffer_rgba(), dtype='uint8').reshape(canvas2.get_width_height()[::-1] + (4,))
        plt.close(fig2)

        # ===== Col3: XY heatmap =====
        sum_xy = rt.sum(axis=2); cnt_xy = (rt > 0).sum(axis=2)
        img_xy = np.divide(sum_xy, cnt_xy, out=np.zeros_like(sum_xy, dtype=float), where=cnt_xy > 0)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        canvas3 = FigureCanvas(fig3)
        im3 = ax3.imshow(img_xy.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax3.set_title("Radar Tensor (X-Y BEV)", fontdict={'fontsize': font_size, 'fontname': font_name})
        ax3.set_xlabel("X bins"); ax3.set_ylabel("Y bins")
        ax3.set_xticks([])
        ax3.set_yticks([])
        fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        canvas3.draw()
        frame_xy = np.frombuffer(canvas3.buffer_rgba(), dtype='uint8').reshape(canvas3.get_width_height()[::-1] + (4,))
        plt.close(fig3)

        # ===== Col4: XZ heatmap =====
        sum_xz = rt.sum(axis=1); cnt_xz = (rt > 0).sum(axis=1)
        img_xz = np.divide(sum_xz, cnt_xz, out=np.zeros_like(sum_xz, dtype=float), where=cnt_xz > 0)
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        canvas4 = FigureCanvas(fig4)
        im4 = ax4.imshow(img_xz.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax4.set_title("Radar Tensor (X-Z Front View)", fontdict={'fontsize': font_size, 'fontname': font_name})
        ax4.set_xlabel("X bins"); ax4.set_ylabel("Z bins")
        ax4.set_xticks([])
        ax4.set_yticks([])
        fig4.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        canvas4.draw()
        frame_xz = np.frombuffer(canvas4.buffer_rgba(), dtype='uint8').reshape(canvas4.get_width_height()[::-1] + (4,))
        plt.close(fig4)
            
            
            
            
        # ===== Combine horizontally =====
        panels = [f[..., :3] for f in [frame_rgb, frame_depth, frame_proj, frame_xy, frame_xz, frame_radar] if f is not None]
        if panels:
            # Resize all panels to the same height and width
            target_h = panels[0].shape[0]
            target_w = panels[0].shape[1]
            for i in range(len(panels)):
                if panels[i].shape[0] != target_h or panels[i].shape[1] != target_w:
                    panels[i] = np.array(Image.fromarray(panels[i]).resize((target_w, target_h)))

            # Create a blank canvas for the 2x3 grid
            grid_rows = 2
            grid_cols = 3
            grid_h = target_h * grid_rows
            grid_w = target_w * grid_cols
            grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            # Place panels into the grid
            for idx, panel in enumerate(panels):
                row = idx // grid_cols
                col = idx % grid_cols
                if row < grid_rows:  # Ensure we don't exceed the grid size
                    grid_frame[row * target_h:(row + 1) * target_h, col * target_w:(col + 1) * target_w, :] = panel

            combined_frames.append(grid_frame)
    if is_plot=="save" and len(combined_frames) > 0:
        imageio.mimsave(combined_gif_path, combined_frames, duration=0.1)
        print(f"Saved combined GIF with 2x3 grid: {combined_gif_path}")
    elif is_plot=="show" and len(combined_frames) > 0:
        # Display the combined frames as an animation
        import IPython.display as display
        from PIL import Image as PILImage
        pil_frames = [PILImage.fromarray(frame) for frame in combined_frames]
        display.display(pil_frames[0])
        for frame in pil_frames[1:]:
            display.clear_output(wait=True)
            display.display(frame)
    else:
        print("No valid frames to save.")
    
