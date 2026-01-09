import matplotlib
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch


from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import os


def save_human_mesh(vertices, faces, filename):
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
    except Exception as e:
        print(f"Error saving mesh to {filename}: {e}")
        print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")

def plot_mesh(ax, vertices, faces, title):
    ax.set_title(title)
    mesh = Poly3DCollection(vertices[faces], alpha=0.1, edgecolor='k')
    ax.add_collection3d(mesh)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def save_random_comparison_figures(pred_vertices, gt_vertices, faces_male, faces_female, genders, iteration, result_path, phase, num_samples=5):
    num_samples = min(num_samples, pred_vertices.size(0))  # 确保样本数量不超过实际的batch大小
    indices = random.sample(range(pred_vertices.size(0)), num_samples)

    for i in indices:
        filename = f"{result_path}/{phase}_comparison_iter_{iteration}_sample_{i}.png"
        plot_comparison(pred_vertices[i].cpu().numpy(), gt_vertices[i].cpu().numpy(), faces_male, faces_female, genders.cpu().numpy(), i, filename)
        print(f"Saved {phase} comparison figure: {filename}")

def plot_comparison(pred_vertices, gt_vertices, faces_male, faces_female, genders, idx, filename):
    fig = plt.figure(figsize=(12, 8))
    
    # 动态计算中心点
    def calculate_limits(vertices):
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        center = (min_vals + max_vals) / 2
        range_vals = max_vals - min_vals
        max_range = range_vals.max() / 2
        return center, max_range

    pred_center, pred_max_range = calculate_limits(pred_vertices)
    gt_center, gt_max_range = calculate_limits(gt_vertices)
    
    # 使用顶点计算出的最大范围设置坐标轴
    max_range = max(pred_max_range, gt_max_range)
    top_lim = [-max_range, max_range]
    side_front_lim = [-max_range, max_range]

    def set_axes_limits(ax, center, max_range):
        # 记录 center 和 max_range 的值以便调试
        # logging.info(f"设置坐标轴限制，center: {center}, max_range: {max_range}")
        
        # 检查输入中是否有 NaN 或 Inf 值，如果有则替换为默认值
        if np.any(np.isnan(center)) or np.any(np.isinf(center)):
            logging.warning("检测到无效的 center 值，将 NaN/Inf 替换为 0。")
            center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(max_range) or np.isinf(max_range) or max_range == 0:
            logging.warning("检测到无效的 max_range 值，将 NaN/Inf 或 0 替换为一个小的正值。")
            max_range = 1.0  # 赋予一个默认值以避免错误

        # 现在安全地设置坐标轴限制
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])

    # Plot predictions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_mesh(ax1, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Top View')
    ax1.view_init(elev=90, azim=-90)  # 设置为顶视图
    set_axes_limits(ax1, pred_center, pred_max_range)
    ax1.set_box_aspect([1, 1, 1])

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_mesh(ax2, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction -  Side View')
    ax2.view_init(elev=0, azim=-90)  # 设置为侧视图
    set_axes_limits(ax2, pred_center, pred_max_range)
    ax2.set_box_aspect([1, 1, 1])

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_mesh(ax3, pred_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Prediction - Front View')
    ax3.view_init(elev=0, azim=0)  # 设置为前视图
    set_axes_limits(ax3, pred_center, pred_max_range)
    ax3.set_box_aspect([1, 1, 1])

    # Plot ground truth
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_mesh(ax4, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Top View')
    ax4.view_init(elev=90, azim=-90)  # 设置为顶视图
    set_axes_limits(ax4, gt_center, gt_max_range)
    ax4.set_box_aspect([1, 1, 1])

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    plot_mesh(ax5, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth -  Side View')
    ax5.view_init(elev=0, azim=-90)  # 设置为侧视图
    set_axes_limits(ax5, gt_center, gt_max_range)
    ax5.set_box_aspect([1, 1, 1])

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_mesh(ax6, gt_vertices, faces_male if genders[idx] > 0.5 else faces_female, 'Ground Truth - Front View')
    ax6.view_init(elev=0, azim=0)  # 设置为前视图
    set_axes_limits(ax6, gt_center, gt_max_range)
    ax6.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_smplx_mesh_comparison_gif(
    pred, gt,
    radar_input=None,          # [T, x, N, 4] or [T, x, Dx, Dy, Dz]
    pred_center=None, gt_center=None,  # [T,3] in WORLD coords
    world_bbox=(3.6/2, 2.4/5.5*2, 2.4/1),  # (wx, wy, wz)
    image_bbox=(36, 24, 24),   # (Bx, By, Bz) in VOXELS
    out_path='mesh_comparison.gif',
    elev=20, azim=120,
    xlim=(-0.5, 0.5), ylim=(-1, 1), zlim=(0, 2),
    interval=100, dpi=100,
    mesh_color_pred='blue', mesh_color_gt='green',
    pc_color='crimson', pc_size=2,
    center_pred_color='orange', center_gt_color='purple',
):
    import os, numpy as np, torch
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    pred, gt = to_np(pred), to_np(gt)
    if pred_center is not None: pred_center = to_np(pred_center)
    if gt_center   is not None: gt_center   = to_np(gt_center)
    T, _, _ = pred.shape

    pc_sel, ten_sel = None, None
    if radar_input is not None:
        r = to_np(radar_input)
        if r.ndim == 4:   # [T, x, N, 4]
            pc_sel  = np.nan_to_num(r[:, -1, :, :3])
        elif r.ndim == 5: # [T, x, Dx, Dy, Dz]
            ten_sel = np.nan_to_num(r[:, -1])

    # --- helpers ---
    def setup_3d(ax):
        ax.cla()
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    def draw_world_bbox_centered(ax, wx, wy, wz, center, color):
        cx, cy, cz = center
        x0, x1 = cx - wx, cx + wx
        y0, y1 = cy - wy, cy + wy
        z0, z1 = cz - wz/2., cz + wz/2.
        C = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a,b in E:
            xs,ys,zs = C[[a,b]].T
            ax.plot(xs,ys,zs,c=color,lw=1)

    # assume tensor world span matches world_bbox: x∈[-wx,wx], y∈[-wy,wy], z∈[0,wz]
    def world_to_image_center(center_w, dims, wx, wy, wz):
        Dx, Dy, Dz = dims
        cx, cy, cz = center_w
        x = (cx + 3.) / 6. * 121
        y = (cy - 0.5) / 5.5 * 111
        z = (cz - 0.0) / 3. * 31
        return np.array([x, y, z], dtype=float)

    def draw_image_bbox_xy_centered(ax, Dx, Dy, Bx, By, cx, cy, color):
        x0, x1 = cx - Bx/2., cx + Bx/2.
        y0, y1 = cy - By/2., cy + By/2.
        x0, x1 = max(0,x0), min(Dx,x1)
        y0, y1 = max(0,y0), min(Dy,y1)
        ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0], c=color, lw=1)

    def draw_image_bbox_xz_centered(ax, Dx, Dz, Bx, Bz, cx, cz, color):
        x0, x1 = cx - Bx/2., cx + Bx/2.
        z0, z1 = cz - Bz/2., cz + Bz/2.
        x0, x1 = max(0,x0), min(Dx,x1)
        z0, z1 = max(0,z0), min(Dz,z1)
        ax.plot([x0,x1,x1,x0,x0],[z0,z0,z1,z1,z0], c=color, lw=1)

    # --- figure / axes ---
    if ten_sel is not None:
        fig = plt.figure(figsize=(20,5))
        ax_pred = fig.add_subplot(141, projection='3d')
        ax_gt   = fig.add_subplot(142, projection='3d')
        ax_xy   = fig.add_subplot(143)   # XY mean over z
        ax_xz   = fig.add_subplot(144)   # XZ mean over y
        Dx, Dy, Dz = ten_sel.shape[1:]
        Bx, By, Bz = image_bbox
    else:
        fig = plt.figure(figsize=(10,5))
        ax_pred = fig.add_subplot(121, projection='3d')
        ax_gt   = fig.add_subplot(122, projection='3d')
        ax_xy = ax_xz = None

    wx, wy, wz = world_bbox

    def update(t):
        setup_3d(ax_pred); setup_3d(ax_gt)
        ax_pred.set_title('Predicted Mesh'); ax_gt.set_title('Ground Truth Mesh')

        pf, gf = pred[t], gt[t]
        ax_pred.scatter(pf[:,0],pf[:,1],pf[:,2], s=1, c=mesh_color_pred)
        ax_gt.scatter(  gf[:,0],gf[:,1],gf[:,2], s=1, c=mesh_color_gt)

        if pred_center is not None:
            ax_pred.scatter(pred_center[t,0],pred_center[t,1],pred_center[t,2], s=40, c=center_pred_color, marker='x')
            draw_world_bbox_centered(ax_pred, wx, wy, wz, pred_center[t], center_pred_color)
        if gt_center is not None:
            ax_gt.scatter(gt_center[t,0],gt_center[t,1],gt_center[t,2], s=40, c=center_gt_color, marker='x')
            draw_world_bbox_centered(ax_gt, wx, wy, wz, gt_center[t], center_gt_color)

        if pc_sel is not None:
            pc = pc_sel[t]
            if pc.size:
                ax_pred.scatter(pc[:,0],pc[:,1],pc[:,2], s=pc_size, c=pc_color, alpha=0.7)
                ax_gt.scatter(  pc[:,0],pc[:,1],pc[:,2], s=pc_size, c=pc_color, alpha=0.7)

        if ten_sel is not None:
            ten = ten_sel[t]                 # [Dx,Dy,Dz]
            ax_xy.cla(); ax_xz.cla()
            ax_xy.set_title('Radar XY (mean z)')
            ax_xz.set_title('Radar XZ (mean y)')

            img_xy = ten.mean(axis=2)        # [Dx,Dy] -> show [Dy,Dx]
            img_xz = ten.mean(axis=1)        # [Dx,Dz] -> show [Dz,Dx]
            ax_xy.imshow(img_xy.T, origin='lower', aspect='auto')
            ax_xz.imshow(img_xz.T, origin='lower', aspect='auto')

            # draw TWO centered boxes on each projection for comparison
            if pred_center is not None:
                c_img = world_to_image_center(pred_center[t], (Dx,Dy,Dz), wx, wy, wz)
                draw_image_bbox_xy_centered(ax_xy, Dx, Dy, Bx, By, c_img[0], c_img[1], center_pred_color)
                draw_image_bbox_xz_centered(ax_xz, Dx, Dz, Bx, Bz, c_img[0], c_img[2], center_pred_color)
            if gt_center is not None:
                c_img = world_to_image_center(gt_center[t], (Dx,Dy,Dz), wx, wy, wz)
                draw_image_bbox_xy_centered(ax_xy, Dx, Dy, Bx, By, c_img[0], c_img[1], center_gt_color)
                draw_image_bbox_xz_centered(ax_xz, Dx, Dz, Bx, Bz, c_img[0], c_img[2], center_gt_color)

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    writer = PillowWriter(fps=max(1, 1000 // max(1, interval)))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
