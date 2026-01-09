import os
import json
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
from .Train_and_model_plotting_3D_mesh import save_human_mesh, plot_comparison, plot_smplx_mesh_comparison_gif
from .Train_and_model_loss import batch_rodrigues, compute_joints_from_vertices

# ===== DDP-minimal helpers =====
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, Subset  # for rank-0 sequential GIF pass

def unwrap_model(m):
    return m.module if isinstance(m, (nn.DataParallel, DDP)) else m

def is_main_process():
    # default to True when not using DDP
    return int(os.environ.get("RANK", "0")) == 0
# =======================================


def compute_average_vertex_error(pred_vertices, gt_vertices):
    # L2 distance between predicted and ground-truth vertices
    return torch.norm(pred_vertices - gt_vertices, dim=-1).mean(-1)

def compute_average_joint_localization_error(pred_joints, gt_joints):
    return torch.norm(pred_joints - gt_joints, dim=-1).mean(-1)

def compute_average_joint_rotation_error(pred_pose, gt_pose):
    pred_rot = batch_rodrigues(pred_pose.reshape(-1, 3)).reshape(pred_pose.shape[0], -1, 3, 3)
    gt_rot = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(gt_pose.shape[0], -1, 3, 3)
    cos_theta = ((pred_rot * gt_rot).sum(dim=(-1, -2)) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle_error_rad = torch.acos(cos_theta)
    angle_error_deg = torch.rad2deg(angle_error_rad)
    return angle_error_deg.mean(-1)

def compute_mesh_localization_error(pred_trans, gt_trans):
    # Compare predicted root translation to mean GT vertex position (centroid)
    return torch.norm(pred_trans - gt_trans, dim=-1) * 1000

def compute_gender_prediction_accuracy(pred_genders, gt_genders):
    # print(pred_genders, gt_genders)  # preserved (commented in your latest)
    return (pred_genders > 0.5).eq(gt_genders.cuda()).float()

def evaluate(model, dataloader, result_path, plot_gif = True, modality = 'radar_points'):
    model.eval()
    device = next(unwrap_model(model).parameters()).device  # derive device even under DDP

    # totally 50 actions
    total_samples = [0] * 50
    vertex_total, joint_loc_total, joint_rot_total = [0] * 50, [0] * 50, [0] * 50
    mesh_loc_total, gender_acc_total = [0] * 50, [0] * 50

    # choose one random sample index from the WHOLE dataset (global order)
    indices_to_save = random.sample(range(len(dataloader.dataset)), min(1, len(dataloader.dataset)))
    # compute the two consecutive global batches for GIF window
    gif_start_batch = indices_to_save[0] // dataloader.batch_size
    gif_batches = {gif_start_batch, gif_start_batch + 1}

    # -------- Main distributed metrics pass (unchanged) --------
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            radar_PC = batch[modality].cuda()
            
            gt = batch['parameter']
            betas_gt = gt['betas'][:, :10].cuda()
            pose_gt = gt['pose_body'].cuda()
            root_orient_gt = gt['root_orient'].cuda()
            trans_gt = gt['trans'].cuda()
            gender_gt = gt['gender'].cuda()
            
            indicator = batch['indicator']
            

            pred_betas, pred_pose, pred_root_orient, pred_trans, gender_pred, _ = model(radar_PC)
            vertices_pred = unwrap_model(model).get_smplx_output(
                pred_betas, pred_pose, pred_root_orient, pred_trans, gender_pred
            ).squeeze(1)
            vertices_gt = unwrap_model(model).get_smplx_output(
                betas_gt, pose_gt, root_orient_gt, trans_gt, gender_gt).squeeze(1)

            pred_joints = compute_joints_from_vertices(model, vertices_pred, gender_gt)
            gt_joints   = compute_joints_from_vertices(model, vertices_gt, gender_gt)

            bsz = radar_PC.size(0)
            for id in range(bsz):
                act_id = indicator[id][1]

                vertex_total[act_id-1]    += compute_average_vertex_error(vertices_pred, vertices_gt)[id].item()
                joint_loc_total[act_id-1] += compute_average_joint_localization_error(pred_joints, gt_joints)[id].item()
                joint_rot_total[act_id-1] += compute_average_joint_rotation_error(pred_pose, pose_gt)[id].item()
                mesh_loc_total[act_id-1]  += compute_mesh_localization_error(pred_trans, trans_gt)[id].item()
                gender_acc_total[act_id-1] += compute_gender_prediction_accuracy(gender_pred, gender_gt)[id].item()
                total_samples[act_id-1]   += 1

            # Per-sample static comparison — save only on rank 0 (preserved)
            if is_main_process():
                core = unwrap_model(model)
                for i in range(bsz):
                    index = batch_idx * dataloader.batch_size + i
                    if index in indices_to_save:
                        comparison_plot_filename = os.path.join(result_path, f'frame_{batch_idx}_{i}_comparison.png')
                        plot_comparison(
                            vertices_pred[i].detach().cpu().numpy(),
                            vertices_gt[i].detach().cpu().numpy(),
                            core.faces_male, core.faces_female,
                            gender_gt.detach().cpu().numpy(),
                            i, comparison_plot_filename
                        )

    # -------- Rank-0 sequential re-pass: build GIF from TRUE consecutive global batches --------
    if is_main_process() and plot_gif:
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

        bs = dataloader.batch_size
        ds = dataloader.dataset
        N = len(ds)

        # gif_batches: iterable of global batch indices you want (e.g., {10, 11, 12})
        # Build the exact sample indices those batches cover.
        selected_indices = []
        for b in sorted(gif_batches):
            start = b * bs
            end = min(start + bs, N)
            if start >= N:  # skip out-of-range batches quietly
                continue
            selected_indices.extend(range(start, end))

        if not selected_indices:
            print("[GIF] No valid indices selected; skipping GIF generation.")
        else:
            # Keep order identical to the original global order
            subset = Subset(ds, selected_indices)
            seq_loader = DataLoader(
                subset,
                batch_size=bs,
                shuffle=False,
                sampler=SequentialSampler(subset),   # iterate exactly in ascending order
                num_workers=0,
                pin_memory=True,
            )

            gif_preds, gif_gts, gif_radar_PC, gif_preds_trans, gif_gts_trans = [], [], [], [], []
            unwrap_model(model).eval()
            with torch.inference_mode():
                for batch in seq_loader:
                    radar_PC = batch[modality].to(device, non_blocking=True)
                    gt = batch['parameter']
                    betas_gt = gt['betas'][:, :10].to(device, non_blocking=True)
                    pose_gt = gt['pose_body'].to(device, non_blocking=True)
                    root_orient_gt = gt['root_orient'].to(device, non_blocking=True)
                    trans_gt = gt['trans'].to(device, non_blocking=True)
                    gender_gt = gt['gender'].to(device, non_blocking=True)

                    vertices_gt = unwrap_model(model).get_smplx_output(
                        betas_gt, pose_gt, root_orient_gt, trans_gt, gender_gt
                    ).squeeze(1)

                    pred_betas, pred_pose, pred_root_orient, pred_trans, gender_pred, _ = model(radar_PC)
                    vertices_pred = unwrap_model(model).get_smplx_output(
                        pred_betas, pred_pose, pred_root_orient, pred_trans, gender_pred
                    ).squeeze(1)

                    pred_joints = compute_joints_from_vertices(model, vertices_pred, gender_gt) / 1000.0
                    gt_joints   = compute_joints_from_vertices(model, vertices_gt,   gender_gt) / 1000.0

                    gif_preds.append(vertices_pred.detach().cpu())
                    gif_gts.append(vertices_gt.detach().cpu())
                    gif_radar_PC.append(radar_PC.detach().cpu())
                    gif_gts_trans.append(gt_joints[:, 0].detach().cpu())
                    gif_preds_trans.append(pred_joints[:, 0].detach().cpu())

            if gif_preds:
                gif_preds = torch.cat(gif_preds, dim=0) / 1000.0
                gif_gts   = torch.cat(gif_gts,   dim=0) / 1000.0
                gif_radar_PC = torch.cat(gif_radar_PC, dim=0).numpy()
                gif_preds_trans = torch.cat(gif_preds_trans, dim=0).numpy()
                gif_gts_trans   = torch.cat(gif_gts_trans,   dim=0).numpy()

                # keep only the first 20 frames for the GIF (optional)
                max_frames = 100
                gif_radar_PC   = gif_radar_PC[:max_frames]
                gif_preds_trans = gif_preds_trans[:max_frames]
                gif_gts_trans   = gif_gts_trans[:max_frames]

                zlim = (-1., 2.) if modality == 'radar_points' else (0., 3.)
                first_batch_idx = min(gif_batches) if len(gif_batches) > 0 else 0
                comparison_gif_filename = os.path.join(result_path, f'frame_{first_batch_idx}_comparison.gif')

                plot_smplx_mesh_comparison_gif(
                    gif_preds.numpy()[:max_frames], gif_gts.numpy()[:max_frames],
                    radar_input=gif_radar_PC, pred_center=gif_preds_trans, gt_center=gif_gts_trans,
                    out_path=comparison_gif_filename,
                    elev=20, azim=120, xlim=(-1.5, 1.5), ylim=(1.5, 3.5),
                    zlim=zlim
                )



    # -------- Aggregate results --------

    overall_results = {
        'mean_vertex_error': vertex_total,
        'mean_joint_localization_error': joint_loc_total,
        'mean_joint_rotation_error': joint_rot_total,
        'mean_mesh_localization_error': mesh_loc_total,
        'mean_gender_accuracy': gender_acc_total,
        'total_samples': total_samples
    }
    return overall_results





def results_organize(
    sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total,
    summary_csv_path, best_csv_path, epoch: int, round_fp: int = 4
):
    # arrays
    S_v = np.asarray(sum_vertex, dtype=float)
    S_l = np.asarray(sum_jloc,   dtype=float)
    S_r = np.asarray(sum_jrot,   dtype=float)
    S_m = np.asarray(sum_mloc,   dtype=float)
    S_g = np.asarray(sum_gacc,   dtype=float)
    N   = np.asarray(n_total,    dtype=float)
    assert N.size == 50, "Expect 50 actions"

    # per-action averages (for best CSV columns)
    with np.errstate(invalid="ignore", divide="ignore"):
        A_v, A_l, A_r, A_m, A_g = S_v/N, S_l/N, S_r/N, S_m/N, S_g/N

    # helpers
    def gsum(arr, a, b): return float(np.nansum(arr[a:b+1]))
    def micro_mean(S, a, b):
        c = gsum(N, a, b)
        return float(gsum(S, a, b) / c) if c > 0 else np.nan

    # ---- SUMMARY CSV (no counts, only 5 metrics as micro means) ----
    row = {
        "epoch": int(epoch),

        "vertex_mean_1_35":  micro_mean(S_v, 0, 34),
        "vertex_mean_36_40": micro_mean(S_v, 35, 39),
        "vertex_mean_41_50": micro_mean(S_v, 40, 49),
        "vertex_mean_1_50":  micro_mean(S_v, 0, 49),

        "jloc_mean_1_35":  micro_mean(S_l, 0, 34),
        "jloc_mean_36_40": micro_mean(S_l, 35, 39),
        "jloc_mean_41_50": micro_mean(S_l, 40, 49),
        "jloc_mean_1_50":  micro_mean(S_l, 0, 49),

        "jrot_mean_1_35":  micro_mean(S_r, 0, 34),
        "jrot_mean_36_40": micro_mean(S_r, 35, 39),
        "jrot_mean_41_50": micro_mean(S_r, 40, 49),
        "jrot_mean_1_50":  micro_mean(S_r, 0, 49),

        "mloc_mean_1_35":  micro_mean(S_m, 0, 34),
        "mloc_mean_36_40": micro_mean(S_m, 35, 39),
        "mloc_mean_41_50": micro_mean(S_m, 40, 49),
        "mloc_mean_1_50":  micro_mean(S_m, 0, 49),

        "gacc_mean_1_35":  micro_mean(S_g, 0, 34),
        "gacc_mean_36_40": micro_mean(S_g, 35, 39),
        "gacc_mean_41_50": micro_mean(S_g, 40, 49),
        "gacc_mean_1_50":  micro_mean(S_g, 0, 49),
    }
    os.makedirs(os.path.dirname(summary_csv_path) or ".", exist_ok=True)
    df_sum = pd.read_csv(summary_csv_path) if os.path.exists(summary_csv_path) else pd.DataFrame(columns=row.keys())
    df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
    df_sum["epoch"] = df_sum["epoch"].astype(int)
    for c in df_sum.columns:
        if c != "epoch":
            df_sum[c] = pd.to_numeric(df_sum[c], errors="coerce").round(round_fp)
    df_sum.to_csv(summary_csv_path, index=False, float_format=f"%.{round_fp}f")

    # ---- BEST CSV (per-action) — trigger by improved vertex_mean_1_50 (micro) ----
    act_cols = [str(i) for i in range(1, 51)]
    extra    = ["mean_1_35", "mean_36_40", "mean_41_50", "mean_1_50"]
    cols     = ["metric"] + act_cols + extra

    cur_indicator = row["vertex_mean_1_50"]  # micro mean over 1..50 vertex error

    def group_cols_from_sums(S):  # group micro means for this epoch
        return [micro_mean(S, 0, 34), micro_mean(S, 35, 39),
                micro_mean(S, 40, 49), micro_mean(S, 0, 49)]

    def build_best_df():
        rows = [
            ["avg_vertex_error"]    + list(A_v) + group_cols_from_sums(S_v),
            ["avg_joint_loc_error"] + list(A_l) + group_cols_from_sums(S_l),
            ["avg_joint_rot_error"] + list(A_r) + group_cols_from_sums(S_r),
            ["avg_mesh_loc_error"]  + list(A_m) + group_cols_from_sums(S_m),
            ["avg_gender_accuracy"] + list(A_g) + group_cols_from_sums(S_g),
        ]
        dfb = pd.DataFrame(rows, columns=cols)
        ncols = [c for c in dfb.columns if c != "metric"]
        dfb[ncols] = dfb[ncols].apply(pd.to_numeric, errors="coerce").round(round_fp)
        return dfb

    update_best = True
    if os.path.exists(best_csv_path):
        prev = pd.read_csv(best_csv_path)
        prev_indicator = float(prev.loc[prev["metric"] == "avg_vertex_error", "mean_1_50"].iloc[0])
        update_best = cur_indicator < prev_indicator

    if update_best:
        os.makedirs(os.path.dirname(best_csv_path) or ".", exist_ok=True)
        build_best_df().to_csv(best_csv_path, index=False, float_format=f"%.{round_fp}f")

    # ---- return only the 5 current micro means over 1..50 ----
    total = float(np.nansum(N))
    return {
        "vertex": float(np.nansum(S_v) / total) if total else np.nan,
        "jloc":   float(np.nansum(S_l) / total) if total else np.nan,
        "jrot":   float(np.nansum(S_r) / total) if total else np.nan,
        "mloc":   float(np.nansum(S_m) / total) if total else np.nan,
        "gacc":   float(np.nansum(S_g) / total) if total else np.nan,
    }
    
    
    
# ===== Optional joint plotting utils (unchanged) =====
import matplotlib.pyplot as plt

def get_smplx_joint_connections(num_joints):
    # Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    connections = []
    for child, parent in enumerate(parents):
        if parent != -1 and child < num_joints and parent < num_joints:
            connections.append((parent, child))
    return connections

def plot_joints_side_by_side(pred_joints, gt_joints, save_path):
    # Scale joints to meters and normalize to [-1, 1] for each axis
    pred_joints = pred_joints / 1000.0
    gt_joints = gt_joints / 1000.0

    def normalize(joints):
        min_vals = joints.min(axis=0)
        max_vals = joints.max(axis=0)
        # Avoid division by zero
        scale = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        return 2 * (joints - min_vals) / scale - 1

    pred_joints = normalize(pred_joints)
    gt_joints = normalize(gt_joints)

    fig = plt.figure(figsize=(10, 5))
    connections = get_smplx_joint_connections(pred_joints.shape[0])

    # Predicted joints
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title('Predicted Joints')
    ax1.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], c='r')
    for a, b in connections:
        ax1.plot([pred_joints[a, 0], pred_joints[b, 0]],
                 [pred_joints[a, 1], pred_joints[b, 1]],
                 [pred_joints[a, 2], pred_joints[b, 2]], c='r')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])
    ax1.view_init(elev=20, azim=-70)

    # Ground-truth joints
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title('GT Joints')
    ax2.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2], c='g')
    for a, b in connections:
        ax2.plot([gt_joints[a, 0], gt_joints[b, 0]],
                 [gt_joints[a, 1], gt_joints[b, 1]],
                 [gt_joints[a, 2], gt_joints[b, 2]], c='g')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    ax2.view_init(elev=20, azim=-70)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
