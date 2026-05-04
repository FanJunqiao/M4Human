"""
M4Human multi-GPU training entry point.
Run with:  torchrun --nproc_per_node=<N> main1_multigpu.py [--config config.yaml]
"""

import os
import argparse
import json
import logging
import yaml
from datetime import datetime, timedelta
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from mmwave_models.Point_models.p4Transformer_encode import P4Transformer
from mmwave_models.Tensor_models.RTmesh.tpn_mul_attn5 import Simple3DConvModelWithTripleCNNFPNAndAttention
from mmwave_models.Tensor_models.retr_models.retr import RETR

from dataset.m4human_dataset import RF3DPoseDataset, ToTensor
from sources.Train_and_model_loss import combined_loss
from sources.evaluation_module_pc_multigpu import evaluate, results_organize


# ===========================================================================
# Config loader
# ===========================================================================

def load_config(path: str = "config.yaml") -> dict:
    """
    Read config.yaml and flatten it into the dict layout expected by
    final_training().  Nested YAML keys are mapped to the same flat names
    that were previously in the hardcoded CONFIG dict.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    lw = raw["train"]["loss_weights"]
    cfg = {
        # paths
        "cached_root":        raw["paths"]["cached_root"],
        "smplx_paths":        raw["paths"]["smplx"],
        # model
        "model":              raw["model"]["name"],
        "modality":           raw["model"]["modality"],
        # training
        "batch_size":         raw["train"]["batch_size"],
        "epochs":             raw["train"]["epochs"],
        "patience":           raw["train"]["patience"],
        "lr":                 raw["train"]["lr"],
        "lr_decay_step":      raw["train"]["lr_decay_step"],
        "lr_decay_gamma":     raw["train"]["lr_decay_gamma"],
        "l2_lambda":          raw["train"]["l2_lambda"],
        "betas_weight":       lw["betas"],
        "pose_body_weight":   lw["pose_body"],
        "root_orient_weight": lw["root_orient"],
        "trans_weight":       lw["trans"],
        "vertices_weight":    lw["vertices"],
        "gender_weight":      lw["gender"],
        # eval
        "test_mode":          raw["eval"]["test_mode"],
        "test_model_path":    raw["eval"]["test_model_path"],
        "plot_gif":           raw["eval"]["plot_gif"],
        # dataset split
        "scale":              raw["dataset"]["scale"],
        "split":              raw["dataset"]["split"],
    }
    return cfg


# ===========================================================================
# DDP helpers
# ===========================================================================

def setup_ddp():
    """Initialise the NCCL process group and bind this process to its GPU."""
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def unwrap_model(m):
    """Return the underlying module whether or not it is wrapped in DDP/DataParallel."""
    return m.module if isinstance(m, (nn.DataParallel, DDP)) else m


# ===========================================================================
# Logging / path helpers
# ===========================================================================

def get_unique_exp_path(base_path: str = "experiments") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(base_path, f"exp_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def setup_logger(log_path: str) -> logging.Logger:
    """Attach file + console handlers on rank-0 only to avoid duplicate log lines."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    if is_main_process():
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


def ensure_directory_exists(directory: str):
    os.makedirs(directory, exist_ok=True)


# ===========================================================================
# Training / validation epoch
# ===========================================================================

def run_epoch(
    model,
    dataloader,
    criterion1,
    criterion2,
    gender_criterion,
    optimizer=None,
    gradient_clip: bool = False,
    log_gradient_norm: bool = False,
    modality: str = "radar_points",
):
    """
    Run one full pass over dataloader.
    Pass optimizer=None for eval mode (no backward).
    Returns (sum_of_losses, number_of_batches).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    loss_dict_sum = {k: 0 for k in ("betas", "pose_body", "root_orient", "trans", "vertices")}
    avg_gradient_norm = 0.0
    progress_bar = tqdm(dataloader, desc="Training" if is_train else "Validation")

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in progress_bar:
            radar_input   = batch[modality].cuda()
            gt            = batch["parameter"]
            gt_betas      = gt["betas"][:, :10].cuda()
            gt_pose_body  = gt["pose_body"].cuda()
            gt_root_orient = gt["root_orient"].cuda()
            gt_trans      = gt["trans"].cuda()
            gt_genders    = gt["gender"].cuda()
            gt_joint_root = batch["joints_root"][:, 0].cuda()

            if is_train:
                optimizer.zero_grad()

            pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred, center_pred = \
                model(radar_input)

            # Vertex-level loss is disabled; GT/pred vertices kept as None
            pred_vertices, gt_vertices = None, None

            loss, loss_components = combined_loss(
                pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices,
                gt_betas,   gt_pose_body,   gt_root_orient,   gt_trans,   gt_vertices,
                gender_pred, gt_genders, gt_joint_root,
                criterion1, criterion2, gender_criterion,
                unwrap_model(model), pred_center=center_pred,
            )
            losses.append(loss.item())

            if is_train:
                loss.backward()

                if log_gradient_norm:
                    total_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                    avg_gradient_norm = (avg_gradient_norm * (len(losses) - 1) + total_norm) / len(losses)
                    # Skip step if gradient explodes
                    if total_norm > 3.2 * avg_gradient_norm:
                        print(f"Gradient Norm: {total_norm:.4f}, Avg: {avg_gradient_norm:.4f} — skipped")
                        continue

                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)

                optimizer.step()

            for key in loss_dict_sum:
                loss_dict_sum[key] += loss_components[key].item()

            avg_loss_dict = {k: f"{loss_dict_sum[k] / len(losses):.4f}" for k in loss_dict_sum}
            progress_bar.set_postfix(loss=loss.item(), **avg_loss_dict)

    torch.cuda.empty_cache()
    return sum(losses), len(losses)


# ===========================================================================
# Main training function
# ===========================================================================

def final_training(config: dict, exp_path: str, logger: logging.Logger):

    load_save = True

    # --- Datasets ---
    train_dataset = RF3DPoseDataset(
        [], cache_dir=config["cached_root"], split="train",
        transform=transforms.Compose([ToTensor()]),
        load_save=load_save, main_modality=config["modality"],
        scale_id=config["scale"], split_id=config["split"],
    )
    val_dataset = RF3DPoseDataset(
        [], cache_dir=config["cached_root"], split="val",
        transform=transforms.Compose([ToTensor()]),
        load_save=load_save, main_modality=config["modality"],
        scale_id=config["scale"], split_id=config["split"],
    )
    test_dataset = RF3DPoseDataset(
        [], cache_dir=config["cached_root"], split="test",
        transform=transforms.Compose([ToTensor()]),
        load_save=load_save, main_modality=config["modality"],
        scale_id=config["scale"], split_id=config["split"],
    )

    # --- DataLoaders with DistributedSampler ---
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        sampler=DistributedSampler(train_dataset, shuffle=True),
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        sampler=DistributedSampler(val_dataset, shuffle=False),
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        sampler=DistributedSampler(test_dataset, shuffle=False),
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1,
    )

    # --- Model selection ---
    if config["modality"] == "radar_points":
        model = P4Transformer(
            config["smplx_paths"],
            radius=0.3, nsamples=32, spatial_stride=32,
            temporal_kernel_size=3, temporal_stride=1,
            emb_relu=False,
            dim=1024, depth=10, heads=8, dim_head=256,
            mlp_dim=2048, num_classes=17 * 3, dropout1=0.0, dropout2=0.0,
        ).cuda()
    elif config["modality"] == "rawImage_XYZ" and config["model"] == "RT-Mesh":
        model = Simple3DConvModelWithTripleCNNFPNAndAttention(config["smplx_paths"]).cuda()
    elif config["modality"] == "rawImage_XYZ" and config["model"] == "RETR":
        model = RETR(
            smplx_model_paths=config["smplx_paths"], task="SEG",
            topk=64, in_channels=[4 * 31, 4 * 121],
        ).cuda()

    # -----------------------------------------------------------------------
    # Training branch
    # -----------------------------------------------------------------------
    if not config["test_mode"]:
        logger.info("Experiment Configuration:\n" + json.dumps(config, indent=4))

        model = DDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

        unwrap_model(model).loss_weights = {
            "betas":       config["betas_weight"],
            "pose_body":   config["pose_body_weight"],
            "root_orient": config["root_orient_weight"],
            "trans":       config["trans_weight"],
            "vertices":    config["vertices_weight"],
            "gender":      config["gender_weight"],
        }

        # Separate weight-decay and no-decay param groups (bias / norm layers)
        decay_params, no_decay_params = [], []
        for name, param in unwrap_model(model).named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.Adam(
            [
                {"params": decay_params,    "weight_decay": config["l2_lambda"]},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config["lr"],
        )
        scheduler = StepLR(optimizer, step_size=config["lr_decay_step"], gamma=config["lr_decay_gamma"])

        criterion1       = nn.L1Loss()
        criterion2       = nn.MSELoss()
        gender_criterion = nn.BCELoss()

        best_test_mve    = float("inf")
        patience_counter = 0
        gradient_clip    = False
        log_gradient_norm = False

        for epoch in range(config["epochs"]):
            if epoch > 5:
                # Enable gradient norm logging after warm-up to detect explosions
                gradient_clip     = False
                log_gradient_norm = True

            # Ensure different shuffles per epoch across ranks
            for loader in (train_loader, val_loader, test_loader):
                loader.sampler.set_epoch(epoch)

            train_sum, train_n = run_epoch(
                model, train_loader, criterion1, criterion2, gender_criterion,
                optimizer, gradient_clip=gradient_clip,
                log_gradient_norm=log_gradient_norm, modality=config["modality"],
            )
            val_sum, val_n = run_epoch(
                model, val_loader, criterion1, criterion2, gender_criterion,
                modality=config["modality"],
            )

            # Aggregate losses from all ranks
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            packed = torch.tensor(
                [train_sum, train_n, val_sum, val_n],
                device=device, dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            train_sum_g, train_n_g, val_sum_g, val_n_g = packed.tolist()

            train_loss = train_sum_g / max(1.0, train_n_g)
            val_loss   = val_sum_g   / max(1.0, val_n_g)

            scheduler.step()

            if is_main_process():
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6e}"
                )

            # --- Per-epoch evaluation on test set ---
            result_path = os.path.join(exp_path, f"test_epoch_{epoch + 1}")
            if is_main_process():
                ensure_directory_exists(result_path)

            local_metrics = evaluate(
                unwrap_model(model), test_loader, result_path,
                modality=config["modality"], plot_gif=config["plot_gif"],
            )

            # Aggregate evaluation metrics from all ranks
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            vec = torch.tensor(
                [
                    local_metrics["mean_vertex_error"],
                    local_metrics["mean_joint_localization_error"],
                    local_metrics["mean_joint_rotation_error"],
                    local_metrics["mean_mesh_localization_error"],
                    local_metrics["mean_gender_accuracy"],
                    local_metrics["total_samples"],
                ],
                device=device, dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total = vec.tolist()

            csv_file      = os.path.join(exp_path, "results.csv")
            csv_best_file = os.path.join(exp_path, "results_best.csv")
            if is_main_process():
                global_results = results_organize(
                    sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total,
                    epoch=epoch, summary_csv_path=csv_file, best_csv_path=csv_best_file,
                )
                logger.info(json.dumps(global_results, indent=2))

            # --- Checkpoint on improvement ---
            if is_main_process():
                current_test_mve = global_results["vertex"]
                if current_test_mve < best_test_mve:
                    best_test_mve    = current_test_mve
                    patience_counter = 0
                    torch.save(
                        unwrap_model(model).state_dict(),
                        os.path.join(exp_path, f"best_model_epoch{epoch // 5}.pth"),
                    )
                    logger.info(f"Model improved on test MVE ({current_test_mve:.4f}). Saved best model.")
                else:
                    patience_counter += 1
                    if patience_counter >= config["patience"]:
                        logger.info("Early stopping triggered.")
                        break

        if is_main_process():
            logger.info("Training complete.")

    # -----------------------------------------------------------------------
    # Test-only branch
    # -----------------------------------------------------------------------
    else:
        logger.info("Experiment Configuration:\n" + json.dumps(config, indent=4))

        if config["test_model_path"]:
            model.load_state_dict(torch.load(config["test_model_path"]))

        model = DDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )
        unwrap_model(model).eval()
        unwrap_model(model).loss_weights = {
            "betas":       config["betas_weight"],
            "pose_body":   config["pose_body_weight"],
            "root_orient": config["root_orient_weight"],
            "trans":       config["trans_weight"],
            "vertices":    config["vertices_weight"],
            "gender":      config["gender_weight"],
        }

        result_path = os.path.join(exp_path, "test_epoch")
        if is_main_process():
            ensure_directory_exists(result_path)

        local_metrics = evaluate(
            unwrap_model(model), test_loader, result_path,
            modality=config["modality"], plot_gif=config["plot_gif"],
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        vec = torch.tensor(
            [
                local_metrics["mean_vertex_error"],
                local_metrics["mean_joint_localization_error"],
                local_metrics["mean_joint_rotation_error"],
                local_metrics["mean_mesh_localization_error"],
                local_metrics["mean_gender_accuracy"],
                local_metrics["total_samples"],
            ],
            device=device, dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total = vec.tolist()

        csv_file      = os.path.join(exp_path, "results.csv")
        csv_best_file = os.path.join(exp_path, "results_best.csv")
        if is_main_process():
            global_results = results_organize(
                sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total,
                epoch=0, summary_csv_path=csv_file, best_csv_path=csv_best_file,
            )
            logger.info(json.dumps(global_results, indent=2))


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    local_rank = setup_ddp()

    EXP_PATH = get_unique_exp_path(base_path="experiments")
    if is_main_process():
        ensure_directory_exists(EXP_PATH)

    LOGGER = setup_logger(os.path.join(EXP_PATH, "train_test.log"))
    CONFIG = load_config(args.config)

    final_training(CONFIG, EXP_PATH, LOGGER)
    cleanup_ddp()