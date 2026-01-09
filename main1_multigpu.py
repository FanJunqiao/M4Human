import os
import json
import logging
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist  # [DDP]
from torch.nn.parallel import DistributedDataParallel as DDP  # [DDP]
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # [DDP]
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

from mmwave_models.Point_models.p4Transformer_encode import P4Transformer
from mmwave_models.Point_models.p4Mamba_encode import P4Mamba
from mmwave_models.Tensor_models.RFmesh.tpn_mul_attn5 import Simple3DConvModelWithTripleCNNFPNAndAttention
from mmwave_models.Tensor_models.retr_models.retr import RETR  

from dataset.dataset_mmMesh2 import RF3DPoseDataset, ToTensor
from sources.Train_and_model_loss import combined_loss
from sources.evaluation_module_pc_multigpu import evaluate, results_organize  # ← move your evaluate() function to a separate module for reuse


# ====================
# DDP helpers (only for DDP; no other logic changed)
# ====================
def setup_ddp():
    
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # .cuda() will use this device
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def unwrap_model(m):
    return m.module if isinstance(m, (nn.DataParallel, DDP)) else m

def load_weights_dp_aware(model, path):
    state = torch.load(path, map_location="cpu")
    try:
        unwrap_model(model).load_state_dict(state, strict=True)
    except RuntimeError:
        new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        unwrap_model(model).load_state_dict(new_state, strict=True)


# ====================
# Utility Functions (unchanged, except logging handlers only on rank0)
# ====================
def get_unique_exp_path(base_path='experiments'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_path = os.path.join(base_path, f'exp_{timestamp}')
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # to avoid duplicated logs under DDP, only rank0 attaches handlers
    if is_main_process():
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ====================
# Unified Training + Testing Loop
# ====================

def run_epoch(model, dataloader, criterion1, criterion2, gender_criterion, optimizer=None, gradient_clip=False, log_gradient_norm=False, modality = 'radar_points'):
    # unchanged signature and core behavior; .cuda() still used (device already set by setup_ddp)
    is_train = optimizer is not None
    avg_gradient_norm = 0
    model.train() if is_train else model.eval()
    losses = []
    loss_dict_sum = {
        'betas': 0, 'pose_body': 0, 'root_orient': 0,
        'trans': 0, 'vertices': 0,
    }
    progress_bar = tqdm(dataloader, desc="Training" if is_train else "Validation")

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in progress_bar:
            radar_input = batch[modality].cuda()
            gt_bbbox = batch['bbbox']
            gt = batch['parameter']
            gt_betas = gt['betas'][:, :10].cuda()
            gt_pose_body = gt['pose_body'].cuda()
            gt_root_orient = gt['root_orient'].cuda()
            gt_trans = gt['trans'].cuda()
            gt_genders = gt['gender'].cuda()
            gt_joint_root = batch['joints_root'][:,0].cuda()
            
            if is_train:
                optimizer.zero_grad()

            pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred, center_pred = model(radar_input)
            # pred_vertices = unwrap_model(model).get_smplx_output(pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred)
            # gt_vertices = unwrap_model(model).get_smplx_output(gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_genders)
            pred_vertices = None
            gt_vertices = None
            
            loss, loss_components  = combined_loss(
                pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices,
                gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices,
                gender_pred, gt_genders, gt_joint_root, criterion1, criterion2, gender_criterion, unwrap_model(model), pred_center=center_pred,
            )
            losses.append(loss.item())

            if is_train:
                loss.backward()

                # Compute and log gradient norm
                if log_gradient_norm:
                    total_norm = 0
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    avg_gradient_norm = (avg_gradient_norm * (len(losses) - 1) + total_norm) / len(losses)
                    
                    if total_norm > 3.2*avg_gradient_norm:
                        print(f"Gradient Norm: {total_norm:.4f}, Avg Gradient Norm: {avg_gradient_norm:.4f}")
                        continue

                if gradient_clip:
                    # Gradient clipping: clip gradients to max norm 1.0 (you can tune this)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
                optimizer.step()

            

            for key in loss_dict_sum:
                loss_dict_sum[key] += loss_components[key].item()

            avg_loss_dict = {k: f"{loss_dict_sum[k] / len(losses):.4f}" for k in loss_dict_sum}
            progress_bar.set_postfix(loss=loss.item(), **avg_loss_dict)


    torch.cuda.empty_cache()  # free cached memory after epoch
    return sum(losses), len(losses)


def final_training(config, exp_path, logger):
    
    load_save = True        
    # === DDP: use DistributedSampler instead of shuffle ===
    train_dataset = RF3DPoseDataset([], cache_dir=config["cached_root"], transform=transforms.Compose([ToTensor()]), split = "train", load_save=load_save, main_modality=config['modality'], protocol_id=config["protocol"], split_id=config["split"])
    val_dataset   = RF3DPoseDataset([], cache_dir=config["cached_root"],   transform=transforms.Compose([ToTensor()]), split = "val",   load_save=load_save, main_modality=config['modality'], protocol_id=config["protocol"], split_id=config["split"])
    test_dataset  = RF3DPoseDataset([], cache_dir=config["cached_root"],  transform=transforms.Compose([ToTensor()]), split = "test",  load_save=load_save, main_modality=config['modality'], protocol_id=config["protocol"], split_id=config["split"])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                sampler=DistributedSampler(train_dataset, shuffle=True), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'],
                                sampler=DistributedSampler(val_dataset, shuffle=False), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'],
                                sampler=DistributedSampler(test_dataset, shuffle=False), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1)

    if config['modality'] == 'radar_points':
        model = P4Mamba(config['smplx_paths'],
                            radius=0.3, nsamples=32, spatial_stride=32,
                                temporal_kernel_size=3, temporal_stride=1,
                                emb_relu=False,
                                dim=1024, depth=10, heads=8, dim_head=256,
                                mlp_dim=2048, num_classes=17*3, dropout1=0.0, dropout2=0.0).cuda()
    elif config['modality'] == 'rawImage_XYZ' and config['model'] == 'RF-Mesh':
        model = Simple3DConvModelWithTripleCNNFPNAndAttention(config['smplx_paths']).cuda()
    elif config['modality'] == 'rawImage_XYZ' and config['model'] == 'RETR':
        model = RETR(smplx_model_paths=config['smplx_paths'], task="SEG", topk=64, in_channels=[4*31, 4*121]).cuda()
        
    
    if config['test_mode'] == False:
        logger.info("Experiment Configuration:")
        logger.info(json.dumps(config, indent=4))
        
        # wrap with DDP (no other logic changed)
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]), find_unused_parameters=True)

        unwrap_model(model).loss_weights = {
            'betas': config['betas_weight'],
            'pose_body': config['pose_body_weight'],
            'root_orient': config['root_orient_weight'],
            'trans': config['trans_weight'],
            'vertices': config['vertices_weight'],
            'gender': config['gender_weight']
        }

        # ========= Optimizer with Parameter Groups =========
        decay_params, no_decay_params = [], []
        for name, param in unwrap_model(model).named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.Adam([
            {'params': decay_params, 'weight_decay': config['l2_lambda']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=config['lr'])

        # ========= LR Scheduler =========
        scheduler = StepLR(optimizer,
                        step_size=config['lr_decay_step'],
                        gamma=config['lr_decay_gamma'])

        criterion1 = nn.L1Loss()
        criterion2 = nn.MSELoss()
        gender_criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        gradient_clip = False 
        log_gradient_norm = False
        for epoch in range(config['epochs']):
            if epoch > 5:
                gradient_clip = False # to avoid exploding gradients
                log_gradient_norm = True # log gradient norm to debug

            # DDP: make sure shuffling differs across epochs
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)

            train_sum, train_n = run_epoch(model, train_loader, criterion1, criterion2, gender_criterion, optimizer, gradient_clip=gradient_clip, log_gradient_norm=log_gradient_norm, modality=config['modality'])
            val_sum, val_n  = run_epoch(model, val_loader, criterion1, criterion2, gender_criterion, modality=config['modality'])
            
            # ---- global reduction ----
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            packed = torch.tensor([train_sum, train_n, val_sum, val_n],
                                device=device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            train_sum_g, train_n_g, val_sum_g, val_n_g = packed.tolist()


            train_loss = train_sum_g / max(1.0, train_n_g)
            val_loss   = val_sum_g   / max(1.0, val_n_g)
            
             # ---- continuation ----           
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            if is_main_process():
                logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.6e}")


            # ---- evaluation ----        
            result_path = os.path.join(exp_path, f'test_epoch_{epoch+1}')
            if is_main_process():
                ensure_directory_exists(result_path)
            
            local_metrics = evaluate(unwrap_model(model), test_loader, result_path, modality=config['modality'], plot_gif=config['plot_gif']) ## if epoch % 1 == 0:# evaluation only on rank0 to avoid duplicate work/files           
            
            # ---- global reduction ----
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            vec = torch.tensor([
                local_metrics['mean_vertex_error'],
                local_metrics['mean_joint_localization_error'],
                local_metrics['mean_joint_rotation_error'],
                local_metrics['mean_mesh_localization_error'],
                local_metrics['mean_gender_accuracy'],
                local_metrics['total_samples']
            ], device=device, dtype=torch.float64)  
            
                          
            if dist.is_available() and dist.is_initialized():
                torch.distributed.barrier()
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total = vec.tolist()
            csv_file = os.path.join(exp_path, f'results.csv')
            csv_best_file = os.path.join(exp_path, f'results_best.csv')
            if is_main_process():
                global_results = results_organize(sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total, epoch=epoch, summary_csv_path=csv_file, best_csv_path=csv_best_file)
            
            if is_main_process():
                logger.info(json.dumps(global_results, indent=2))
                
                
            
            # ---- save ----    
            if is_main_process():
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(unwrap_model(model).state_dict(), os.path.join(exp_path, f'best_model_epoch{epoch//5}.pth'))
                    logger.info("Model improved. Saved best model.")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        logger.info("Early stopping.")
                        break

        if is_main_process():
            logger.info("Training complete.")
        

        



    else:
        logger.info("Experiment Configuration:")
        logger.info(json.dumps(config, indent=4))

        if config['test_model_path']:
            # make sure DDP load is dp-aware if you later wrap; here we load before wrapping
            model.load_state_dict(torch.load(config['test_model_path']))

        # wrap with DDP for consistency (evaluation only)
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]), find_unused_parameters=True)
        unwrap_model(model).eval()

        unwrap_model(model).loss_weights = {
            'betas': config['betas_weight'],
            'pose_body': config['pose_body_weight'],
            'root_orient': config['root_orient_weight'],
            'trans': config['trans_weight'],
            'vertices': config['vertices_weight'],
            'gender': config['gender_weight']
        }
        
        result_path = os.path.join(exp_path, f'test_epoch')
        if is_main_process():
            ensure_directory_exists(result_path)
        # if is_main_process():  # avoid duplicate writes
        #     test_metrics = evaluate(unwrap_model(model), test_loader, result_path, modality=config['modality'])
        #     logger.info(json.dumps(test_metrics, indent=2))
        local_metrics = evaluate(unwrap_model(model), test_loader, result_path, modality=config['modality'], plot_gif=config['plot_gif']) ## if epoch % 1 == 0:# evaluation only on rank0 to avoid duplicate work/files           
            
        # ---- global reduction ----
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        vec = torch.tensor([
            local_metrics['mean_vertex_error'],
            local_metrics['mean_joint_localization_error'],
            local_metrics['mean_joint_rotation_error'],
            local_metrics['mean_mesh_localization_error'],
            local_metrics['mean_gender_accuracy'],
            local_metrics['total_samples']
        ], device=device, dtype=torch.float64)  
        
                        
        if dist.is_available() and dist.is_initialized():
            torch.distributed.barrier()
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)

        sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total = vec.tolist()
        csv_file = os.path.join(exp_path, f'results.csv')
        csv_best_file = os.path.join(exp_path, f'results_best.csv')
        if is_main_process():
            global_results = results_organize(sum_vertex, sum_jloc, sum_jrot, sum_mloc, sum_gacc, n_total, epoch=0, summary_csv_path=csv_file, best_csv_path=csv_best_file)
        
        if is_main_process():
            logger.info(json.dumps(global_results, indent=2))
        
        
# ====================
# Entry Point
# ====================
if __name__ == '__main__':
    # DDP init
    local_rank = setup_ddp()

    EXP_PATH = get_unique_exp_path(base_path="experiments")
    if is_main_process():
        ensure_directory_exists(EXP_PATH)
    LOGGER = setup_logger(os.path.join(EXP_PATH, 'train_test.log'))

    CONFIG = {
        'data_root': '/media/jiarui/HDD1/Dataset/Edinburgh_mmwave',
        'cached_root': '../mmDataset/MR-Mesh/',
        'batch_size': 64,
        'train_ratio': 0.8,
        'val_ratio': 0.05,
        'epochs': 100,
        'patience': 100,
        'lr': 2e-4,
        'lr_decay_step': 5,
        'lr_decay_gamma': 0.9,
        'l2_lambda': 1e-3,
        'betas_weight': 0.3,
        'pose_body_weight': 15,
        'root_orient_weight': 1,
        'trans_weight': 10,
        'vertices_weight': 1,
        'gender_weight': 0.5,
        'smplx_paths': {
            'male': 'models/smplx/SMPLX_MALE.npz',
            'female': 'models/smplx/SMPLX_FEMALE.npz',
            'neutral': 'models/smplx/SMPLX_NEUTRAL.npz'
        },
        'modality': 'radar_points',  # 'radar_points' or 'rawImage_XYZ'
        'model': 'P4Transformer',  # 'P4Transformer' or 'RF-Mesh' or 'RETR'
        'protocol': "p1",
        'split': "s3",
        'test_mode': False,  # Set to True to skip training and only run evaluation
        "test_model_path": "./experiments/exp_20251007_125735/best_model_epoch5.pth",  # Path to a pre-trained model if needed
        "plot_gif": True,
    }

    final_training(CONFIG, EXP_PATH, LOGGER)

    # DDP cleanup
    cleanup_ddp()
