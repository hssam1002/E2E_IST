import torch.optim as optim
from net.network import E2E_SwinJSCC
from data.datasets import get_loader
from utils import *
from loss.distortion import MS_SSIM  # Ensure this is accessible
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import profile, record_function, ProfilerActivity

import os
import torch
import torch.nn as nn
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. Arguments Configuration ---
parser = argparse.ArgumentParser(description='E2E-IST for Progressive Image Transmission')

# [Mode]
parser.add_argument('--training', action='store_true', help='Flag to start training.')
parser.add_argument('--patience', type=int, default=100, help='Early stopping patience (epochs).')
parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained .pth model.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate (lower for fine-tuning).')

# [Dataset Settings]
parser.add_argument('--trainset', type=str, default='DIV2K', help='Dataset for training.')
parser.add_argument('--testset', type=str, default='Kodak', choices=['Kodak', 'CLIC2021', 'DIV2K'])

# [Model & Channel Settings]
parser.add_argument('--channel_type', type=str, default='awgn', choices=['awgn', 'rayleigh', 'noiseless'], 
                    help='Wireless channel model.')
parser.add_argument('--train_snr', type=int, default=10, 
                    help='Single SNR value (dB) used for training.')

# [Progressive Strategy]
parser.add_argument('--alpha_mode', type=str, default='base', 
                    choices=['base', 'linear', 'inverse', 'square', 'exponential', 'uniform'],
                    help="'base' = Non-progressive (All-at-once). Others = Progressive weights.")
parser.add_argument('--progressive_mode', type=str, default='progressive', 
                    choices=['progressive', 'adaptive'],
                    help="Only valid if alpha_mode != 'base'. 'adaptive' uses SSF.")
parser.add_argument('--packet_size', type=int, default=24, help='Number of features per transmission step (F).')

# [Adaptive / SSF Settings]
parser.add_argument('--ssf_target', type=str, default='both', choices=['enc', 'dec', 'both'],
                    help="Where to enable SSF in 'adaptive' mode.")

# [Visualization Range]
parser.add_argument('--save_start', type=int, default = 192, help='Start index for saving recon images.')
parser.add_argument('--save_end', type=int, default = 192, help='End index for saving recon images.')

args = parser.parse_args()

# --- 2. Logic Setup (Alpha & Mode) ---
# alpha_mode가 'base'이면 -> Base Model 학습 (Loop 없음, 마지막만 최적화)
if args.alpha_mode == 'base':
    args.progressive_mode = 'all'
    use_ssf = False
    print(f"[*] Alpha Mode is 'base'. Switching to Fast Path (progressive_mode='all'). SSF Disabled.")
else:
    # [Rule 3 & 4]
    print(f"[*] Alpha Mode is '{args.alpha_mode}'. Using Weighted MSE Loss (ALM).")
    if args.progressive_mode == 'adaptive':
        use_ssf = True
        print(f"[*] Mode: Adaptive. SSF Enabled. Pretrained weights required.")
    else:
        use_ssf = False
        print(f"[*] Mode: Progressive. SSF Disabled (Identity).")

# --- 3. Configuration Class ---
class config():
    seed = 42
    CUDA = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [Path Configuration]
    base_save_path = "/data4/hongsik/E2E_IST/results"
    train_data_dir = "/data4/hongsik/data/DIV2K" 
    test_data_dir = f"/data4/hongsik/data/{args.testset}"

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = f'{base_save_path}/{args.trainset}_SNR{args.train_snr}_{args.alpha_mode}_{args.progressive_mode}/{filename}'
    log = workdir + f'/Log_{filename}.log'
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    learning_rate = args.lr
    tot_epoch = 10000
    print_step = 1
    save_model_freq = 5 # 5 epoch마다 검증
    
    batch_size = 8 * torch.cuda.device_count()
    
    # Model Config
    common_kwargs = dict(
        img_size = (256, 256), patch_size = 2, in_chans = 3, window_size = 8, mlp_ratio = 4., 
        qkv_bias = True, qk_scale = None, norm_layer = nn.LayerNorm, patch_norm = True, model='E2E', use_ssf = use_ssf,
        use_checkpoint = True)

    encoder_kwargs = dict(embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], **common_kwargs)
    decoder_kwargs = dict(embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], **common_kwargs)
    #encoder_kwargs = dict(embed_dims=[96, 144, 192, 240], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], **common_kwargs)
    #decoder_kwargs = dict(embed_dims=[240, 192, 144, 96], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], **common_kwargs)
    

# Alpha Sequence
# --- 4. Helper: Alpha Sequence Generator ---
def get_alpha_sequence(L, mode='linear', device='cuda'):
    """
    Generate decaying alpha sequence with length L and sum 1.
    Modes: 'base', 'linear', 'inverse', 'square', 'exponential', 'uniform'
    """
    if mode == 'base': return torch.zeros(L, dtype=torch.float32, device=device)
    elif mode == 'linear':      v = torch.linspace(L, 1, steps=L)
    elif mode == 'inverse':     v = 1.0 / torch.arange(1, L + 1, dtype=torch.float32)
    elif mode == 'square':      v = 1.0 / (torch.arange(1, L + 1, dtype=torch.float32) ** 2)
    elif mode == 'exponential': v = torch.exp(-torch.arange(0, L, dtype=torch.float32))
    elif mode == 'uniform':     v = torch.ones(L, dtype=torch.float32)
    else: raise ValueError(f"Unknown alpha mode: {mode}")
    return (v / v.sum()).to(device)

def load_weights(net, path, strict=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path not found: {path}")
    state_dict = torch.load(path)
    # DataParallel로 저장된 경우 'module.' 제거
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # SSF parameter가 추가된 모델에 Base 모델을 로드할 때 strict=False 필요
    missing, unexpected = net.load_state_dict(new_state_dict, strict=strict)
    print(f"[*] Weights loaded from {path}")
    print(f"    Missing Keys (Expected if Adaptive): {len(missing)}")
    print(f"    Unexpected Keys: {len(unexpected)}")

def freeze_parameters_for_adaptive(net, target='both'):
    """
    Rule 4: Freeze everything except SSF parameters.
    Target: 'enc', 'dec', 'both'
    """
    # 1. 일단 전체 Freeze
    for param in net.parameters():
        param.requires_grad = False
        
    # 2. SSF Unfreeze
    trainable_count = 0
    for name, param in net.named_parameters():
        is_ssf = 'ssf' in name

        is_target = False
        if target == 'both': is_target = True
        elif target == 'enc' and 'encoder' in name: is_target = True
        elif target == 'dec' and 'decoder' in name: is_target = True
        
        if is_ssf and is_target:
            param.requires_grad = True
            trainable_count += 1
            
    print(f"[*] Adaptive Mode: Freezed base model. {trainable_count} SSF parameters are trainable.")

# --- 5. Training Loop ---
def train_one_epoch(args, epoch, net, optimizer, lambda_l, rho):
    """
    Train one epoch.
    If progressive mode is on, computes Lagrangian loss and accumulates constraint violation 'h'.
    Returns:
        h_avg_norm (float): Norm of the average constraint violation for this epoch (used for Rho update).
    """
    net.train()
    elapsed, losses = AverageMeter(), AverageMeter()
    
    C_total = config.encoder_kwargs['embed_dims'][-1]
    F = args.packet_size
    # Configuration
    L = (C_total + F - 1) // F
    
    # Get Alpha
    alpha = get_alpha_sequence(L, mode=args.alpha_mode, device=config.device)
    
    # Accumulator for h (constraint violations)
    h_accumulator = torch.zeros(L, device=config.device)
    num_batches = 0

    for batch_idx, input_img in enumerate(train_loader):
        start_time = time.time()
        input_img = input_img.cuda()
        num_batches += 1

        # 1. Forward Pass (Single SNR Training)
        results = net(input_img, args.train_snr)
        mse_list = [m.mean() for m in results['mse']] # List of Tensors (Length L)

        # 2. Loss Calculation
        # [Rule 2] Base Mode (Overall MSE)
        if args.progressive_mode == 'all':
            loss_main = mse_list[-1]
            total_loss = loss_main
        else: # [Rule 3 & 4] Progressive/Adaptive (ALM)
            loss_main = mse_list[-1] # Objective

            # 1) d_S(0) : Mean (X-0)^2 
            d_s_0 = torch.mean(input_img ** 2).detach()
            all_distortions = [d_s_0] + mse_list
            h_list = []
            
            for l in range(1, L + 1):
                # d_S(l - 1)
                prev_mse = all_distortions[l-1]
                # d_S(l)
                curr_mse = all_distortions[l]

                reduction_ratio = (prev_mse - curr_mse) / d_s_0
                h_val = alpha[l-1] - reduction_ratio
                h_list.append(h_val)
            
            h_stack = torch.stack(h_list)
            h_accumulator += h_stack.detach()

            h_relu = torch.nn.functional.relu(h_stack)
            
            lagrangian = torch.sum(lambda_l * h_relu) + (rho / 2) * torch.sum(h_relu ** 2)
            total_loss = loss_main + lagrangian

        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()

        # Logging
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item())
            
    if (epoch % config.print_step) == 0:
        current_lr = optimizer.param_groups[0]['lr']

        avg_lam = lambda_l.mean().item()
        max_lam = lambda_l.max().item()
        
        logger.info(
                f'Epoch {epoch} [{batch_idx}] | '
                f'Loss {losses.val:.4f} | '
                f'SNR {args.train_snr} | '
                f'LR {current_lr:.2e} | '
                f'Rho {rho:.1f} | '          # 현재 Penalty Weight
                f'Lam(Avg) {avg_lam:.2e}'    # Lambda 평균 (지수표기)
            )

    return h_accumulator / num_batches
    
# --- 6. Validation / Test Function ---
def validate(loader, net, val_snr, epoch, config, save_freq=20):
    net.eval()

    if args.channel_type not in ['awgn', 'rayleigh']:
        logger.info(f"====== Validation Results (Noiseless), Epoch {epoch + 1} ======")
    else:
        logger.info(f"====== Validation Results (SNR {val_snr} dB), Epoch {epoch + 1} ======")

    psnrs, ssims = AverageMeter(), AverageMeter()
    ms_ssim_module = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    
    if not os.path.exists(config.samples):
        os.makedirs(config.samples)

    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.cuda()

            results = net(input_img, val_snr)
            # Progressive 구조인 경우 마지막 출력 사용
            recon_img = results['recon_img'][-1]
            final_mse = results['mse'][-1].mean()

            # PSNR
            if final_mse.item() > 0:
                psnrs.update(10 * np.log10(1 / final_mse.item()))

            # MS-SSIM
            ssims.update(ms_ssim_module(recon_img, input_img).item())

            # --- [Visualization Logic] ---
            if i == 10 and ((epoch + 1) % save_freq == 0):
                plt.figure(figsize=(6, 3)) 

                # Original
                orig = input_img[0].cpu().permute(1, 2, 0).numpy()
                orig = np.clip(orig, 0, 1)

                # Reconstructed
                recon = recon_img[0].cpu().permute(1, 2, 0).numpy()
                recon = np.clip(recon, 0, 1)

                # 2. Original 따로 저장
                # 파일명 예: val_epoch_50_original.png
                save_path_orig = os.path.join(config.samples, f"val_epoch_{epoch + 1}_original.png")
                plt.imsave(save_path_orig, orig)

                # 3. Recon 따로 저장
                # 파일명 예: val_epoch_50_recon_10dB.png
                save_path_recon = os.path.join(config.samples, f"val_epoch_{epoch + 1}_recon_{val_snr}dB.png")
                plt.imsave(save_path_recon, recon)
            # ----------------------------------------

    logger.info(f"Testset: {args.testset} | {val_snr} | PSNR: {psnrs.avg:.2f} dB | MS-SSIM: {ssims.avg:.4f}")
    return psnrs.avg
    
# --- 7. Main Execution Block ---
if __name__ == '__main__':
    seed_torch(config.seed)
    logger = logger_configuration(config, save_log=True)
    logger.info("Initializing E2E-IST...")
    
    # 1. Model Initialization
    net = E2E_SwinJSCC(args, config).cuda()

    # [Rule 3 & 4] Load Pretrained Weights Logic
    if args.pretrained:
        # Base -> Base Fine-tune / Base -> Adaptive
        # strict = False allows loading Base weights into a model with SSF layers (missing keys)
        load_weights(net, args.pretrained, strict=False)

    # [Rule 4] Freeze logic for Adaptive Mode
    if args.progressive_mode == 'adaptive':
        freeze_parameters_for_adaptive(net, target = args.ssf_target)

    # Multi-GPU Setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

    # 2. Optimizer Setup (Separated)
    # Optimizer (filter frozen params)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.learning_rate)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), 
                        lr=config.learning_rate, 
                        weight_decay=1e-4)
    #scheduler = CosineAnnealingLR(optimizer, T_max=config.tot_epoch, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-7, verbose=True)
    train_loader, val_loader = get_loader(args, config)
    
    # [ALM Parameters]
    C_total = config.encoder_kwargs['embed_dims'][-1]
    F = args.packet_size
    L_steps = (C_total + F - 1) // F  # 단계 수 계산
    lambda_l = torch.zeros(L_steps, device=config.device)
    
    # [Rule A Parameters]
    rho = 1.0 # 2. Rho Init
    gamma = 1.2 #  Growth rate for rho
    zeta = 1.0  # Improvement threshold
    prev_h_norm = float('inf') # To store ||h_t||
    curr_h_norm = 0.0
    best_psnr = -1e9
    epochs_no_improve = 0
    
    if args.training:
        for epoch in range(config.tot_epoch):
            h_avg = train_one_epoch(args, epoch, net, optimizer, lambda_l, rho)
            #scheduler.step()
            
            # [ALM Update] Only for Progressive/Adaptive
            if args.progressive_mode != 'all':
                lambda_l = torch.nn.functional.relu(lambda_l + rho * h_avg) # Update Lambda: lambda_{t+1} = lambda_t + rho * h_{t+1}
                
                curr_h_norm = torch.norm(h_avg).item()
                if curr_h_norm > zeta * prev_h_norm:
                    rho *= gamma
                prev_h_norm = curr_h_norm
                
            if (epoch % config.print_step) == 0:
                logger.info(f"Start Epoch {epoch} | Mode: {args.progressive_mode} | Alpha: {args.alpha_mode}")
                logger.info(f"   >> [ALM Update] Rho: {rho:.1f} | Lambda Avg: {lambda_l.mean().item():.2e} | H Norm: {curr_h_norm:.4f}")

            # Validation
            if (epoch + 1) % config.save_model_freq == 0:
                avg_psnr = validate(val_loader, net, args.train_snr, epoch, config, save_freq = 50)

                scheduler.step(avg_psnr)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    epochs_no_improve = 0
                    save_name = f'best_model_{args.alpha_mode}.pth'
                    save_path = os.path.join(config.models, save_name)
                    if isinstance(net, nn.DataParallel):
                        torch.save(net.module.state_dict(), save_path)
                    else:
                        torch.save(net.state_dict(), save_path)
                    logger.info(f"Best Model Saved! PSNR: {best_psnr:.4f}")
                else:
                    epochs_no_improve += config.save_model_freq
                
                if epochs_no_improve >= args.patience:
                    logger.info("Early Stopping.")
                    break
    else:
        logger.info("Running Test Mode...")
        print(f"--- Testing Target SNR ({args.train_snr}) ---")
        validate(val_loader, net, args.train_snr, 0, config)