"""
Training functions
"""

import time
import os
import torch
import torch.nn as nn
import numpy as np
from utils import AverageMeter
from pytorch_msssim import ms_ssim
from config import DEFAULT_NOISELESS_EPOCH_SIZE

def compute_loss(args, mse_list, recon_list, input_img, device):
    """
    Compute loss: L = lambda_1 * MSE - lambda_2 * MS-SSIM
    
    Args:
        args: Parsed arguments
        mse_list (list): List of MSE losses for each step
        recon_list (list): List of reconstructed images for each step
        input_img (torch.Tensor): Input image (B, 3, H, W)
        device: Device to compute on
    
    Returns:
        torch.Tensor: Total loss
    """
    # Parse loss weights: L = lambda_1 * MSE - lambda_2 * MS-SSIM
    loss_weights = [float(x.strip()) for x in args.loss_weights.split(',')]
    if len(loss_weights) != 2:
        raise ValueError(f"loss_weights must have 2 values [lambda_1 (MSE), lambda_2 (MS-SSIM)]. Got {len(loss_weights)}")
    lambda_mse, lambda_ms_ssim = loss_weights
    
    # 네트워크에서 objective(표의 d_s(·) 조합)에 맞게 여러 reconstruction을 반환하므로,
    # 여기서는 단순히 모든 항에 대해 동일한 손실을 계산해서 합산한다.
    # (모드별로 어떤 조합을 쓸지는 net.forward에서 결정)
    mse_selections = mse_list
    recon_selections = recon_list
    
    # Compute loss for each selected reconstruction
    # For rand_mask_2: computes loss separately for partial and full reconstructions
    #   - partial: random mask applied (partial channels)
    #   - full: all channels (full reconstruction)
    # Both partial and full losses are computed and summed
    total_loss = None
    for mse_loss, recon_img in zip(mse_selections, recon_selections):
        # Ensure mse_loss is scalar
        if mse_loss.dim() > 0:
            mse_loss = mse_loss.mean()
        
        # MSE component: lambda_1 * MSE
        chunk_loss = lambda_mse * mse_loss
        
        # MS-SSIM component: - lambda_2 * MS-SSIM (negative because MS-SSIM is similarity, higher is better)
        # MS-SSIM is computed separately for each reconstruction (partial/full for rand_mask_2)
        if lambda_ms_ssim > 0:
            ms_ssim_sim = ms_ssim(recon_img, input_img, data_range=1.0)
            if ms_ssim_sim.dim() > 0:
                ms_ssim_sim = ms_ssim_sim.mean()
            # MS-SSIM is similarity (0-1), so we subtract it (higher similarity = lower loss)
            chunk_loss = chunk_loss - lambda_ms_ssim * ms_ssim_sim
        
        # Accumulate total loss
        if total_loss is None:
            total_loss = chunk_loss
        else:
            total_loss = total_loss + chunk_loss
    
    return total_loss

def train_one_epoch(args, epoch, net, optimizer, train_loader, config, 
                     logger, global_step):
    """
    Train for one epoch.
    Args:
        args: Parsed arguments
        epoch (int): Current epoch number
        net (nn.Module): Model to train
        optimizer: Optimizer
        train_loader: Training data loader
        config: Config object
        logger: Logger object
        global_step: Global step counter (will be updated)
    
    Returns:
        int: Updated global_step
    """
    net.train()
    
    # Training strategy: 처음 noiseless_epoch epoch는 noiseless, 그 이후는 train_snr_list 사용
    use_noiseless = epoch < DEFAULT_NOISELESS_EPOCH_SIZE
    
    # Channel 타입 동적 변경
    if use_noiseless:
        current_channel_type = 'noiseless'
    else:
        current_channel_type = args.channel_type
    
    if isinstance(net, nn.DataParallel):
        net.module.channel.set_channel_type(current_channel_type)
    else:
        net.channel.set_channel_type(current_channel_type)
    
    elapsed = AverageMeter()
    losses = AverageMeter()
    mse_losses = AverageMeter()
    ms_ssim_losses = AverageMeter()

    num_batches = 0
    
    for batch_idx, input_img in enumerate(train_loader):
        start_time = time.time()
        input_img = input_img.to(config.device)
        num_batches += 1
        
        # SNR selection: noiseless면 0, 아니면 랜덤 선택
        if use_noiseless:
            train_snr = 0
        else:
            train_snr = np.random.choice(config.train_snr_list)
        
        # Forward pass
        results = net(input_img, train_snr)
        mse_list = results['mse']
        recon_list = results['recon_img']
        
        # Compute loss: L = lambda_1 * MSE - lambda_2 * MS-SSIM
        total_loss = compute_loss(
            args, mse_list, recon_list, input_img, config.device
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update global step
        global_step += 1
        
        # Calculate MS-SSIM for logging (use final reconstruction)
        final_recon = recon_list[-1]
        ms_ssim_sim = ms_ssim(final_recon, input_img, data_range=1.0)
        if ms_ssim_sim.dim() > 0:
            ms_ssim_sim = ms_ssim_sim.mean()
        
        # Logging
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item())
        # Get final MSE for logging
        final_mse = mse_list[-1].mean() if isinstance(mse_list[-1], torch.Tensor) else mse_list[-1]
        mse_losses.update(final_mse.item())
        ms_ssim_losses.update(ms_ssim_sim.item())
    
    # End of epoch logging
    if (epoch % config.print_step) == 0:
        current_lr = optimizer.param_groups[0]['lr']
        channel_info = f'noiseless (Epoch: {epoch}/{DEFAULT_NOISELESS_EPOCH_SIZE})' if use_noiseless else f'{current_channel_type}'
        logger.info(
            f'Epoch {epoch} | '
            f'Total {losses.avg:.6f} | '
            f'MSE {mse_losses.avg:.6f} | '
            f'MS-SSIM {ms_ssim_losses.avg:.4f} | '
            f'Channel: {channel_info} | '
            f'SNR List {config.train_snr_list} | '
            f'LR {current_lr:.2e}'
        )
    
    return global_step

def train_model(args, net, optimizer, scheduler, train_loader, val_loader, 
                config, logger):
    """
    Main training loop.
    
    Args:
        args: Parsed arguments
        net (nn.Module): Model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Config object
        logger: Logger object
    """
    from test import validate
    
    best_psnr = -1e9
    global_step = 0
    
    for epoch in range(config.tot_epoch):
        # Training
        global_step = train_one_epoch(
            args, epoch, net, optimizer, train_loader, 
            config, logger, global_step
        )

        # Logging
        if (epoch % config.print_step) == 0:
            logger.info(
                f"Start Epoch {epoch} | "
                f"Mode: {args.progressive_mode}"
            )
        
        # Validation
        if (epoch + 1) % config.save_model_freq == 0:
            # Calculate average SNR: convert dB to linear scale, average, then convert back to dB
            # SNR is in dB scale, so we need to average in linear scale
            snr_linear = np.mean([10 ** (snr_db / 10) for snr_db in config.train_snr_list])
            val_snr = 10 * np.log10(snr_linear)
            
            avg_psnr = validate(
                val_loader, net, val_snr, epoch, 
                config, args, logger, save_freq=50
            )
            
            # Learning rate scheduler update
            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(avg_psnr)  # ReduceLROnPlateau needs metric
            else:
                scheduler.step()  # Other schedulers update per epoch
            
            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_name = f'best_model.pth'
                save_path = os.path.join(config.models, save_name)
                
                if isinstance(net, nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)
                
                logger.info(f"Best Model Saved! PSNR: {best_psnr:.4f}")