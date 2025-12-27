"""
Training functions
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
from utils import AverageMeter

def compute_loss(args, mse_list, ms_ssim_list, recon_list, input_img, device):
    """
    Compute loss based on learning mode with weighted combination of MSE and MS-SSIM.
    
    Args:
        args: Parsed arguments
        mse_list (list): List of MSE losses for each step
        ms_ssim_list (list): List of MS-SSIM losses for each step (already 1 - ms_ssim)
        recon_list (list): List of reconstructed images for each step
        input_img (torch.Tensor): Input image (B, 3, H, W)
        device: Device to compute on
    
    Returns:
        torch.Tensor: Total loss
    """
    # Parse loss weights
    loss_weights = [float(x.strip()) for x in args.loss_weights.split(',')]
    if len(loss_weights) != 2:
        raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
    w_mse, w_ms_ssim = loss_weights
    
    # Select which losses to use based on learning mode
    if args.learning_mode == 'non-progressive':
        # loss = lambda_mse * mse + lambda_ms * (1 - ms_ssim)
        mse_selections = [mse_list[0]]
        ms_ssim_selections = [ms_ssim_list[0]]
    elif args.learning_mode == 'rand_mask_1':
        # loss = lambda_mse * mse + lambda_ms * (1 - ms_ssim)
        mse_selections = [mse_list[0]]
        ms_ssim_selections = [ms_ssim_list[0]]
    elif args.learning_mode == 'rand_mask_2':
        # loss = loss_total + loss_rand
        # loss_total = lambda_mse * mse_full + lambda_ms * (1 - ms_ssim_full)
        # loss_rand = lambda_mse * mse_partial + lambda_ms * (1 - ms_ssim_partial)
        mse_selections = mse_list  # [partial_mse, full_mse]
        ms_ssim_selections = ms_ssim_list  # [partial_ms_ssim_loss, full_ms_ssim_loss]
    else:
        raise ValueError(f"Unknown learning_mode: {args.learning_mode}")
    
    # Initialize loss components
    # Start with first loss to ensure requires_grad is set correctly
    total_loss = None
    
    # Compute loss for each selected reconstruction
    for mse_loss, ms_ssim_loss in zip(mse_selections, ms_ssim_selections):
        chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # MSE component
        if w_mse > 0:
            # Ensure mse_loss is scalar
            if mse_loss.dim() > 0:
                mse_loss = mse_loss.mean()
            chunk_loss = chunk_loss + w_mse * mse_loss
        
        # MS-SSIM component (already 1 - ms_ssim from network)
        if w_ms_ssim > 0:
            # Ensure ms_ssim_loss is scalar
            if ms_ssim_loss.dim() > 0:
                ms_ssim_loss = ms_ssim_loss.mean()
            chunk_loss = chunk_loss + w_ms_ssim * ms_ssim_loss
        
        # Accumulate total loss
        if total_loss is None:
            total_loss = chunk_loss
        else:
            total_loss = total_loss + chunk_loss
    
    # If no losses were computed, return zero loss with grad
    if total_loss is None:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss

def train_one_epoch(args, epoch, net, optimizer, train_loader, config, logger, global_step):
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
    
    elapsed = AverageMeter()
    losses = AverageMeter()
    psnr_avg = AverageMeter()
    ms_ssim_avg = AverageMeter()
    
    for batch_idx, input_img in enumerate(train_loader):
        start_time = time.time()
        input_img = input_img.to(config.device)
        
        # Forward pass
        results = net(input_img, args.train_snr)
        mse_list = results['mse']
        ms_ssim_list = results['ms_ssim']
        recon_list = results['recon_img']
        
        # Compute loss (with weighted combination of MSE and MS-SSIM)
        total_loss = compute_loss(args, mse_list, ms_ssim_list, recon_list, input_img, config.device)
        
        # Scale loss by accumulation steps
        total_loss = total_loss / config.gradient_accumulation_steps
        
        # Backward pass
        total_loss.backward()
        
        # Update weights only after accumulating gradients for specified steps
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # Update global step after optimizer step
            global_step += 1
        
        # Compute PSNR and MS-SSIM for logging
        # Select which reconstruction to use based on learning mode
        if args.learning_mode == 'non-progressive':
            mse_val = mse_list[0].mean()
            ms_ssim_loss = ms_ssim_list[0].mean()
        elif args.learning_mode == 'rand_mask_1':
            mse_val = mse_list[0].mean()
            ms_ssim_loss = ms_ssim_list[0].mean()
        elif args.learning_mode == 'rand_mask_2':
            # Use full reconstruction for metrics (final performance)
            mse_val = mse_list[1].mean()  # full_mse
            ms_ssim_loss = ms_ssim_list[1].mean()  # full_ms_ssim_loss
        else:
            mse_val = mse_list[0].mean()
            ms_ssim_loss = ms_ssim_list[0].mean()
        
        # Calculate PSNR
        if mse_val.item() > 0:
            psnr = 10 * np.log10(1.0 / mse_val.item())
            psnr_avg.update(psnr)
        
        # Calculate MS-SSIM similarity (1 - loss)
        ms_ssim_sim = 1.0 - ms_ssim_loss.item()
        ms_ssim_avg.update(ms_ssim_sim)
        
        # Logging (scale loss back for logging since we divided by accumulation_steps)
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item() * config.gradient_accumulation_steps)
    
    # Handle remaining gradients at the end of epoch (if batch count is not divisible by accumulation_steps)
    if len(train_loader) % config.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    
    # End of epoch logging
    if (epoch % config.print_step) == 0:
        current_lr = optimizer.param_groups[0]['lr']
        steps_epoch = global_step // len(train_loader)
        logger.info(
            f'Epoch {epoch} | '
            f'Step {global_step} | '
            f'Steps/Epoch {steps_epoch} | '
            f'Loss {losses.avg:.6f} | '
            f'PSNR {psnr_avg.avg:.2f} dB | '
            f'MS-SSIM {ms_ssim_avg.avg:.4f} | '
            f'SNR {args.train_snr} | '
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
    from test import validate, validate_progressive
    
    global_step = 0
    best_PSNR = 0
    
    for epoch in range(config.tot_epoch):
        # Training
        global_step = train_one_epoch(args, epoch, net, optimizer, train_loader, config, logger, global_step)
        
        # Logging
        if (epoch % config.print_step) == 0:
            logger.info(
                f"Start Epoch {epoch} | "
                f"Mode: {args.learning_mode}"
            )
        
        # Validation
        if (epoch + 1) % config.save_model_freq == 0:
            # Normal validation (uses learning_mode configuration)
            avg_psnr, avg_loss = validate(
                val_loader, net, args.train_snr, epoch, 
                config, args, logger, save_freq=50
            )
            
            # Progressive validation (100 epoch마다만 실행)
            if (epoch + 1) % 100 == 0:
                logger.info("=" * 80)
                logger.info("Running Progressive Validation (every 100 epochs)...")
                logger.info("=" * 80)
                validate_progressive(
                    val_loader, net, args.train_snr, epoch,
                    config, args, logger, save_freq=50
                )
            
            # Learning rate scheduler update (MultiStepLR updates per step/epoch)
            scheduler.step()
            
            # Save best model based on PSNR (higher is better)
            if avg_psnr > best_PSNR:
                best_PSNR = avg_psnr
                save_name = f'best_model_{args.learning_mode}.pth'
                save_path = os.path.join(config.models, save_name)
                
                if isinstance(net, nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)
                
                logger.info(f"Best Model Saved! PSNR: {best_PSNR:.4f} dB, Loss: {avg_loss:.6f}")