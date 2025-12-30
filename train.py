"""
Training functions
"""

import time
import os
import torch
import torch.nn as nn
import numpy as np
from utils import AverageMeter

def compute_loss(args, mse_list, input_img, L):
    """
    Compute loss based on progressive mode.
    
    Args:
        args: Parsed arguments
        mse_list (list): List of MSE losses for each step
        input_img (torch.Tensor): Input image
        L (int): Number of progressive steps
    
    Returns:
        torch.Tensor: Total loss
    """
    final_recon_loss = mse_list[-1]

    if args.progressive_mode == 'off':
        total_loss = final_recon_loss
    elif args.progressive_mode == 'rand_mask_1':
        total_loss = mse_list[0]
    
    elif args.progressive_mode == 'rand_mask_2':
        total_loss = mse_list[0] + mse_list[1]
    else:
        raise ValueError(f"Unknown progressive_mode: {args.progressive_mode}")
    
    return total_loss

def train_one_epoch(args, epoch, net, optimizer, train_loader, config, 
                     logger):
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
    
    Returns:
        torch.Tensor: Average loss for the epoch
    """
    net.train()
    
    elapsed = AverageMeter()
    losses = AverageMeter()
    mse_losses = AverageMeter()
    
    # Calculate number of progressive steps
    C_total = config.encoder_kwargs['embed_dims'][-1]
    F = args.packet_size
    L = (C_total + F - 1) // F

    num_batches = 0
    
    for batch_idx, input_img in enumerate(train_loader):
        start_time = time.time()
        input_img = input_img.to(config.device)
        num_batches += 1
        
        # Random SNR selection for each batch
        train_snr = np.random.choice(config.train_snr_list)
        
        # Forward pass
        results = net(input_img, train_snr)
        mse_list = [m.mean() for m in results['mse']]
        
        # Compute loss
        total_loss = compute_loss(
            args, mse_list, input_img, L
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Logging
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item())
        mse_losses.update(mse_list[-1].item())
    
    # End of epoch logging
    if (epoch % config.print_step) == 0:
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f'Epoch {epoch} | '
            f'Total {losses.avg:.6f} | '
            f'MSE {mse_losses.avg:.6f} | '
            f'SNR List {config.train_snr_list} | '
            f'LR {current_lr:.2e}'
        )
    
    return total_loss / num_batches if num_batches > 0 else total_loss

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
    epochs_no_improve = 0
    
    for epoch in range(config.tot_epoch):
        # Training
        train_one_epoch(
            args, epoch, net, optimizer, train_loader, 
            config, logger
        )

        # Logging
        if (epoch % config.print_step) == 0:
            logger.info(
                f"Start Epoch {epoch} | "
                f"Mode: {args.progressive_mode}"
            )
        
        # Validation
        if (epoch + 1) % config.save_model_freq == 0:
            # Calculate average SNR in linear scale, then convert back to dB
            snr_linear = np.mean([10 ** (snr_db / 10) for snr_db in config.train_snr_list])
            val_snr = 10 * np.log10(snr_linear)
            
            avg_psnr = validate(
                val_loader, net, val_snr, epoch, 
                config, args, logger, save_freq=50
            )
            
            # Learning rate scheduler update
            scheduler.step(avg_psnr)
            
            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                epochs_no_improve = 0
                save_name = f'best_model.pth'
                save_path = os.path.join(config.models, save_name)
                
                if isinstance(net, nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)
                
                logger.info(f"Best Model Saved! PSNR: {best_psnr:.4f}")
            else:
                epochs_no_improve += config.save_model_freq
            
            # Early stopping
            if epochs_no_improve >= args.patience:
                logger.info("Early Stopping.")
                break