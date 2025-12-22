"""
Training functions
"""

import time
import os
import torch
import torch.nn as nn
from utils import AverageMeter
from model_utils import get_alpha_sequence
from config import DEFAULT_RHO_INIT, DEFAULT_RHO_GAMMA, DEFAULT_RHO_MAX, DEFAULT_ZETA

def compute_loss(args, mse_list, input_img, alpha, lambda_l, rho, d_s_0, L):
    """
    Compute loss based on progressive mode.
    
    Args:
        args: Parsed arguments
        mse_list (list): List of MSE losses for each step
        input_img (torch.Tensor): Input image
        alpha (torch.Tensor): Alpha weight sequence
        lambda_l (torch.Tensor): Lagrange multiplier
        rho (float): Penalty parameter
        d_s_0 (torch.Tensor): Initial distortion (Mean(X^2))
        L (int): Number of progressive steps
    
    Returns:
        tuple: (total_loss, h_stack, loss_h_term)
    """
    final_recon_loss = mse_list[-1]
    loss_h_term = torch.tensor(0.0, device=input_img.device)
    h_stack = None
    
    if args.progressive_mode == 'off':
        total_loss = final_recon_loss
    
    elif args.progressive_mode in ['mrl', 'adaptive-mrl']:
        total_loss = torch.sum(torch.stack(mse_list))
    
    elif args.progressive_mode == 'rand_mask_1':
        total_loss = mse_list[0]
    
    elif args.progressive_mode == 'rand_mask_2':
        total_loss = mse_list[0] + mse_list[1]
    
    elif args.progressive_mode in ['alm', 'adaptive-alm']:
        loss_main = mse_list[-1]
        
        # Constraint: h_l = alpha_l - reduction_ratio_l
        all_distortions = [d_s_0] + mse_list
        h_list = []
        
        for l in range(1, L + 1):
            prev_mse = all_distortions[l - 1]
            curr_mse = all_distortions[l]
            reduction_ratio = (prev_mse - curr_mse) / d_s_0
            h_val = alpha[l - 1] - reduction_ratio
            h_list.append(h_val)
        
        h_stack = torch.stack(h_list)
        h_relu = torch.nn.functional.relu(h_stack)
        
        # Lagrangian: lambda^T * h + (rho/2) * ||h||^2
        lagrangian = (
            torch.sum(lambda_l * h_relu) + 
            (rho / 2) * torch.sum(h_relu ** 2)
        )
        total_loss = loss_main + lagrangian
        loss_h_term = lagrangian.detach()
    
    else:
        raise ValueError(f"Unknown progressive_mode: {args.progressive_mode}")
    
    return total_loss, h_stack, loss_h_term

def train_one_epoch(args, epoch, net, optimizer, train_loader, config, 
                     lambda_l, rho, logger):
    """
    Train for one epoch.
    
    Args:
        args: Parsed arguments
        epoch (int): Current epoch number
        net (nn.Module): Model to train
        optimizer: Optimizer
        train_loader: Training data loader
        config: Config object
        lambda_l (torch.Tensor): Lagrange multiplier
        rho (float): Penalty parameter
        logger: Logger object
    
    Returns:
        torch.Tensor: Average constraint violation (for rho update)
    """
    net.train()
    
    elapsed = AverageMeter()
    losses = AverageMeter()
    mse_losses = AverageMeter()
    h_losses = AverageMeter()
    
    # Calculate number of progressive steps
    C_total = config.encoder_kwargs['embed_dims'][-1]
    F = args.packet_size
    L = (C_total + F - 1) // F
    
    # Generate alpha sequence (for ALM-based)
    alpha = get_alpha_sequence(L, mode=args.alpha_mode, device=config.device)
    h_accumulator = torch.zeros(L, device=config.device)
    num_batches = 0
    
    for batch_idx, input_img in enumerate(train_loader):
        start_time = time.time()
        input_img = input_img.to(config.device)
        num_batches += 1
        
        # Forward pass
        results = net(input_img, args.train_snr)
        mse_list = [m.mean() for m in results['mse']]
        
        # Calculate initial distortion
        d_s_0 = torch.mean(input_img ** 2).detach()
        
        # Compute loss
        total_loss, h_stack, loss_h_term = compute_loss(
            args, mse_list, input_img, alpha, lambda_l, rho, d_s_0, L
        )
        
        # Accumulate constraint violations (for ALM update)
        if h_stack is not None:
            h_accumulator += h_stack.detach()
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Logging
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item())
        mse_losses.update(mse_list[-1].item())
        h_losses.update(loss_h_term.item())
    
    # End of epoch logging
    if (epoch % config.print_step) == 0:
        current_lr = optimizer.param_groups[0]['lr']
        avg_lam = lambda_l.mean().item() if len(lambda_l) > 0 else 0.0
        logger.info(
            f'Epoch {epoch} | '
            f'Total {losses.avg:.6f} | '
            f'MSE {mse_losses.avg:.6f} | '
            f'H-term {h_losses.avg:.8f} | '
            f'SNR {args.train_snr} | '
            f'Rho {rho:.2e} | '
            f'Lam {avg_lam:.2e} | '
            f'LR {current_lr:.2e}'
        )
    
    return h_accumulator / num_batches if num_batches > 0 else h_accumulator

def update_alm_parameters(lambda_l, h_avg, rho, prev_h_norm, 
                          gamma=DEFAULT_RHO_GAMMA, 
                          max_rho=DEFAULT_RHO_MAX,
                          zeta=DEFAULT_ZETA):
    """
    Update ALM (Augmented Lagrangian Method) parameters.
    
    Args:
        lambda_l (torch.Tensor): Lagrange multiplier
        h_avg (torch.Tensor): Average constraint violation
        rho (float): Current penalty parameter
        prev_h_norm (float): Previous constraint violation norm
        gamma (float): Rho increase rate
        max_rho (float): Maximum rho
        zeta (float): Improvement threshold
    
    Returns:
        tuple: (updated_lambda_l, updated_rho, curr_h_norm)
    """
    # Lambda update: lambda_{t+1} = relu(lambda_t + rho * h_{t+1})
    updated_lambda_l = torch.nn.functional.relu(lambda_l + rho * h_avg)
    
    # Rho update: increase if constraint violation doesn't improve
    curr_h_norm = torch.norm(h_avg).item()
    updated_rho = rho
    
    if curr_h_norm > zeta * prev_h_norm:
        updated_rho = min(max_rho, rho * gamma)
    
    return updated_lambda_l, updated_rho, curr_h_norm

def train_model(args, net, optimizer, scheduler, train_loader, val_loader, 
                config, lambda_l, rho, prev_h_norm, logger):
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
        lambda_l (torch.Tensor): Lagrange multiplier
        rho (float): Penalty parameter
        prev_h_norm (float): Previous constraint violation norm
        logger: Logger object
    """
    from test import validate
    
    best_psnr = -1e9
    epochs_no_improve = 0
    curr_h_norm = 0.0
    
    for epoch in range(config.tot_epoch):
        # Training
        h_avg = train_one_epoch(
            args, epoch, net, optimizer, train_loader, 
            config, lambda_l, rho, logger
        )
        
        # ALM parameter update (only for ALM-based modes)
        if args.progressive_mode in ['alm', 'adaptive-alm']:
            lambda_l, rho, curr_h_norm = update_alm_parameters(
                lambda_l, h_avg, rho, prev_h_norm
            )
            prev_h_norm = curr_h_norm
        
        # Logging
        if (epoch % config.print_step) == 0:
            logger.info(
                f"Start Epoch {epoch} | "
                f"Mode: {args.progressive_mode} | "
                f"Alpha: {args.alpha_mode}"
            )
            if args.progressive_mode in ['alm', 'adaptive-alm']:
                logger.info(
                    f"   >> [ALM Update] Rho: {rho:.4f} | "
                    f"Lambda Avg: {lambda_l.mean().item():.2e} | "
                    f"H Norm: {curr_h_norm:.4f}"
                )
        
        # Validation
        if (epoch + 1) % config.save_model_freq == 0:
            avg_psnr = validate(
                val_loader, net, args.train_snr, epoch, 
                config, args, logger, save_freq=50
            )
            
            # Learning rate scheduler update
            scheduler.step(avg_psnr)
            
            # Save best model
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
            
            # Early stopping
            if epochs_no_improve >= args.patience:
                logger.info("Early Stopping.")
                break