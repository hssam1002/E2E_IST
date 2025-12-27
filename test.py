"""
Test and validation functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import AverageMeter
from loss.distortion import MS_SSIM
from model_utils import load_weights, find_model_path
from utils_plot import save_snr_test_results


def validate(loader, net, val_snr, epoch, config, args, logger, save_freq=20):
    """
    Perform normal validation (non-progressive).
    Uses the model's learning_mode configuration.
    
    Args:
        loader: Data loader
        net (nn.Module): Model to evaluate
        val_snr (int): Validation SNR (dB)
        epoch (int): Current epoch number
        config: Config object
        args: Parsed arguments
        logger: Logger object
        save_freq (int): Image save frequency
    
    Returns:
        tuple: (avg_psnr, avg_loss) - Average PSNR and loss
    """
    net.eval()
    
    if args.channel_type not in ['awgn', 'rayleigh']:
        logger.info(
            f"====== Validation Results (Noiseless), Epoch {epoch + 1} ======"
        )
    else:
        logger.info(
            f"====== Validation Results (SNR {val_snr} dB), Epoch {epoch + 1} ======"
        )
    
    from train import compute_loss
    
    # Loss weights
    loss_weights = [float(x.strip()) for x in args.loss_weights.split(',')]
    if len(loss_weights) != 2:
        raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
    w_mse, w_ms_ssim = loss_weights
    
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    loss_avg = AverageMeter()
    
    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.to(config.device)
            
            # Forward pass without force_progressive (uses learning_mode)
            if isinstance(net, nn.DataParallel):
                results = net.module(input_img, val_snr, force_progressive=False)
            else:
                results = net(input_img, val_snr, force_progressive=False)
            
            # Get final reconstruction based on learning_mode
            mse_list = results['mse']
            ms_ssim_list = results['ms_ssim']
            recon_list = results['recon_img']
            
            # For rand_mask_2, use the full reconstruction (last element)
            # For other modes, use the first (and only) element
            if args.learning_mode == 'rand_mask_2':
                mse_val = mse_list[-1].mean()  # Full reconstruction
                ms_ssim_loss = ms_ssim_list[-1].mean()
            else:
                mse_val = mse_list[0].mean()
                ms_ssim_loss = ms_ssim_list[0].mean()
            
            # Compute metrics
            if mse_val.item() > 0:
                psnr = 10 * np.log10(1.0 / mse_val.item())
                psnr_avg.update(psnr)
            
            # MS-SSIM similarity (1 - loss)
            ms_ssim_sim = 1.0 - ms_ssim_loss.item()
            ssim_avg.update(ms_ssim_sim)
            
            # Loss
            chunk_loss = w_mse * mse_val + w_ms_ssim * ms_ssim_loss
            loss_avg.update(chunk_loss.item())
    
    logger.info(f"PSNR: {psnr_avg.avg:.2f} dB | MS-SSIM: {ssim_avg.avg:.4f} | Loss: {loss_avg.avg:.6f}")
    
    return psnr_avg.avg, loss_avg.avg


def validate_progressive(loader, net, val_snr, epoch, config, args, logger, save_freq=20):
    """
    Perform validation/test with progressive transmission.
    Calculate performance at packet_size, 2*packet_size, ..., C_total channels and sum all losses.
    
    Args:
        loader: Data loader
        net (nn.Module): Model to evaluate
        val_snr (int): Validation SNR (dB)
        epoch (int): Current epoch number
        config: Config object
        args: Parsed arguments
        logger: Logger object
        save_freq (int): Image save frequency
    
    Returns:
        tuple: (avg_psnr, avg_loss) - Average PSNR at final chunk and sum of all losses
    """
    net.eval()
    
    if args.channel_type not in ['awgn', 'rayleigh']:
        logger.info(
            f"====== Validation Results (Noiseless), Epoch {epoch + 1} ======"
        )
    else:
        logger.info(
            f"====== Validation Results (SNR {val_snr} dB), Epoch {epoch + 1} ======"
        )
    
    from train import compute_loss
    
    # Loss weights
    loss_weights = [float(x.strip()) for x in args.loss_weights.split(',')]
    if len(loss_weights) != 2:
        raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
    w_mse, w_ms_ssim = loss_weights
    
    os.makedirs(config.samples, exist_ok=True)
    
    # Get total channels
    with torch.no_grad():
        sample_img = next(iter(loader)).to(config.device)
        if isinstance(net, nn.DataParallel):
            feat_all = net.module.encoder(sample_img)
        else:
            feat_all = net.encoder(sample_img)
        B, Seq, C_total = feat_all.shape
    
    # Progressive transmission: packet_size, 2*packet_size, ..., up to C_total
    # Get packet_size from network or config
    if isinstance(net, nn.DataParallel):
        packet_size = net.module.packet_size
    else:
        packet_size = net.packet_size
    
    logger.info(f"Encoder output shape: (B={B}, Seq={Seq}, C={C_total})")
    logger.info("=" * 80)
    logger.info(f"Progressive Transmission Validation ({packet_size}, {packet_size*2}, ..., {C_total} channels):")
    logger.info(f"{'Channels':<10} | {'PSNR (dB)':<12} | {'MS-SSIM':<12} | {'Loss':<12}")
    logger.info("-" * 80)
    chunk_sizes = list(range(packet_size, C_total + 1, packet_size))
    if chunk_sizes[-1] != C_total:
        chunk_sizes.append(C_total)
    
    chunk_results = {
        'channels': [],
        'psnr': [],
        'ssim': [],
        'loss': []
    }
    
    total_loss_sum = 0.0
    final_psnr = 0.0
    
    with torch.no_grad():
        for num_channels in chunk_sizes:
            psnr_avg = AverageMeter()
            ssim_avg = AverageMeter()
            loss_avg = AverageMeter()
            
            for i, input_img in enumerate(loader):
                input_img = input_img.to(config.device)
                
                # Forward pass with force_progressive=True
                if isinstance(net, nn.DataParallel):
                    results = net.module(input_img, val_snr, force_progressive=True)
                else:
                    results = net(input_img, val_snr, force_progressive=True)
                
                # Find the chunk index corresponding to num_channels
                # chunk_sizes are [packet_size, 2*packet_size, ..., C_total]
                chunk_idx = chunk_sizes.index(num_channels)
                if chunk_idx >= len(results['mse']):
                    chunk_idx = len(results['mse']) - 1
                
                mse_val = results['mse'][chunk_idx].mean()
                ms_ssim_loss = results['ms_ssim'][chunk_idx].mean()
                
                # Compute metrics
                if mse_val.item() > 0:
                    psnr = 10 * np.log10(1.0 / mse_val.item())
                    psnr_avg.update(psnr)
                
                # MS-SSIM similarity (1 - loss)
                ms_ssim_sim = 1.0 - ms_ssim_loss.item()
                ssim_avg.update(ms_ssim_sim)
                
                # Loss for this chunk: w_mse * mse + w_ms_ssim * ms_ssim_loss
                chunk_loss = w_mse * mse_val + w_ms_ssim * ms_ssim_loss
                loss_avg.update(chunk_loss.item())
                
                # Save images at first chunk and last image
                if i == 0 and num_channels == chunk_sizes[0] and ((epoch + 1) % save_freq == 0):
                    if 'recon_img' in results and len(results['recon_img']) > chunk_idx:
                        recon_img = results['recon_img'][chunk_idx]
                        orig = input_img[0].cpu().permute(1, 2, 0).numpy()
                        orig = np.clip(orig, 0, 1)
                        save_path_orig = os.path.join(
                            config.samples, 
                            f"val_epoch_{epoch + 1}_original.png"
                        )
                        plt.imsave(save_path_orig, orig)
                        
                        recon = recon_img[0].cpu().permute(1, 2, 0).numpy()
                        recon = np.clip(recon, 0, 1)
                        save_path_recon = os.path.join(
                            config.samples,
                            f"val_epoch_{epoch + 1}_recon_{num_channels}ch_{val_snr}dB.png"
                        )
                        plt.imsave(save_path_recon, recon)
            
            # Store results
            chunk_results['channels'].append(num_channels)
            chunk_results['psnr'].append(psnr_avg.avg)
            chunk_results['ssim'].append(ssim_avg.avg)
            chunk_results['loss'].append(loss_avg.avg)
            
            total_loss_sum += loss_avg.avg
            
            logger.info(
                f"{num_channels:<10} | "
                f"{psnr_avg.avg:<12.2f} | "
                f"{ssim_avg.avg:<12.4f} | "
                f"{loss_avg.avg:<12.6f}"
            )
            
            # Final chunk PSNR
            if num_channels == C_total:
                final_psnr = psnr_avg.avg
    
    logger.info("=" * 80)
    logger.info(f"Total Loss (sum of all chunks): {total_loss_sum:.6f}")
    logger.info(f"Final PSNR (at {C_total} channels): {final_psnr:.2f} dB")
    
    # Save results to JSON file
    import json
    results_dict = {
        'metadata': {
            'learning_mode': args.learning_mode,
            'train_snr': args.train_snr,
            'test_snr': val_snr,
            'testset': args.testset,
            'channel_type': args.channel_type,
            'epoch': epoch + 1
        },
        'chunk_results': chunk_results,
        'total_loss': total_loss_sum,
        'final_psnr': final_psnr
    }
    
    results_file = os.path.join(config.workdir, f'validation_results_epoch{epoch + 1}.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Validation results saved to: {results_file}")
    
    return final_psnr, total_loss_sum


def test_snr_performance(net, loader, config, args, logger, 
                         snr_list, num_chunks=None):
    """
    Test performance across multiple SNR values.
    Measure performance at specific chunk number.
    
    Args:
        net (nn.Module): Model to test
        loader: Data loader
        config: Config object
        args: Parsed arguments
        logger: Logger object
        snr_list (list): List of SNR values to test (dB)
        num_chunks (int, optional): Chunk number to measure. None for final chunk
    
    Returns:
        dict: Test results dictionary {'snr': [...], 'psnr': [...], 'ssim': [...]}
    """
    net.eval()
    
    results = {
        'snr': [],
        'psnr': [],
        'ssim': []
    }
    
    ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)
    
    # Get packet_size from network
    if isinstance(net, nn.DataParallel):
        packet_size = net.module.packet_size
    else:
        packet_size = net.packet_size
    
    logger.info("=" * 80)
    logger.info("SNR vs Performance Test")
    logger.info(f"Packet Size: {packet_size} (from network config)")
    logger.info(f"Learning Mode: {args.learning_mode}")
    if num_chunks is not None:
        logger.info(f"Measuring performance at chunk {num_chunks}")
    else:
        logger.info(f"Measuring performance at final chunk (all chunks)")
    logger.info(f"SNR Range: {snr_list}")
    logger.info("=" * 80)
    
    for snr in snr_list:
        logger.info(f"\n--- Testing SNR: {snr} dB ---")
        
        psnr_avg = AverageMeter()
        ssim_avg = AverageMeter()
        
        with torch.no_grad():
            for i, input_img in enumerate(loader):
                input_img = input_img.to(config.device)
                
                if isinstance(net, nn.DataParallel):
                    results_dict = net.module(input_img, snr, force_progressive=True)
                else:
                    results_dict = net(input_img, snr, force_progressive=True)
                mse_list = results_dict['mse']
                ms_ssim_list = results_dict['ms_ssim']
                
                if num_chunks is not None:
                    chunk_idx = min(num_chunks - 1, len(mse_list) - 1)
                else:
                    chunk_idx = len(mse_list) - 1
                
                mse_val = mse_list[chunk_idx].mean()
                ms_ssim_loss = ms_ssim_list[chunk_idx].mean()
                
                if mse_val.item() > 0:
                    psnr = 10 * np.log10(1.0 / mse_val.item())
                    psnr_avg.update(psnr)
                
                # MS-SSIM similarity (1 - loss)
                ms_ssim_sim = 1.0 - ms_ssim_loss.item()
                ssim_avg.update(ms_ssim_sim)
        
        results['snr'].append(snr)
        results['psnr'].append(psnr_avg.avg)
        results['ssim'].append(ssim_avg.avg)
        
        logger.info(
            f"SNR: {snr:5.1f} dB | "
            f"PSNR: {psnr_avg.avg:.2f} dB | "
            f"MS-SSIM: {ssim_avg.avg:.4f}"
        )
    
    logger.info("=" * 80)
    logger.info("SNR vs Performance Test Completed")
    logger.info("=" * 80)
    
    return results


def load_test_model(net, args, logger):
    """
    Load model for test mode.
    
    Args:
        net (nn.Module): Model
        args: Parsed arguments
        logger: Logger object
    """
    if args.model_dir:
        model_path = find_model_path(
            args.model_dir,
            args.learning_mode
        )
        if model_path:
            logger.info(f"Loading model from: {model_path}")
            if isinstance(net, nn.DataParallel):
                load_weights(net.module, model_path, strict=False)
            else:
                load_weights(net, model_path, strict=False)
        else:
            logger.warning(
                f"Model not found in {args.model_dir} for "
                f"mode={args.learning_mode}. "
                f"Using current model weights."
            )


def run_test_mode(args, net, val_loader, config, logger):
    """
    Run test mode.
    
    Args:
        args: Parsed arguments
        net (nn.Module): Model
        val_loader: Validation data loader
        config: Config object
        logger: Logger object
    """
    logger.info("Running Test Mode...")
    
    load_test_model(net, args, logger)
    
    # SNR performance test (if test_snr_list is provided)
    if args.test_snr_list is not None:
        try:
            # Parse comma-separated SNR list
            snr_list = [float(x.strip()) for x in args.test_snr_list.split(',')]
            snr_list = sorted(snr_list)  # Sort for consistent ordering
            logger.info(f"SNR list provided: {snr_list}")
            
            # Perform SNR performance test
            results = test_snr_performance(
                net, val_loader, config, args, logger,
                snr_list=snr_list,
                num_chunks=args.test_snr_chunk
            )
            
            # Save results and plots
            save_snr_test_results(results, config.workdir, args, logger, num_chunks=args.test_snr_chunk)
            
        except ValueError as e:
            logger.error(f"Invalid SNR list format: {args.test_snr_list}. Use comma-separated values (e.g., '-5,0,5,10,15,20'). Error: {e}")
            return
    else:
        # Normal test mode (single SNR)
        test_snr = 0 if args.channel_type == 'noiseless' else args.train_snr
        logger.info(f"--- Testing Mode: {args.learning_mode}, SNR: {test_snr} dB ---")
        validate(val_loader, net, test_snr, 0, config, args, logger)