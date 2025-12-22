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
    Perform validation/test. Calculate chunk-by-chunk performance.
    
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
        float: Average PSNR of final chunk (for backward compatibility)
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
    
    # For 'off' mode, test both sequential and variance-sorted strategies
    if args.progressive_mode == 'off':
        logger.info("=" * 80)
        logger.info("Testing 'off' mode with two transmission strategies")
        logger.info("=" * 80)
        
        # 1. Sequential transmission
        logger.info(f"\n--- Strategy 1: Sequential Transmission (Packet Size: {args.packet_size}) ---")
        results_seq = test_off_mode(loader, net, val_snr, config, args, logger, use_variance_sorting=False)
        
        # 2. Variance-based sorted transmission
        logger.info(f"\n--- Strategy 2: Variance-based Sorted Transmission (Packet Size: {args.packet_size}) ---")
        results_var = test_off_mode(loader, net, val_snr, config, args, logger, use_variance_sorting=True)
        
        logger.info("=" * 80)
        logger.info("Comparison Results (Chunk-by-Chunk):")
        logger.info(f"{'Chunk':<8} | {'Sequential':<30} | {'Variance-Sorted':<30}")
        logger.info(f"{'':8} | {'PSNR (dB)':<15} {'MS-SSIM':<15} | {'PSNR (dB)':<15} {'MS-SSIM':<15}")
        logger.info("-" * 80)
        for chunk_num, psnr_seq, ssim_seq, psnr_var, ssim_var in zip(
            results_seq['chunk'], 
            results_seq['psnr'], 
            results_seq['ssim'],
            results_var['psnr'],
            results_var['ssim']
        ):
            logger.info(
                f"{chunk_num:>8} | "
                f"{psnr_seq:>14.2f} {ssim_seq:>14.4f} | "
                f"{psnr_var:>14.2f} {ssim_var:>14.4f}"
            )
        logger.info("=" * 80)
        
        # Return sequential final PSNR for backward compatibility
        return results_seq['psnr'][-1]
    
    # For other modes, use force_progressive
    num_chunks = None
    chunk_psnrs = []
    chunk_ssims = []
    ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)
    
    os.makedirs(config.samples, exist_ok=True)
    
    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.to(config.device)
            
            # Force progressive decoding for all modes in validate
            if isinstance(net, nn.DataParallel):
                results = net.module(input_img, val_snr, force_progressive=True)
            else:
                results = net(input_img, val_snr, force_progressive=True)
            mse_list = results['mse']
            recon_list = results['recon_img']
            
            if num_chunks is None:
                num_chunks = len(mse_list)
                chunk_psnrs = [AverageMeter() for _ in range(num_chunks)]
                chunk_ssims = [AverageMeter() for _ in range(num_chunks)]
            
            for chunk_idx in range(num_chunks):
                mse_val = mse_list[chunk_idx].mean()
                recon_img = recon_list[chunk_idx]
                
                if mse_val.item() > 0:
                    psnr = 10 * np.log10(1.0 / mse_val.item())
                    chunk_psnrs[chunk_idx].update(psnr)
                
                ssim_val = ms_ssim_module(recon_img, input_img).item()
                chunk_ssims[chunk_idx].update(ssim_val)
            
            # Save images periodically
            if i == 10 and ((epoch + 1) % save_freq == 0):
                orig = input_img[0].cpu().permute(1, 2, 0).numpy()
                orig = np.clip(orig, 0, 1)
                save_path_orig = os.path.join(
                    config.samples, 
                    f"val_epoch_{epoch + 1}_original.png"
                )
                plt.imsave(save_path_orig, orig)
                
                recon = recon_list[-1][0].cpu().permute(1, 2, 0).numpy()
                recon = np.clip(recon, 0, 1)
                save_path_recon = os.path.join(
                    config.samples,
                    f"val_epoch_{epoch + 1}_recon_{val_snr}dB.png"
                )
                plt.imsave(save_path_recon, recon)
    
    # Log chunk-by-chunk results
    logger.info(f"Chunk-by-chunk results (Mode: {args.progressive_mode}):")
    for chunk_idx in range(num_chunks):
        chunk_num = chunk_idx + 1
        logger.info(
            f"  Chunk {chunk_num:2d}/{num_chunks}: "
            f"PSNR={chunk_psnrs[chunk_idx].avg:.2f} dB, "
            f"MS-SSIM={chunk_ssims[chunk_idx].avg:.4f}"
        )
    
    final_psnr = chunk_psnrs[-1].avg if chunk_psnrs else 0.0
    logger.info(
        f"Testset: {args.testset} | {val_snr} dB | "
        f"Final PSNR: {final_psnr:.2f} dB | Final MS-SSIM: {chunk_ssims[-1].avg:.4f}"
    )
    
    return final_psnr


def test_off_mode(loader, net, snr, config, args, logger, use_variance_sorting=False):
    """
    Test 'off' mode with two transmission strategies.
    Calculate chunk-by-chunk performance.
    
    Args:
        loader: Data loader
        net (nn.Module): Model to evaluate
        snr (float): SNR (dB)
        config: Config object
        args: Parsed arguments
        logger: Logger object
        use_variance_sorting (bool): Whether to use variance-based sorting
    
    Returns:
        dict: Chunk-by-chunk average PSNR, MS-SSIM dictionary
    """
    net.eval()
    
    num_chunks = None
    chunk_psnrs = []
    chunk_ssims = []
    ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)
    
    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.to(config.device)
            
            if isinstance(net, nn.DataParallel):
                results = net.module(input_img, snr, use_variance_sorting=use_variance_sorting, force_progressive=True)
            else:
                results = net(input_img, snr, use_variance_sorting=use_variance_sorting, force_progressive=True)
            
            mse_list = results['mse']
            recon_list = results['recon_img']
            
            if num_chunks is None:
                num_chunks = len(mse_list)
                chunk_psnrs = [AverageMeter() for _ in range(num_chunks)]
                chunk_ssims = [AverageMeter() for _ in range(num_chunks)]
            
            for chunk_idx in range(num_chunks):
                mse_val = mse_list[chunk_idx].mean()
                recon_img = recon_list[chunk_idx]
                
                if mse_val.item() > 0:
                    psnr = 10 * np.log10(1.0 / mse_val.item())
                    chunk_psnrs[chunk_idx].update(psnr)
                
                ssim_val = ms_ssim_module(recon_img, input_img).item()
                chunk_ssims[chunk_idx].update(ssim_val)
    
    result = {
        'chunk': list(range(1, num_chunks + 1)),
        'psnr': [meter.avg for meter in chunk_psnrs],
        'ssim': [meter.avg for meter in chunk_ssims]
    }
    
    logger.info(f"Chunk-by-chunk results ({'Variance-sorted' if use_variance_sorting else 'Sequential'}):")
    for chunk_num, psnr, ssim in zip(result['chunk'], result['psnr'], result['ssim']):
        logger.info(f"  Chunk {chunk_num:2d}: PSNR={psnr:.2f} dB, MS-SSIM={ssim:.4f}")
    
    return result


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
    
    logger.info("=" * 80)
    logger.info("SNR vs Performance Test")
    logger.info(f"Packet Size: {args.packet_size} (fixed)")
    logger.info(f"Progressive Mode: {args.progressive_mode}")
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
                recon_list = results_dict['recon_img']
                
                if num_chunks is not None:
                    chunk_idx = min(num_chunks - 1, len(mse_list) - 1)
                else:
                    chunk_idx = len(mse_list) - 1
                
                mse_val = mse_list[chunk_idx].mean()
                recon_img = recon_list[chunk_idx]
                
                if mse_val.item() > 0:
                    psnr = 10 * np.log10(1.0 / mse_val.item())
                    psnr_avg.update(psnr)
                
                ssim_val = ms_ssim_module(recon_img, input_img).item()
                ssim_avg.update(ssim_val)
        
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
        alpha_mode_for_search = args.alpha_mode if args.progressive_mode in ['alm', 'adaptive-alm'] else None
        model_path = find_model_path(
            args.model_dir,
            args.progressive_mode,
            alpha_mode=alpha_mode_for_search
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
                f"mode={args.progressive_mode}, alpha_mode={args.alpha_mode}. "
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
        logger.info(f"--- Testing Mode: {args.progressive_mode}, Packet Size: {args.packet_size}, SNR: {test_snr} dB ---")
        validate(val_loader, net, test_snr, 0, config, args, logger)