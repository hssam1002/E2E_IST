"""
Test and validation functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import AverageMeter
from pytorch_msssim import ms_ssim
from model_utils import load_weights, find_model_path
from utils_plot import save_snr_test_results

# Import calculate_cbr from evaluate_all_models
def calculate_cbr(chunk_num, packet_size=256, img_size=(256, 256)):
    """
    Calculate Channel Bit Ratio (CBR)
    
    CBR = (# Chunk * 256) / (256 * 256 * 3 * 2)
    where 2 accounts for complex channel (real + imag)
    
    Args:
        chunk_num (int): Number of chunks transmitted
        packet_size (int): Packet size (F), default 256 as per user specification
        img_size (tuple): Image size (H, W)
    
    Returns:
        float: CBR value
    """
    H, W = img_size
    # User specified formula: CBR = (# Chunk * 256) / (256 * 256 * 3 * 2)
    transmitted_bits = chunk_num * 256
    original_bits = 256 * 256 * 3 * 2  # Fixed as per user specification
    cbr = transmitted_bits / original_bits
    return cbr


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
            f"====== Validation Results (SNR {val_snr:.2f} dB), Epoch {epoch + 1} ======"
        )
    
    # For 'off' mode, test sequential transmission
    if args.progressive_mode == 'off':
        logger.info("=" * 80)
        logger.info("Testing 'off' mode with sequential transmission")
        logger.info("=" * 80)
        
        # Sequential transmission
        logger.info(f"\n--- Sequential Transmission (Packet Size: {args.packet_size}) ---")
        results_seq = test_off_mode(loader, net, val_snr, config, args, logger)
        
        logger.info("=" * 80)
        logger.info("Chunk-by-Chunk Results:")
        logger.info(f"{'Chunk':<8} | {'PSNR (dB)':<15} {'MS-SSIM':<15}")
        logger.info("-" * 80)
        for chunk_num, psnr, ssim in zip(
            results_seq['chunk'], 
            results_seq['psnr'], 
            results_seq['ssim']
        ):
            logger.info(
                f"{chunk_num:>8} | "
                f"{psnr:>14.2f} {ssim:>14.4f}"
            )
        logger.info("=" * 80)
        
        # Return sequential final PSNR for backward compatibility
        return results_seq['psnr'][-1]
    
    # For other modes, use force_progressive
    num_chunks = None
    chunk_psnrs = []
    chunk_ssims = []
    
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
                
                ssim_val = ms_ssim(recon_img, input_img, data_range=1.0).item()
                chunk_ssims[chunk_idx].update(ssim_val)
            
            # Save images periodically
            if i == 5 and ((epoch + 1) % save_freq == 0):
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


def test_off_mode(loader, net, snr, config, args, logger):
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
    
    Returns:
        dict: Chunk-by-chunk average PSNR, MS-SSIM dictionary
    """
    net.eval()
    
    num_chunks = None
    chunk_psnrs = []
    chunk_ssims = []
    
    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.to(config.device)
            
            if isinstance(net, nn.DataParallel):
                results = net.module(input_img, snr, force_progressive=True)
            else:
                results = net(input_img, snr, force_progressive=True)
            
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
                
                ssim_val = ms_ssim(recon_img, input_img, data_range=1.0).item()
                chunk_ssims[chunk_idx].update(ssim_val)
    
    result = {
        'chunk': list(range(1, num_chunks + 1)),
        'psnr': [meter.avg for meter in chunk_psnrs],
        'ssim': [meter.avg for meter in chunk_ssims]
    }
    
    for chunk_num, psnr, ssim in zip(result['chunk'], result['psnr'], result['ssim']):
        logger.info(f"  Chunk {chunk_num:2d}: PSNR={psnr:.2f} dB, MS-SSIM={ssim:.4f}")
    
    return result


def test_snr_performance(net, loader, config, args, logger, snr_list):
    """
    Test performance for all (SNR, CBR) combinations.
    Measure PSNR and MS-SSIM for each (SNR, chunk) pair.
    
    Args:
        net (nn.Module): Model to test
        loader: Data loader
        config: Config object
        args: Parsed arguments
        logger: Logger object
        snr_list (list): List of SNR values to test (dB)
    
    Returns:
        dict: Test results dictionary with structure:
            {
                'snr': [snr1, snr1, snr1, ..., snr2, snr2, ...],  # Flattened: one per (snr, chunk)
                'chunk': [1, 2, 3, ..., 1, 2, 3, ...],  # Flattened: one per (snr, chunk)
                'cbr': [cbr1, cbr2, cbr3, ..., cbr1, cbr2, ...],  # Flattened: one per (snr, chunk)
                'psnr': [psnr for (snr1, chunk1), (snr1, chunk2), ..., (snr2, chunk1), ...],
                'ms_ssim': [ssim for (snr1, chunk1), (snr1, chunk2), ..., (snr2, chunk1), ...]
            }
    """
    net.eval()
    
    logger.info("=" * 80)
    logger.info("SNR vs Performance Test (All Chunks)")
    logger.info(f"Packet Size: {args.packet_size} (fixed)")
    logger.info(f"Progressive Mode: {args.progressive_mode}")
    logger.info(f"SNR Range: {snr_list}")
    logger.info("=" * 80)
    
    # First pass: determine number of chunks
    num_chunks = None
    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.to(config.device)
            if isinstance(net, nn.DataParallel):
                results_dict = net.module(input_img, snr_list[0], force_progressive=True)
            else:
                results_dict = net(input_img, snr_list[0], force_progressive=True)
            num_chunks = len(results_dict['mse'])
            break
    
    logger.info(f"Number of chunks: {num_chunks}")
    
    # Calculate CBR for each chunk
    cbr_list = [calculate_cbr(chunk_num) for chunk_num in range(1, num_chunks + 1)]
    
    # Results structure: flattened list for each (SNR, chunk) combination
    results = {
        'snr': [],
        'chunk': [],
        'cbr': [],
        'psnr': [],
        'ms_ssim': []
    }
    
    # Test each SNR
    for snr in snr_list:
        logger.info(f"\n--- Testing SNR: {snr} dB ---")
        
        # Initialize meters for each chunk
        chunk_psnrs = [AverageMeter() for _ in range(num_chunks)]
        chunk_ssims = [AverageMeter() for _ in range(num_chunks)]
        
        with torch.no_grad():
            for i, input_img in enumerate(loader):
                input_img = input_img.to(config.device)
                
                if isinstance(net, nn.DataParallel):
                    results_dict = net.module(input_img, snr, force_progressive=True)
                else:
                    results_dict = net(input_img, snr, force_progressive=True)
                mse_list = results_dict['mse']
                recon_list = results_dict['recon_img']
                
                # Measure performance for all chunks
                for chunk_idx in range(num_chunks):
                    mse_val = mse_list[chunk_idx].mean()
                    recon_img = recon_list[chunk_idx]
                    
                    if mse_val.item() > 0:
                        psnr = 10 * np.log10(1.0 / mse_val.item())
                        chunk_psnrs[chunk_idx].update(psnr)
                    
                    ssim_val = ms_ssim(recon_img, input_img, data_range=1.0).item()
                    chunk_ssims[chunk_idx].update(ssim_val)
        
        # Store results for this SNR (flattened: one entry per chunk)
        for chunk_idx in range(num_chunks):
            results['snr'].append(snr)
            results['chunk'].append(chunk_idx + 1)
            results['cbr'].append(cbr_list[chunk_idx])
            results['psnr'].append(chunk_psnrs[chunk_idx].avg)
            results['ms_ssim'].append(chunk_ssims[chunk_idx].avg)
        
        # Log results for this SNR
        logger.info(f"SNR: {snr:5.1f} dB - Chunk-by-chunk results:")
        for chunk_idx in range(num_chunks):
            logger.info(
                f"  Chunk {chunk_idx + 1:2d}/{num_chunks} (CBR={cbr_list[chunk_idx]:.4f}): "
                f"PSNR={chunk_psnrs[chunk_idx].avg:.2f} dB, "
                f"MS-SSIM={chunk_ssims[chunk_idx].avg:.4f}"
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
            args.progressive_mode
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
                f"mode={args.progressive_mode} "
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
    if args.test_snr_list is not None and args.test_snr_list.strip():
        try:
            # Parse comma-separated SNR list
            snr_list = [float(x.strip()) for x in args.test_snr_list.split(',')]
            snr_list = sorted(snr_list)  # Sort for consistent ordering
            logger.info(f"SNR list provided: {snr_list}")
            
            # Perform SNR performance test (all chunks for each SNR)
            results = test_snr_performance(
                net, val_loader, config, args, logger,
                snr_list=snr_list
            )
            
            # Save results and plots
            save_snr_test_results(results, config.workdir, args, logger)
            
        except ValueError as e:
            logger.error(f"Invalid SNR list format: {args.test_snr_list}. Use comma-separated values (e.g., '-5,0,5,10,15,20'). Error: {e}")
            return
    else:
        # Normal test mode (single SNR) - use train_snr_list linear mean
        if hasattr(config, 'train_snr_list') and config.train_snr_list:
            snr_linear = np.mean([10 ** (snr_db / 10) for snr_db in config.train_snr_list])
            test_snr = 10 * np.log10(snr_linear)
        else:
            test_snr = 0 if args.channel_type == 'noiseless' else 10.0  # default SNR
        logger.info(f"--- Testing Mode: {args.progressive_mode}, Packet Size: {args.packet_size}, SNR: {test_snr:.2f} dB (train_snr_list average) ---")
        validate(val_loader, net, test_snr, 0, config, args, logger)