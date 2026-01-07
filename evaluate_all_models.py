"""
Evaluate all models in results folder with same validation set
and plot PSNR, MS-SSIM vs CBR
"""

import os
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.datasets import get_loader
from config import setup_argument_parser, Config
from model_utils import setup_model, load_weights
from test import validate
from utils import seed_torch, logger_configuration
from pytorch_msssim import ms_ssim
from utils import AverageMeter

# Fixed seed for validation to ensure same noise across different models
VALIDATION_SEED = 42


def calculate_cbr(chunk_num, packet_size=16, img_size=(256, 256)):
    """
    Calculate Channel Bit Ratio (CBR)
    
    CBR = transmitted_symbols / original_symbols
    where:
        transmitted_symbols = chunk_num * packet_size * 256 / 2
        original_symbols = 256 * 256 * 3
    
    Args:
        chunk_num (int): Number of chunks transmitted
        packet_size (int): Packet size (F)
        img_size (tuple): Image size (H, W)
    
    Returns:
        float: CBR value
    """
    H, W = img_size
    # Transmitted symbols: chunk_num * packet_size * 256 / 2
    transmitted_symbols = chunk_num * packet_size * 256 / 2
    # Original symbols: H * W * 3
    original_symbols = H * W * 3
    cbr = transmitted_symbols / original_symbols
    return cbr


def extract_model_info(model_dir):
    """
    Extract model information from log file
    
    Args:
        model_dir (str): Directory containing model and log
    
    Returns:
        dict: Model information (embed_dims, depths, num_heads, packet_size, etc.)
    """
    info = {
        'embed_dims': None,
        'depths': None,
        'num_heads': None,
        'packet_size': 16,  # default
        'progressive_mode': 'rand_mask_2',  # default
        'channel_type': 'awgn',  # default
        'train_snr_list': '-10,-5,0,5,10',  # default
    }
    
    # Try to find log file
    log_files = glob.glob(os.path.join(model_dir, 'Log_*.log'))
    if log_files:
        log_file = log_files[0]
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Try to extract packet_size
                if 'packet_size' in content or 'Packet Size' in content:
                    # Simple extraction - can be improved
                    pass
        except:
            pass
    
    return info


def evaluate_model(model_path, args_template, config_template, val_loader, logger):
    """
    Evaluate a single model and return chunk-by-chunk results
    
    Args:
        model_path (str): Path to model file
        args_template: Template args object
        config_template: Template config object
        val_loader: Validation data loader
        logger: Logger object
    
    Returns:
        dict: Results with 'cbr', 'psnr', 'ms_ssim' lists
    """
    # Create new args and config for this model
    import copy
    import argparse
    args = argparse.Namespace()
    # Copy all attributes from template
    for attr in dir(args_template):
        if not attr.startswith('_'):
            setattr(args, attr, getattr(args_template, attr))
    
    # Set model path
    args.model_dir = os.path.dirname(model_path)  # models directory
    args.pretrained = model_path
    
    # Try to extract architecture info from log file
    # Log file is in the parent directory of models directory
    models_dir = os.path.dirname(model_path)  # models directory
    model_dir = os.path.dirname(models_dir)  # parent directory (contains Log_*.log)
    log_files = glob.glob(os.path.join(model_dir, 'Log_*.log'))
    if log_files:
        try:
            with open(log_files[0], 'r') as f:
                log_content = f.read()
                # Extract embed_dims, depths, num_heads from log
                import re
                # Look for encoder config in log - more flexible pattern
                embed_match = re.search(r"'embed_dims':\s*\[([\d,\s]+)\]", log_content)
                depths_match = re.search(r"'depths':\s*\[([\d,\s]+)\]", log_content)
                heads_match = re.search(r"'num_heads':\s*\[([\d,\s]+)\]", log_content)
                
                if embed_match and depths_match and heads_match:
                    embed_dims_str = embed_match.group(1).strip()
                    depths_str = depths_match.group(1).strip()
                    heads_str = heads_match.group(1).strip()
                    
                    args.embed_dims = embed_dims_str
                    args.depths = depths_str
                    args.num_heads = heads_str
                    logger.info(f"  Extracted architecture: embed_dims={embed_dims_str}, depths={depths_str}, num_heads={heads_str}")
                else:
                    logger.warning(f"  Could not extract architecture from log file (matches: embed={bool(embed_match)}, depths={bool(depths_match)}, heads={bool(heads_match)})")
                    # If extraction fails, use default architecture
                    logger.warning(f"  Using default architecture - this may cause size mismatch errors!")
        except Exception as e:
            logger.warning(f"  Could not extract architecture from log: {e}")
    
    # Create config
    config = Config(args)
    
    # Debug: Log the actual architecture used
    logger.info(f"  Creating model with: embed_dims={config.encoder_kwargs['embed_dims']}, depths={config.encoder_kwargs['depths']}, num_heads={config.encoder_kwargs['num_heads']}")
    
    # Setup model
    net = setup_model(args, config, logger)
    
    # Load weights with strict=False to handle architecture mismatches
    try:
        if isinstance(net, nn.DataParallel):
            load_weights(net.module, model_path, strict=False)
        else:
            load_weights(net, model_path, strict=False)
    except Exception as e:
        logger.error(f"  Failed to load weights: {e}")
        raise
    
    # Calculate validation SNR (average of train_snr_list)
    snr_linear = np.mean([10 ** (snr_db / 10) for snr_db in config.train_snr_list])
    val_snr = 10 * np.log10(snr_linear)
    
    # Evaluate model
    net.eval()
    
    # Fix seed for reproducible validation
    import random
    random.seed(VALIDATION_SEED)
    np.random.seed(VALIDATION_SEED)
    torch.manual_seed(VALIDATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(VALIDATION_SEED)
    
    num_chunks = None
    chunk_psnrs = []
    chunk_ssims = []
    
    with torch.no_grad():
        for i, input_img in enumerate(val_loader):
            input_img = input_img.to(config.device)
            
            # Force progressive decoding
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
    
    # Calculate CBR for each chunk
    packet_size = args.packet_size
    cbr_list = []
    psnr_list = []
    ms_ssim_list = []
    
    for chunk_idx in range(num_chunks):
        chunk_num = chunk_idx + 1
        cbr = calculate_cbr(chunk_num, packet_size)
        cbr_list.append(cbr)
        psnr_list.append(chunk_psnrs[chunk_idx].avg)
        ms_ssim_list.append(chunk_ssims[chunk_idx].avg)
    
    # Extract architecture info for model name
    embed_dims_str = ','.join(map(str, config.encoder_kwargs['embed_dims']))
    depths_str = ','.join(map(str, config.encoder_kwargs['depths']))
    model_name = f"E{embed_dims_str}_D{depths_str}"
    
    return {
        'model_path': model_path,
        'model_name': model_name,
        'cbr': cbr_list,
        'psnr': psnr_list,
        'ms_ssim': ms_ssim_list,
        'num_chunks': num_chunks,
        'packet_size': packet_size
    }


def plot_results(all_results, save_path):
    """
    Plot PSNR and MS-SSIM vs CBR for all models
    
    Args:
        all_results (list): List of result dictionaries
        save_path (str): Path to save plot
    """
    if len(all_results) == 0:
        print("Warning: No results to plot!")
        return
    
    # Use distinct colors and markers for better visualization
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '+']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot PSNR vs CBR
    for idx, result in enumerate(all_results):
        # Use model_name if available, otherwise fallback to path
        if 'model_name' in result:
            label = result['model_name']
        else:
            model_name = os.path.basename(os.path.dirname(result['model_path']))
            label = model_name[:20] if len(model_name) > 20 else model_name
        
        ax1.plot(result['cbr'], result['psnr'], 
                marker=markers[idx % len(markers)], 
                label=label, 
                linewidth=2, 
                markersize=6,
                color=colors[idx])
    
    ax1.set_xlabel('CBR (Channel Bit Ratio)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs CBR', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best', ncol=1)
    
    # Plot MS-SSIM vs CBR
    for idx, result in enumerate(all_results):
        # Use model_name if available, otherwise fallback to path
        if 'model_name' in result:
            label = result['model_name']
        else:
            model_name = os.path.basename(os.path.dirname(result['model_path']))
            label = model_name[:20] if len(model_name) > 20 else model_name
        
        ax2.plot(result['cbr'], result['ms_ssim'], 
                marker=markers[idx % len(markers)], 
                label=label, 
                linewidth=2, 
                markersize=6,
                color=colors[idx])
    
    ax2.set_xlabel('CBR (Channel Bit Ratio)', fontsize=12)
    ax2.set_ylabel('MS-SSIM', fontsize=12)
    ax2.set_title('MS-SSIM vs CBR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best', ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def main():
    """Main function to evaluate all models"""
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Evaluate models and create CBR comparison plots')
    parser.add_argument(
        '--result_dirs',
        type=str,
        nargs='+',
        default=None,
        help='List of result directories to evaluate (e.g., /path/to/result1 /path/to/result2). If not provided, evaluates all models in results folder.'
    )
    parser.add_argument(
        '--testset',
        type=str,
        default='Kodak',
        help='Test dataset to use (default: Kodak)'
    )
    
    eval_args = parser.parse_args()
    
    # Setup model args
    model_parser = setup_argument_parser()
    model_args = model_parser.parse_args(['--testset', eval_args.testset])
    
    config = Config(model_args)
    seed_torch(config.seed)
    logger = logger_configuration(config, save_log=False)
    
    if eval_args.result_dirs:
        logger.info(f"Evaluating {len(eval_args.result_dirs)} specific result directories...")
        for dir_path in eval_args.result_dirs:
            logger.info(f"  - {dir_path}")
    else:
        logger.info("Evaluating all models in results folder...")
    
    # Get validation loader
    _, val_loader = get_loader(model_args, config)
    
    # Find model files
    results_dir = "/data4/hongsik/E2E_IST/results"
    
    if eval_args.result_dirs:
        # Use specific result directories
        model_files = []
        for result_dir in eval_args.result_dirs:
            # Check if it's a models directory or result directory
            if result_dir.endswith('models'):
                model_path = os.path.join(result_dir, 'best_model.pth')
            else:
                model_path = os.path.join(result_dir, 'models', 'best_model.pth')
            
            if os.path.exists(model_path):
                model_files.append(model_path)
                logger.info(f"  Found model: {model_path}")
            else:
                logger.warning(f"  Model not found: {model_path}")
    else:
        # Find all model files
        model_pattern = os.path.join(results_dir, "**/best_model.pth")
        model_files = glob.glob(model_pattern, recursive=True)
        # Filter out OLD folder
        model_files = [f for f in model_files if 'OLD' not in f]
    
    logger.info(f"Found {len(model_files)} models to evaluate")
    
    # Evaluate each model
    all_results = []
    for model_path in model_files:
        try:
            logger.info(f"Evaluating: {model_path}")
            result = evaluate_model(model_path, model_args, config, val_loader, logger)
            all_results.append(result)
            logger.info(f"  Completed: {result['num_chunks']} chunks, Final PSNR: {result['psnr'][-1]:.2f} dB")
        except Exception as e:
            logger.error(f"  Error evaluating {model_path}: {e}")
            continue
    
    # Save results to JSON
    results_data = []
    for result in all_results:
        results_data.append({
            'model_path': result['model_path'],
            'model_name': result.get('model_name', 'Unknown'),
            'cbr': result['cbr'],
            'psnr': result['psnr'],
            'ms_ssim': result['ms_ssim'],
            'num_chunks': result['num_chunks'],
            'packet_size': result['packet_size']
        })
    
    # Save results to JSON
    if eval_args.result_dirs:
        json_filename = '3_results_comparison.json'
        plot_filename = 'cbr_comparison_3_results.png'
    else:
        json_filename = 'all_models_evaluation.json'
        plot_filename = 'cbr_comparison.png'
    
    json_path = os.path.join(results_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to: {json_path}")
    
    # Plot results
    plot_path = os.path.join(results_dir, plot_filename)
    plot_results(all_results, plot_path)
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()

