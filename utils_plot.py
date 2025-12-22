"""
Plotting and saving utilities for test results
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def save_test_results(results, save_dir, args, logger):
    """
    Save test results to files.
    
    Args:
        results (dict): Test results dictionary
        save_dir (str): Save directory
        args: Parsed arguments
        logger: Logger object
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, 'test_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to CSV: {csv_path}")
    
    json_results = {
        'args': {
            'channel_type': args.channel_type,
            'testset': args.testset,
            'pretrained': args.pretrained
        },
        'results': results
    }
    json_path = os.path.join(save_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to JSON: {json_path}")
    
    plot_test_results(results, save_dir, logger)


def plot_test_results(results, save_dir, logger):
    """
    Visualize test results as graphs.
    
    Args:
        results (dict): Test results dictionary
        save_dir (str): Save directory
        logger: Logger object
    """
    df = pd.DataFrame(results)
    
    modes = df['mode'].unique()
    
    # MSE graph
    plt.figure(figsize=(12, 6))
    for mode in modes:
        mode_data = df[df['mode'] == mode]
        mode_data = mode_data.sort_values('packet_size')
        plt.plot(
            mode_data['packet_size'],
            mode_data['mse'],
            marker='o',
            label=mode,
            linewidth=2,
            markersize=8
        )
    plt.xlabel('Packet Size', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('MSE vs Packet Size by Progressive Mode', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    mse_plot_path = os.path.join(save_dir, 'mse_vs_packet_size.png')
    plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"MSE plot saved: {mse_plot_path}")
    
    # PSNR graph
    plt.figure(figsize=(12, 6))
    for mode in modes:
        mode_data = df[df['mode'] == mode]
        mode_data = mode_data.sort_values('packet_size')
        plt.plot(
            mode_data['packet_size'],
            mode_data['psnr'],
            marker='o',
            label=mode,
            linewidth=2,
            markersize=8
        )
    plt.xlabel('Packet Size', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR vs Packet Size by Progressive Mode', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    psnr_plot_path = os.path.join(save_dir, 'psnr_vs_packet_size.png')
    plt.savefig(psnr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"PSNR plot saved: {psnr_plot_path}")
    
    # MS-SSIM graph
    plt.figure(figsize=(12, 6))
    for mode in modes:
        mode_data = df[df['mode'] == mode]
        mode_data = mode_data.sort_values('packet_size')
        plt.plot(
            mode_data['packet_size'],
            mode_data['ssim'],
            marker='o',
            label=mode,
            linewidth=2,
            markersize=8
        )
    plt.xlabel('Packet Size', fontsize=12)
    plt.ylabel('MS-SSIM', fontsize=12)
    plt.title('MS-SSIM vs Packet Size by Progressive Mode', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ssim_plot_path = os.path.join(save_dir, 'ssim_vs_packet_size.png')
    plt.savefig(ssim_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"MS-SSIM plot saved: {ssim_plot_path}")


def save_snr_test_results(results, save_dir, args, logger, num_chunks=None):
    """
    Save SNR test results to files.
    
    Args:
        results (dict): Test results dictionary {'snr': [...], 'psnr': [...], 'ssim': [...]}
        save_dir (str): Save directory
        args: Parsed arguments
        logger: Logger object
        num_chunks (int, optional): Measured chunk number
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, 'snr_test_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"SNR test results saved to CSV: {csv_path}")
    
    json_results = {
        'args': {
            'channel_type': args.channel_type,
            'testset': args.testset,
            'progressive_mode': args.progressive_mode,
            'packet_size': args.packet_size,
            'num_chunks': num_chunks
        },
        'results': results
    }
    json_path = os.path.join(save_dir, 'snr_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"SNR test results saved to JSON: {json_path}")
    
    plot_snr_test_results(results, save_dir, logger, num_chunks)


def plot_snr_test_results(results, save_dir, logger, num_chunks=None):
    """
    Visualize SNR test results as graphs.
    
    Args:
        results (dict): Test results dictionary {'snr': [...], 'psnr': [...], 'ssim': [...]}
        save_dir (str): Save directory
        logger: Logger object
        num_chunks (int, optional): Measured chunk number
    """
    df = pd.DataFrame(results)
    df = df.sort_values('snr')
    
    chunk_info = f" (chunk {num_chunks})" if num_chunks is not None else " (final chunk)"
    
    # PSNR graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['snr'], df['psnr'], marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'PSNR vs SNR{chunk_info}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    psnr_plot_path = os.path.join(save_dir, 'psnr_vs_snr.png')
    plt.savefig(psnr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"PSNR vs SNR plot saved: {psnr_plot_path}")
    
    # MS-SSIM graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['snr'], df['ssim'], marker='s', linewidth=2, markersize=8, color='green')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('MS-SSIM', fontsize=12)
    plt.title(f'MS-SSIM vs SNR{chunk_info}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ssim_plot_path = os.path.join(save_dir, 'ssim_vs_snr.png')
    plt.savefig(ssim_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"MS-SSIM vs SNR plot saved: {ssim_plot_path}")