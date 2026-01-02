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


def save_snr_test_results(results, save_dir, args, logger):
    """
    Save SNR test results to files.
    
    Args:
        results (dict): Test results dictionary with structure:
            {
                'snr': [snr1, snr2, ...],
                'chunk': [1, 2, 3, ...],
                'cbr': [cbr1, cbr2, ...],
                'psnr': [[psnr for each chunk at snr1], ...],
                'ms_ssim': [[ssim for each chunk at snr1], ...]
            }
        save_dir (str): Save directory
        args: Parsed arguments
        logger: Logger object
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Results are already flattened: each row is (snr, chunk, cbr, psnr, ms_ssim)
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, 'snr_test_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"SNR test results saved to CSV: {csv_path}")
    
    json_results = {
        'args': {
            'channel_type': args.channel_type,
            'testset': args.testset,
            'progressive_mode': args.progressive_mode,
            'packet_size': args.packet_size
        },
        'results': results
    }
    json_path = os.path.join(save_dir, 'snr_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"SNR test results saved to JSON: {json_path}")
    
    plot_snr_test_results(results, save_dir, logger)


def plot_snr_test_results(results, save_dir, logger):
    """
    Visualize SNR test results as graphs.
    Plot PSNR and MS-SSIM vs CBR for each SNR.
    
    Args:
        results (dict): Test results dictionary with flattened structure:
            {
                'snr': [snr1, snr1, ..., snr2, snr2, ...],  # One per (snr, chunk)
                'chunk': [1, 2, 3, ..., 1, 2, 3, ...],  # One per (snr, chunk)
                'cbr': [cbr1, cbr2, ..., cbr1, cbr2, ...],  # One per (snr, chunk)
                'psnr': [psnr for each (snr, chunk)],
                'ms_ssim': [ssim for each (snr, chunk)]
            }
        save_dir (str): Save directory
        logger: Logger object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Get unique SNR values
    unique_snrs = sorted(df['snr'].unique())
    
    # Plot PSNR vs CBR for each SNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_snrs)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '+']
    
    for snr_idx, snr in enumerate(unique_snrs):
        # Filter data for this SNR and sort by CBR
        snr_data = df[df['snr'] == snr].sort_values('cbr')
        
        ax1.plot(snr_data['cbr'], snr_data['psnr'], 
                marker=markers[snr_idx % len(markers)],
                label=f'SNR={snr:.1f} dB',
                linewidth=2,
                markersize=6,
                color=colors[snr_idx])
        
        ax2.plot(snr_data['cbr'], snr_data['ms_ssim'],
                marker=markers[snr_idx % len(markers)],
                label=f'SNR={snr:.1f} dB',
                linewidth=2,
                markersize=6,
                color=colors[snr_idx])
    
    ax1.set_xlabel('CBR (Channel Bit Ratio)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs CBR (by SNR)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best', ncol=2)
    
    ax2.set_xlabel('CBR (Channel Bit Ratio)', fontsize=12)
    ax2.set_ylabel('MS-SSIM', fontsize=12)
    ax2.set_title('MS-SSIM vs CBR (by SNR)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best', ncol=2)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'snr_cbr_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SNR vs CBR comparison plot saved: {plot_path}")