"""
모든 progressive mode 모델들을 평가하고 성능 비교 플롯 및 샘플 이미지를 생성하는 스크립트.

평가 항목:
1. SNR vs PSNR (모든 모델 비교)
2. SNR vs MS-SSIM (모든 모델 비교)
3. SNR 10dB에서 CBR vs PSNR (모든 모델 비교)
4. SNR 10dB에서 CBR vs MS-SSIM (모든 모델 비교)
5. 각 모델에 대해 첫 chunk와 full chunk reconstructed image 저장 (하나의 샘플)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_msssim import ms_ssim
from PIL import Image
import torchvision.transforms as transforms

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import setup_argument_parser, Config
from model_utils import setup_model, load_weights, find_model_path
from data.datasets import get_loader
from utils import logger_configuration, seed_torch
from test import calculate_cbr


def evaluate_model(net, loader, snr_list, config, args, logger, sample_idx=0):
    """
    모델을 평가하고 결과를 반환.
    
    Args:
        net: 모델
        loader: 데이터 로더
        snr_list: 테스트할 SNR 리스트
        config: Config 객체
        args: 파싱된 인자
        logger: 로거
        sample_idx: 샘플 이미지로 사용할 인덱스 (기본 0)
    
    Returns:
        dict: 평가 결과
            {
                'snr': [...],
                'chunk': [...],
                'cbr': [...],
                'psnr': [...],
                'ms_ssim': [...],
                'sample_images': {
                    'original': tensor,
                    'first_chunk': tensor,
                    'full_chunk': tensor
                }
            }
    """
    net.eval()
    
    results = {
        'snr': [],
        'chunk': [],
        'cbr': [],
        'psnr': [],
        'ms_ssim': []
    }
    
    sample_images = {
        'original': None,
        'first_chunk': None,
        'full_chunk': None
    }
    
    # 첫 번째 이미지로 chunk 수 결정
    num_chunks = None
    with torch.no_grad():
        for i, input_img in enumerate(loader):
            input_img = input_img.to(config.device)
            if isinstance(net, nn.DataParallel):
                results_dict = net.module(input_img, snr_list[0], force_progressive=True)
            else:
                results_dict = net(input_img, snr_list[0], force_progressive=True)
            num_chunks = len(results_dict['mse'])
            
            # 샘플 이미지 저장 (첫 번째 이미지 사용)
            if i == sample_idx:
                sample_images['original'] = input_img[0].cpu()
                sample_images['first_chunk'] = results_dict['recon_img'][0][0].cpu()
                sample_images['full_chunk'] = results_dict['recon_img'][-1][0].cpu()
            break
    
    logger.info(f"Number of chunks: {num_chunks}")
    
    # CBR 계산
    cbr_list = [calculate_cbr(chunk_num, packet_size=args.packet_size) 
                for chunk_num in range(1, num_chunks + 1)]
    
    # 각 SNR에 대해 평가
    for snr in snr_list:
        logger.info(f"Evaluating at SNR: {snr} dB")
        
        # 각 chunk에 대한 평균을 저장할 리스트
        chunk_psnrs = [[] for _ in range(num_chunks)]
        chunk_ssims = [[] for _ in range(num_chunks)]
        
        with torch.no_grad():
            for i, input_img in enumerate(loader):
                input_img = input_img.to(config.device)
                
                if isinstance(net, nn.DataParallel):
                    results_dict = net.module(input_img, snr, force_progressive=True)
                else:
                    results_dict = net(input_img, snr, force_progressive=True)
                
                mse_list = results_dict['mse']
                recon_list = results_dict['recon_img']
                
                # 각 chunk에 대해 성능 측정
                for chunk_idx in range(num_chunks):
                    mse_val = mse_list[chunk_idx].mean()
                    recon_img = recon_list[chunk_idx]
                    
                    if mse_val.item() > 0:
                        psnr = 10 * np.log10(1.0 / mse_val.item())
                    else:
                        psnr = 100.0  # 무한대에 가까운 값
                    
                    ssim_val = ms_ssim(recon_img, input_img, data_range=1.0).mean().item()
                    
                    chunk_psnrs[chunk_idx].append(psnr)
                    chunk_ssims[chunk_idx].append(ssim_val)
        
        # 각 chunk에 대한 평균 계산
        for chunk_idx in range(num_chunks):
            avg_psnr = np.mean(chunk_psnrs[chunk_idx])
            avg_ssim = np.mean(chunk_ssims[chunk_idx])
            
            results['snr'].append(snr)
            results['chunk'].append(chunk_idx + 1)
            results['cbr'].append(cbr_list[chunk_idx])
            results['psnr'].append(avg_psnr)
            results['ms_ssim'].append(avg_ssim)
    
    results['sample_images'] = sample_images
    return results


def plot_snr_vs_performance(all_results, save_dir, logger):
    """
    SNR vs PSNR, SNR vs MS-SSIM 플롯 생성 (모든 모델 비교)
    
    Args:
        all_results: {mode_name: results_dict} 형태의 딕셔너리
        save_dir: 저장 디렉토리
        logger: 로거
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 최종 chunk만 사용 (full transmission)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    mode_names = {
        'full': 'Full',
        'full_m': 'Full_m',
        'full_dual': 'Full_dual',
        'full_part': 'Full_part',
        'full_part_m': 'Full_part_m',
        'mask_only': 'Mask_only',
        'hybrid_all': 'hybrid_all'
    }
    
    for mode_idx, (mode, results) in enumerate(all_results.items()):
        # 최종 chunk만 필터링
        df_dict = {
            'snr': [],
            'psnr': [],
            'ms_ssim': []
        }
        
        # 각 SNR에 대해 최종 chunk 찾기
        snr_to_final = {}
        for i, snr in enumerate(results['snr']):
            chunk = results['chunk'][i]
            if snr not in snr_to_final or chunk > snr_to_final[snr]['chunk']:
                snr_to_final[snr] = {
                    'chunk': chunk,
                    'psnr': results['psnr'][i],
                    'ms_ssim': results['ms_ssim'][i]
                }
        
        snrs = sorted(snr_to_final.keys())
        psnrs = [snr_to_final[s]['psnr'] for s in snrs]
        ssims = [snr_to_final[s]['ms_ssim'] for s in snrs]
        
        label = mode_names.get(mode, mode)
        ax1.plot(snrs, psnrs, marker=markers[mode_idx % len(markers)],
                label=label, linewidth=2, markersize=8, color=colors[mode_idx])
        ax2.plot(snrs, ssims, marker=markers[mode_idx % len(markers)],
                label=label, linewidth=2, markersize=8, color=colors[mode_idx])
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('SNR vs PSNR (Final Chunk)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('MS-SSIM', fontsize=12)
    ax2.set_title('SNR vs MS-SSIM (Final Chunk)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'snr_vs_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SNR vs Performance plot saved: {plot_path}")


def plot_cbr_vs_performance_10dB(all_results, save_dir, logger):
    """
    SNR 10dB에서 CBR vs PSNR, CBR vs MS-SSIM 플롯 생성 (모든 모델 비교)
    
    Args:
        all_results: {mode_name: results_dict} 형태의 딕셔너리
        save_dir: 저장 디렉토리
        logger: 로거
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    mode_names = {
        'full': 'Full',
        'full_m': 'Full_m',
        'full_dual': 'Full_dual',
        'full_part': 'Full_part',
        'full_part_m': 'Full_part_m',
        'mask_only': 'Mask_only',
        'hybrid_all': 'hybrid_all'
    }
    
    for mode_idx, (mode, results) in enumerate(all_results.items()):
        # SNR 10dB 데이터만 필터링
        cbrs = []
        psnrs = []
        ssims = []
        
        for i, snr in enumerate(results['snr']):
            if abs(snr - 10.0) < 0.1:  # 10dB 근처
                cbrs.append(results['cbr'][i])
                psnrs.append(results['psnr'][i])
                ssims.append(results['ms_ssim'][i])
        
        # CBR 순서로 정렬
        sorted_data = sorted(zip(cbrs, psnrs, ssims))
        cbrs, psnrs, ssims = zip(*sorted_data) if sorted_data else ([], [], [])
        
        if len(cbrs) > 0:
            label = mode_names.get(mode, mode)
            ax1.plot(cbrs, psnrs, marker=markers[mode_idx % len(markers)],
                    label=label, linewidth=2, markersize=8, color=colors[mode_idx])
            ax2.plot(cbrs, ssims, marker=markers[mode_idx % len(markers)],
                    label=label, linewidth=2, markersize=8, color=colors[mode_idx])
    
    ax1.set_xlabel('CBR (Channel Bandwidth Ratio)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('CBR vs PSNR at SNR=10 dB', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    ax2.set_xlabel('CBR (Channel Bandwidth Ratio)', fontsize=12)
    ax2.set_ylabel('MS-SSIM', fontsize=12)
    ax2.set_title('CBR vs MS-SSIM at SNR=10 dB', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'cbr_vs_performance_10dB.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"CBR vs Performance (10dB) plot saved: {plot_path}")


def save_sample_images(all_results, save_dir, logger):
    """
    각 모델에 대해 샘플 이미지 저장 (original, first_chunk, full_chunk)
    
    Args:
        all_results: {mode_name: results_dict} 형태의 딕셔너리
        save_dir: 저장 디렉토리
        logger: 로거
    """
    os.makedirs(save_dir, exist_ok=True)
    
    mode_names = {
        'full': 'Full',
        'full_m': 'Full_m',
        'full_dual': 'Full_dual',
        'full_part': 'Full_part',
        'full_part_m': 'Full_part_m',
        'mask_only': 'Mask_only',
        'hybrid_all': 'hybrid_all'
    }
    
    # Original 이미지 (모든 모델이 동일하므로 첫 번째 것 사용)
    original = None
    for mode, results in all_results.items():
        if 'sample_images' in results and results['sample_images']['original'] is not None:
            original = results['sample_images']['original']
            break
    
    if original is None:
        logger.warning("No sample images found!")
        return
    
    # Original 저장
    original_np = original.permute(1, 2, 0).numpy()
    original_np = np.clip(original_np, 0, 1)
    original_pil = Image.fromarray((original_np * 255).astype(np.uint8))
    original_path = os.path.join(save_dir, 'original.png')
    original_pil.save(original_path)
    
    # 각 모델의 reconstructed 이미지 저장
    num_modes = len(all_results)
    fig, axes = plt.subplots(3, num_modes, figsize=(4 * num_modes, 12))
    if num_modes == 1:
        axes = axes.reshape(-1, 1)
    
    # 첫 번째 행: Original (모든 열에 동일하게)
    for col in range(num_modes):
        axes[0, col].imshow(original_np)
        axes[0, col].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    
    # 두 번째 행: First chunk
    # 세 번째 행: Full chunk
    for col_idx, (mode, results) in enumerate(all_results.items()):
        if 'sample_images' not in results:
            continue
        
        imgs = results['sample_images']
        label = mode_names.get(mode, mode)
        
        # First chunk
        if imgs['first_chunk'] is not None:
            first_np = imgs['first_chunk'].permute(1, 2, 0).numpy()
            first_np = np.clip(first_np, 0, 1)
            axes[1, col_idx].imshow(first_np)
            axes[1, col_idx].set_title(f'{label}\n(First Chunk)', fontsize=10, fontweight='bold')
            axes[1, col_idx].axis('off')
        
        # Full chunk
        if imgs['full_chunk'] is not None:
            full_np = imgs['full_chunk'].permute(1, 2, 0).numpy()
            full_np = np.clip(full_np, 0, 1)
            axes[2, col_idx].imshow(full_np)
            axes[2, col_idx].set_title(f'{label}\n(Full Chunk)', fontsize=10, fontweight='bold')
            axes[2, col_idx].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'sample_images_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Sample images comparison saved: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate all progressive mode models')
    parser.add_argument('--testset', type=str, default='Kodak', 
                       choices=['Kodak', 'CLIC2021', 'DIV2K'],
                       help='Test dataset')
    parser.add_argument('--snr_list', type=str, default='-10,-5,0,5,10,15,20',
                       help='Comma-separated SNR values (dB)')
    parser.add_argument('--packet_size', type=int, default=16,
                       help='Packet size')
    parser.add_argument('--results_dir', type=str, default='/data4/hongsik/E2E_IST/results',
                       help='Results directory containing model folders')
    parser.add_argument('--save_dir', type=str, default='/data4/hongsik/E2E_IST/results/progressive_modes_comparison',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # SNR 리스트 파싱
    snr_list = [float(x.strip()) for x in args.snr_list.split(',')]
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Logger 설정
    import logging
    logger = logging.getLogger("ProgressiveModesEvaluation")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 모델 디렉토리 찾기
    model_dirs = {}
    modes = ['full', 'full_m', 'full_dual', 'full_part', 'full_part_m', 'mask_only', 'hybrid_all']
    
    for mode in modes:
        mode_dir = os.path.join(args.results_dir, f'DIV2K_SNR10to10_{mode}')
        if os.path.exists(mode_dir):
            # 가장 최근 타임스탬프 디렉토리 찾기
            subdirs = [d for d in os.listdir(mode_dir) 
                      if os.path.isdir(os.path.join(mode_dir, d))]
            if subdirs:
                latest_subdir = sorted(subdirs)[-1]
                model_path = os.path.join(mode_dir, latest_subdir, 'models', 'best_model.pth')
                if os.path.exists(model_path):
                    model_dirs[mode] = model_path
                    logger.info(f"Found model for {mode}: {model_path}")
                else:
                    logger.warning(f"best_model.pth not found for {mode}")
            else:
                logger.warning(f"No subdirectories found in {mode_dir}")
        else:
            logger.warning(f"Model directory not found: {mode_dir}")
    
    if len(model_dirs) == 0:
        logger.error("No models found! Exiting.")
        return
    
    # 각 모델 평가
    all_results = {}
    
    for mode, model_path in model_dirs.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {mode}")
        logger.info(f"{'='*80}")
        
        # Config 생성 (임시 args 사용)
        temp_args = argparse.Namespace()
        temp_args.progressive_mode = mode
        temp_args.testset = args.testset
        temp_args.packet_size = args.packet_size
        temp_args.trainset = 'DIV2K'  # 기본값
        temp_args.channel_type = 'awgn'
        temp_args.train_snr_list = '10'
        temp_args.mask_prob = 0.1
        temp_args.embed_dims = '128,192,256,320'
        temp_args.depths = '2,2,6,2'
        temp_args.num_heads = '4,6,8,10'
        temp_args.transmitted_dim = 320
        temp_args.loss_weights = '10,1'
        temp_args.optimizer = 'AdamW'
        temp_args.scheduler = 'ReduceLROnPlateau'
        temp_args.weight_decay = 1e-4
        temp_args.scheduler_milestones = '1000,2000,3000'
        temp_args.scheduler_gamma = 0.5
        temp_args.lr = 1e-4
        temp_args.pretrained = None
        temp_args.model_dir = None
        temp_args.test_snr_list = None
        temp_args.test_snr_chunk = None
        temp_args.training = False
        
        config = Config(temp_args)
        seed_torch(config.seed)
        
        # 모델 생성 및 로드
        from model_utils import setup_model
        net = setup_model(temp_args, config, logger)
        load_weights(net.module if isinstance(net, nn.DataParallel) else net, 
                    model_path, strict=False)
        
        # 데이터 로더 생성
        _, test_loader = get_loader(temp_args, config)
        
        # 평가
        results = evaluate_model(net, test_loader, snr_list, config, temp_args, logger)
        all_results[mode] = results
        
        logger.info(f"Evaluation completed for {mode}")
    
    # 플롯 생성
    logger.info("\nGenerating plots...")
    plot_snr_vs_performance(all_results, args.save_dir, logger)
    plot_cbr_vs_performance_10dB(all_results, args.save_dir, logger)
    
    # 샘플 이미지 저장
    logger.info("\nSaving sample images...")
    save_sample_images(all_results, args.save_dir, logger)
    
    logger.info(f"\n{'='*80}")
    logger.info("All evaluations completed!")
    logger.info(f"Results saved to: {args.save_dir}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
