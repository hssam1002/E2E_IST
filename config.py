"""
Configuration and argument parsing for E2E-IST
"""

import argparse
import os
import torch
import torch.nn as nn
from datetime import datetime

# Constants
DEFAULT_SEED = 42
DEFAULT_TOTAL_EPOCHS = 10000
DEFAULT_PRINT_STEP = 100
DEFAULT_SAVE_MODEL_FREQ = 20
DEFAULT_BATCH_SIZE_PER_GPU = 8
DEFAULT_PACKET_SIZE = 16  # Default packet size for progressive transmission


def setup_argument_parser():
    """Setup command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='E2E-IST for Progressive Image Transmission'
    )
    
    # Training Mode
    parser.add_argument(
        '--training', 
        action='store_true', 
        help='Enable training mode'
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=100, 
        help='Early stopping patience (epochs)'
    )
    parser.add_argument(
        '--pretrained', 
        type=str, 
        default=None, 
        help='Path to pretrained model (.pth)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4, 
        help='Learning Rate'
    )
    
    # Dataset Settings
    parser.add_argument(
        '--trainset', 
        type=str, 
        default='DIV2K', 
        help='Training dataset'
    )
    parser.add_argument(
        '--testset', 
        type=str, 
        default='Kodak', 
        choices=['Kodak', 'CLIC2021', 'DIV2K'],
        help='Test dataset'
    )
    
    # Model & Channel Settings
    parser.add_argument(
        '--channel_type', 
        type=str, 
        default='awgn', 
        choices=['awgn', 'rayleigh', 'noiseless'],
        help='Channel model type'
    )
    parser.add_argument(
        '--train_snr', 
        type=int, 
        default=10, 
        help='Training SNR (dB)'
    )
    
    # Learning Strategy
    parser.add_argument(
        '--learning_mode', 
        type=str, 
        default='rand_mask_2',
        choices=['non-progressive', 'rand_mask_1', 'rand_mask_2'],
        help='Learning mode strategy'
    )
    
    # Model Architecture Settings
    # Default trial (1)
    # Embed: 128, 192, 256, 320
    # patch_size: 2, 2, 2, 2
    # Num_heads: 4, 6, 8, 10
    # Depths: 2, 2, 6, 2
    # Window_size: 8, 8, 8, 8

    # Default trial (2)
    # Embed: 256, 256, 256, 256
    # patch_size: 8, 2, 2, 2
    # Num_heads: 4, 4, 4, 4
    # Depths: 2, 2, 4, 2
    # Window_size: 32, 16, 8, 4

    parser.add_argument(
        '--emb_dim',
        type=str,
        default='128,192,256,320',
        help='Embedding dimensions for encoder stages (comma-separated). Decoder uses reverse order.'
    )
    parser.add_argument(
        '--patch_embed_size',
        type=int,
        default=2,
        help='Patch size for PatchEmbed (initial embedding layer). PatchMerging stages always use 2x downsampling. Default: 2'
    )
    parser.add_argument(
        '--num_heads',
        type=str,
        default='4,6,8,10',
        help='Number of attention heads for encoder stages (comma-separated). Decoder uses reverse order.'
    )
    parser.add_argument(
        '--depths',
        type=str,
        default='2,2,6,2',
        help='Depths (number of blocks) for encoder stages (comma-separated). If None, uses default [2,2,6,2] for 4 stages or [2]*N for N stages. Decoder uses reverse order.'
    )
    parser.add_argument(
        '--window_size',
        type=str,
        default='8,8,8,8',
        help='Window size for Swin Transformer attention (comma-separated for each stage). If None, uses single value 8 for all stages. Default: None'
    )
    parser.add_argument(
        '--use_checkpoint',
        action='store_true',
        help='Enable gradient checkpointing to save memory (slower but uses less GPU memory)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size per GPU. If None, uses default (8). Total batch size = batch_size * num_gpus'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers. Default: 8'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps. Effective batch size = batch_size * gradient_accumulation_steps. Default: 1'
    )
    parser.add_argument(
        '--packet_size',
        type=int,
        default=DEFAULT_PACKET_SIZE,
        help='Packet size for progressive transmission. Default: 16'
    )
 
    # Loss Function Settings
    parser.add_argument(
        '--loss_weights',
        type=str,
        default='1.0,0.1',
        help='Loss weights for [MSE, MS-SSIM] (comma-separated). Default: 1.0,1.0'
    )
    # Test Mode Settings
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Directory containing test models'
    )
    parser.add_argument(
        '--test_snr_list',
        type=str,
        default=None,
        help='Comma-separated SNR values for SNR performance test (e.g., "-5,0,5,10,15,20"). If provided, performs SNR sweep test.'
    )
    return parser

class Config:
    """Configuration class for training and model settings"""
    
    def __init__(self, args):
        """
        Args:
            args: Parsed command-line arguments
        """
        # Basic settings
        self.seed = DEFAULT_SEED
        self.CUDA = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Path settings
        self.base_save_path = "/data4/hongsik/E2E_IST/results"
        self.train_data_dir = "/data4/hongsik/data/DIV2K"
        self.test_data_dir = f"/data4/hongsik/data/{args.testset}"
        
        # Working directory setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = timestamp
        
        # Check if test mode
        is_test_mode = not hasattr(args, 'training') or not args.training
        
        if is_test_mode:
            self.workdir = (
                f'{self.base_save_path}/'
                f'test_{args.testset}_{args.learning_mode}/'
                f'{timestamp}'
            )
        else:
            self.workdir = (
                f'{self.base_save_path}/'
                f'{args.trainset}_SNR{args.train_snr}_{args.learning_mode}/'
                f'{timestamp}'
            )
        
        self.log = os.path.join(self.workdir, f'Log_{timestamp}.log')
        self.samples = os.path.join(self.workdir, 'samples')
        self.models = os.path.join(self.workdir, 'models')
        self.logger = None
        
        # Training hyperparameters
        self.learning_rate = args.lr
        self.lr = args.lr  # Alias for compatibility
        self.tot_epoch = 5000  # Fixed to 5000 as requested
        self.print_step = DEFAULT_PRINT_STEP
        self.save_model_freq = DEFAULT_SAVE_MODEL_FREQ
        # Batch size
        batch_size_per_gpu = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE_PER_GPU
        self.batch_size = batch_size_per_gpu * torch.cuda.device_count()
        self.num_workers = args.num_workers
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.packet_size = args.packet_size
        
        # Parse embedding dimensions and num_heads from arguments
        embed_dims = [int(x.strip()) for x in args.emb_dim.split(',')]
        num_heads = [int(x.strip()) for x in args.num_heads.split(',')]
        
        if len(embed_dims) != len(num_heads):
            raise ValueError(f"emb_dim and num_heads must have the same length. Got {len(embed_dims)} and {len(num_heads)}")
        
        # patch_embed_size: PatchEmbed에만 사용 (기본값 2)
        patch_embed_size = args.patch_embed_size
        
        # Parse depths from arguments
        depths = [int(x.strip()) for x in args.depths.split(',')]
        if len(depths) != len(embed_dims):
            raise ValueError(f"depths must have the same length as emb_dim. Got {len(depths)} and {len(embed_dims)}")
        
        # Parse window_size from arguments
        window_sizes = [int(x.strip()) for x in args.window_size.split(',')]
        if len(window_sizes) != len(embed_dims):
            raise ValueError(f"window_size must have the same length as emb_dim. Got {len(window_sizes)} and {len(embed_dims)}")
        
        # Model settings (window_size는 stage별로 다를 수 있으므로 common_kwargs에서 제외)
        common_kwargs = dict(
            img_size=(256, 256),
            in_chans = 3,
            mlp_ratio = 4.0,
            qkv_bias = True,
            qk_scale = None,
            norm_layer = nn.LayerNorm,
            patch_norm = True,
            model='E2E',
            use_checkpoint=args.use_checkpoint
        )
        
        # Encoder settings: user-specified parameters (window_size는 list로 전달)
        self.encoder_kwargs = dict(
            embed_dims=embed_dims,
            patch_embed_size=patch_embed_size,  # PatchEmbed에만 사용
            depths=depths,
            num_heads=num_heads,
            window_size=window_sizes,  # List of window sizes for each stage
            **common_kwargs
        )
        
        # Decoder settings: symmetric to encoder (reverse order)
        # Decoder는 PatchEmbed를 사용하지 않으므로 patch_embed_size는 무시됨
        self.decoder_kwargs = dict(
            embed_dims=list(reversed(embed_dims)),
            patch_embed_size=patch_embed_size,  # Decoder에서는 사용되지 않지만 호환성을 위해 포함
            depths=list(reversed(depths)),
            num_heads=list(reversed(num_heads)),
            window_size=list(reversed(window_sizes)),  # Reverse order for decoder
            **common_kwargs
        )