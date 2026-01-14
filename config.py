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
DEFAULT_TOTAL_EPOCHS = 1500
DEFAULT_PRINT_STEP = 50
DEFAULT_SAVE_MODEL_FREQ = 50
DEFAULT_BATCH_SIZE_PER_GPU = 8

# Packet Size
DEFAULT_PACKET_SIZE = 16
DEFAULT_NOISELESS_EPOCH_SIZE = 0  # Number of epochs to use noiseless channel before switching to train_snr_list


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
        '--train_snr_list', 
        type=str, 
        default='-10,-5, 0, 5, 10', 
        help='Training SNR (dB)'
    )
    
    # Progressive Strategy
    parser.add_argument(
        '--progressive_mode', 
        type=str, 
        default='full',
        choices=[
            # New 7 training strategies (Table)
            'full',          # a) Full
            'full_m',        # b) Full_m
            'full_dual',     # c) Full_dual
            'full_part',     # d) Full_part
            'full_part_m',   # e) Full_part_m
            'mask_only',     # f) Mask_only
            'hybrid_all',    # g) hybrid_all
            'chunk1_full',   # h) chunk 1개를 받았을 때 + 전체 받았을 때
            # Legacy names (mapped internally for backward compatibility)
            'off',
            'rand_mask_1',
            'rand_mask_2',
        ],
        help=(
            "Progressive mode / training strategy. "
            "New modes: "
            "'full', 'full_m', 'full_dual', 'full_part', 'full_part_m', 'mask_only', 'hybrid_all', 'chunk1_full'. "
            "Legacy aliases: 'off' -> 'full', 'rand_mask_2' -> 'full_part'."
        )
    )
    parser.add_argument(
        '--mask_prob',
        type=float,
        default=0.1,
        help='Probability that a random masking error occurs during training (e.g., 0.1 = 10%).'
    )
    parser.add_argument(
        '--packet_size', 
        type=int, 
        default = DEFAULT_PACKET_SIZE, 
        help='Number of features per transmission step (F)'
    )
    
    # Architecture Settings
    parser.add_argument(
        '--embed_dims',
        type=str,
        default='128,192,256,320',
        help='Embedding dimensions for encoder stages (comma-separated). Decoder uses reverse order. Default: 128,192,256,320'
    )
    parser.add_argument(
        '--depths',
        type=str,
        default='2,2,6,2',
        help='Depths (number of blocks) for encoder stages (comma-separated). Decoder uses reverse order. Default: 2,2,6,2'
    )
    parser.add_argument(
        '--num_heads',
        type=str,
        default='4,6,8,10',
        help='Number of attention heads for encoder stages (comma-separated). Decoder uses reverse order. Default: 4,6,8,10'
    )
    parser.add_argument(
        '--transmitted_dim',
        type=int,
        default=320,
        help='Transmitted channel dimension (C\') after MLP projection. If None, uses encoder output dimension (C). Default: 320'
    )
    
    # Loss Function Settings
    parser.add_argument(
        '--loss_weights',
        type=str,
        default='10,1',
        help='Loss weights for [lambda_1 (MSE), lambda_2 (MS-SSIM)]. Loss = lambda_1 * MSE - lambda_2 * MS-SSIM. Default: 0.9,0.1. Note: Common ratios in literature range from 1:1 to 10:1 (MSE:MS-SSIM), with 0.8:0.2 being a balanced choice.'
    )
    
    # Optimizer and Scheduler Settings
    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        choices=['Adam', 'AdamW'],
        help='Optimizer type. Default: AdamW'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='ReduceLROnPlateau',
        choices=['Cosine', 'MultiStep', 'ReduceLROnPlateau'],
        help='Learning rate scheduler type. Default: ReduceLROnPlateau'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer. Default: 1e-4'
    )
    parser.add_argument(
        '--scheduler_milestones',
        type=str,
        default='1000,2000,3000',
        help='Milestones for MultiStepLR scheduler (comma-separated epoch numbers). Default: 1000,2000,3000'
    )
    parser.add_argument(
        '--scheduler_gamma',
        type=float,
        default=0.5,
        help='Gamma (LR decay factor) for MultiStepLR scheduler. Default: 0.5'
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
        default='-10,-5, 0, 5, 10',
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
        
        # Parse train_snr_list
        if hasattr(args, 'train_snr_list') and args.train_snr_list:
            self.train_snr_list = [float(x.strip()) for x in args.train_snr_list.split(',')]
        else:
            self.train_snr_list = [10.0]  # Default single SNR
        
        # Check if test mode
        is_test_mode = not hasattr(args, 'training') or not args.training
        
        if is_test_mode:
            self.workdir = (
                f'{self.base_save_path}/'
                f'test_{args.testset}_{args.progressive_mode}_packet{args.packet_size}/'
                f'{timestamp}'
            )
        else:
            snr_str = f"SNR{min(self.train_snr_list):.0f}to{max(self.train_snr_list):.0f}"
            self.workdir = (
                f'{self.base_save_path}/'
                f'{args.trainset}_{snr_str}_{args.progressive_mode}/'
                f'{timestamp}'
            )
        
        self.log = os.path.join(self.workdir, f'Log_{timestamp}.log')
        self.samples = os.path.join(self.workdir, 'samples')
        self.models = os.path.join(self.workdir, 'models')
        self.logger = None
        
        # Training hyperparameters
        self.learning_rate = args.lr
        self.tot_epoch = DEFAULT_TOTAL_EPOCHS
        self.print_step = DEFAULT_PRINT_STEP
        self.save_model_freq = DEFAULT_SAVE_MODEL_FREQ
        self.batch_size = DEFAULT_BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        self.progressive_mode = args.progressive_mode
        # Probability that random masking error occurs (used in progressive training modes)
        self.mask_prob = getattr(args, 'mask_prob', 0.1)
        
        # Parse architecture parameters
        embed_dims = [int(x.strip()) for x in args.embed_dims.split(',')]
        depths = [int(x.strip()) for x in args.depths.split(',')]
        num_heads = [int(x.strip()) for x in args.num_heads.split(',')]
        
        # Transmitted dimension (C'): if None, use encoder output dimension (C)
        if args.transmitted_dim is None:
            self.transmitted_dim = embed_dims[-1]  # Use last encoder dimension
        else:
            self.transmitted_dim = args.transmitted_dim
        
        # Validate architecture parameters
        if len(embed_dims) != len(depths) or len(embed_dims) != len(num_heads):
            raise ValueError(
                f"embed_dims, depths, and num_heads must have the same length. "
                f"Got embed_dims={len(embed_dims)}, depths={len(depths)}, num_heads={len(num_heads)}"
            )
        
        # Model settings
        common_kwargs = dict(
            img_size=(256, 256),
            patch_size=2,
            in_chans=3,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            model='E2E',
            use_checkpoint=True
        )
        
        # Encoder settings: user-specified or default [128, 192, 256, 320] channels, [2, 2, 6, 2] depths
        self.encoder_kwargs = dict(
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            transmitted_dim=self.transmitted_dim,
            **common_kwargs
        )
        
        # Decoder settings: symmetric to encoder (reverse order)
        self.decoder_kwargs = dict(
            embed_dims=list(reversed(embed_dims)),
            depths=list(reversed(depths)),
            num_heads=list(reversed(num_heads)),
            transmitted_dim=self.transmitted_dim,
            **common_kwargs
        )