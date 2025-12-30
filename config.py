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

# Packet Size
DEFAULT_PACKET_SIZE = 16


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
        default='rand_mask_2',
        choices=['off', 'rand_mask_1', 'rand_mask_2'],
        help='Progressive mode strategy'
    )
    parser.add_argument(
        '--packet_size', 
        type=int, 
        default = DEFAULT_PACKET_SIZE, 
        help='Number of features per transmission step (F)'
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
    parser.add_argument(
        '--test_snr_chunk',
        type=int,
        default=None,
        help='Chunk number to measure for SNR test. If None, measures at final chunk.'
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
        
        # Encoder settings: [128, 192, 256, 320] channels, [2, 2, 6, 2] depths
        self.encoder_kwargs = dict(
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            **common_kwargs
        )
        
        # Decoder settings: symmetric to encoder
        self.decoder_kwargs = dict(
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            **common_kwargs
        )