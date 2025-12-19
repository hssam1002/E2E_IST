from net.decoder import create_decoder
from net.encoder import create_encoder
from net.channel import Channel
import torch
import torch.nn as nn
import numpy as np

class E2E_SwinJSCC(nn.Module):
    """
    End-to-End SwinJSCC with Digital Modulation and Progressive Transmission.
    
    This model integrates a Swin Transformer-based Encoder/Decoder with a learnable 
    Digital Modulation Mapper. It supports progressive transmission where features 
    are transmitted in packets, allowing the receiver to reconstruct the image 
    incrementally.

    Args Explanation:
    -----------------
    args.C (int): 
        - Bits per Spatial Token. 
        - Determines how many bits are allocated per single feature token in the spatial grid.
        - Example: 16 -> Each spatial token (pixel in feature map) is converted to 16 bits.

    config.encoder_kwargs['embed_dims'][-1]:
        - Total Channel Number (C_total).
        - Total number of feature channels extracted by the Encoder (e.g., 192).
        - This determines the depth of the feature map.
    """
    def __init__(self, args, config):
        super(E2E_SwinJSCC, self).__init__()
        self.config = config
        self.args = args

        # 1. Initialization based on Config & Logger
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        
        # Create Encoder and Decoder from configs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.channel = Channel(args, config)

        self.total_channels = config.encoder_kwargs['embed_dims'][-1]
        
        # Log network configuration
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info(f"Encoder: {encoder_kwargs}")
            config.logger.info(f"Decoder: {decoder_kwargs}")

        # [Saving Range Setting]
        # args에 해당 값이 없으면 기본적으로 '마지막 단계(total_channels)'만 저장하도록 설정
        # 사용 예: args.save_start=0, args.save_end=10 -> 처음 10개 단계 저장
        self.save_start = getattr(args, 'save_start', self.total_channels)
        self.save_end = getattr(args, 'save_end', self.total_channels)

        # Progressive Steps
        self.packet_size = args.packet_size

        # Resolution Info
        self.H = self.W = 0
        depths = config.encoder_kwargs['depths']
        patch_size = config.encoder_kwargs.get('patch_size', 2)
        downsample_factor = patch_size * (2 ** (len(depths) - 1))
        self.downsample_ratio = int(np.log2(downsample_factor))
        
    def power_normalize(self, x):
        # x: (B, Seq, C) or (B, Seq, 1)
        # Feature-wise 동작을 위해 dim=1 (Sequence)에 대해서만 평균을 구합니다.
        # 결과적으로 각 Channel(dim=2) 별로 독립적인 Power Normalization이 수행됩니다.
        #power = torch.mean(x ** 2, dim = 1, keepdim=True)
        power = torch.mean(x ** 2, dim = 1, keepdim=True)
        x_norm = x / torch.sqrt(power)
        return x_norm

    def forward(self, input_image, snr):
        B, _, H, W = input_image.shape
        
        # Resolution Update
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            ds_H = H // (2 ** self.downsample_ratio)
            ds_W = W // (2 ** self.downsample_ratio)
            self.decoder.update_resolution(ds_H, ds_W)
            self.H, self.W = H, W

        # 1. Feature Extraction (B, Seq, C_total)
        feat_all = self.encoder(input_image)
        B, Seq, C_total = feat_all.shape

        total_packets = (C_total + self.packet_size - 1) // self.packet_size
        random_idx = np.random.randint(0, total_packets) # For Rand_mask
        
        # [Global Normalization & Channel]
        tx_norm = self.power_normalize(feat_all)
        rx_all = self.channel(tx_norm, snr, avg_pwr = True)

        c_end = (random_idx + 1) * self.packet_size
        c_end = min(c_end, C_total)
        
        # =========================================================
        # [Fast Path] Mode: 'all' (Base Model)
        # =========================================================
        if self.args.progressive_mode == 'all':
            recon = self.decoder(rx_all)
            mse_val = nn.MSELoss()(recon, input_image)
            return {'mse': [mse_val], 'recon_img': [recon]}
        # =========================================================
        # [Optimized Path] Mode: 'rand_mask_1'
        # =========================================================
        elif self.args.progressive_mode == 'rand_mask_1':
            # Masking (뒷부분 0으로 채움)
            feat_received_buffer = torch.zeros_like(rx_all)
            feat_received_buffer[:, :, :c_end] = rx_all[:, :, :c_end]
            
            # Decoding & Loss
            recon = self.decoder(feat_received_buffer)
            mse_val = nn.MSELoss()(recon, input_image)
            
            return {'mse': [mse_val], 'recon_img': [recon]}
        # =========================================================
        # [Optimized Path] Mode: 'rand_mask_2'
        # Full 디코딩 1회 + Partial 마스킹 후 디코딩 1회
        # =========================================================
        elif self.args.progressive_mode == 'rand_mask_2':
            # 1. Full Path
            recon_full = self.decoder(rx_all)
            mse_full = nn.MSELoss()(recon_full, input_image)

            # 2. Partial Path (rx_all 재사용)
            feat_received_buffer = torch.zeros_like(rx_all)
            feat_received_buffer[:, :, :c_end] = rx_all[:, :, :c_end]
            
            recon_partial = self.decoder(feat_received_buffer)
            mse_partial = nn.MSELoss()(recon_partial, input_image)

            # [Partial, Full] 순서로 리턴
            return {'mse': [mse_partial, mse_full], 'recon_img': [recon_partial, recon_full]}

        # =========================================================
        # [Slow Path] Mode: 'progressive', 'adaptive', 'mrl'
        # 기존 Loop 방식 유지 (중간 단계 MSE가 모두 필요하거나, ALM 제어 필요 시)
        # =========================================================
        else:
            feat_received_buffer = torch.zeros_like(feat_all)
            mse_losses = []
            recon_imgs = []

            # 0부터 Total Channel까지 1씩 증가 (packet_size=1)
            for c_idx in range(0, C_total, self.packet_size):
                c_start = c_idx
                c_end = c_idx + self.packet_size
                
                # d) Buffer Update
                feat_received_buffer = feat_received_buffer.clone()
                feat_received_buffer[:, :, c_start:c_end] = rx_all[:, :, c_start:c_end]
                
                # e) Decoding
                recon = self.decoder(feat_received_buffer)

                # f) Loss
                mse_val = nn.MSELoss()(recon, input_image)
                mse_losses.append(mse_val)

                # Save Condition
                is_in_range = (self.save_start <= c_end <= self.save_end)
                is_testing = (not self.training)
                is_last_step = (c_end >= C_total)

                if is_in_range or is_testing or is_last_step:
                    recon_imgs.append(recon)

            return {
                'mse': mse_losses,           
                'recon_img': recon_imgs  
            }