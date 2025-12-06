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
        self.packet_size = 1

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

        # =========================================================
        # [Fast Path] Mode: 'all' (Base Model Training)
        # 루프 없이 한 번에 보내고 복원합니다.
        # =========================================================
        if self.args.progressive_mode == 'all':
            # a) Power Normalization (전체)
            tx_norm = self.power_normalize(feat_all)
            
            # b) Channel
            # (B, Seq, C) -> Flatten -> Channel -> Reshape
            tx_flat = tx_norm.reshape(B, -1)
            rx_flat = self.channel(tx_flat, snr, avg_pwr=True)
            rx_all = rx_flat.reshape(B, Seq, -1)
            
            # c) Decoding
            recon = self.decoder(rx_all)
            
            # d) Result Formatting
            mse_val = nn.MSELoss()(recon, input_image)
            
            return {
                'mse': [mse_val],
                'recon_img': [recon]
            }
        # =========================================================
        # [Slow Path] Mode: 'progressive' or 'adaptive'
        # Feature를 하나씩 보내며 누적 복원합니다.
        # =========================================================
        else:
            feat_received_buffer = torch.zeros_like(feat_all)
            
            mse_losses = []
            recon_imgs = []

            # 0부터 Total Channel까지 1씩 증가 (packet_size=1)
            for c_idx in range(0, C_total, self.packet_size):
                c_start = c_idx
                c_end = c_idx + self.packet_size
                
                # a) Packet Extraction
                tx_chunk = feat_all[:, :, c_start:c_end]
                
                # b) Power Normalization
                tx_norm = self.power_normalize(tx_chunk)
                
                # c) Channel
                tx_flat = tx_norm.reshape(B, -1)
                rx_flat = self.channel(tx_flat, snr, avg_pwr=True)
                rx_chunk = rx_flat.reshape(B, Seq, -1)
                
                # d) Buffer Update
                feat_received_buffer[:, :, c_start:c_end] = rx_chunk
                
                # e) Decoding
                recon = self.decoder(feat_received_buffer)
                if self.save_start <= c_end <= self.save_end:
                    recon_imgs.append(recon)
                
                # f) Loss
                mse_val = nn.MSELoss()(recon, input_image)
                mse_losses.append(mse_val)

            return {
                'mse': mse_losses,           
                'recon_img': recon_imgs  
            }