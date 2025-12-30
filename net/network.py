"""
Network 모듈

End-to-End SwinJSCC 네트워크를 구현합니다.
Encoder, Channel, Decoder를 통합하여 Progressive Image Transmission을 수행합니다.
"""

from net.decoder import create_decoder
from net.encoder import create_encoder
from net.channel import Channel
import torch
import torch.nn as nn
import numpy as np

class E2E_SwinJSCC(nn.Module):
    """
    End-to-End SwinJSCC with Progressive Transmission.
    
    Swin Transformer 기반의 Encoder/Decoder와 채널 모델을 통합한
    Joint Source-Channel Coding (JSCC) 모델입니다.
    
    Progressive transmission을 지원하여 특징을 패킷 단위로 전송하고,
    수신자가 단계적으로 이미지를 복원할 수 있습니다.
    
    주요 구성 요소:
    - Encoder: 이미지를 특징 벡터로 인코딩
    - Channel: 무선 통신 채널 효과 시뮬레이션
    - Decoder: 특징 벡터를 이미지로 디코딩
    
    Progressive Mode:
    - 'off': 전체 특징을 한 번에 전송 (Non-progressive 모드)
    - 'rand_mask_1', 'rand_mask_2': 랜덤 마스킹 기반 학습
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

        # Progressive Steps
        self.packet_size = args.packet_size

        # Resolution Info
        self.H = self.W = 0
        depths = config.encoder_kwargs['depths']
        patch_size = config.encoder_kwargs.get('patch_size', 2)
        downsample_factor = patch_size * (2 ** (len(depths) - 1))
        self.downsample_ratio = int(np.log2(downsample_factor))
        
    def power_normalize(self, x):
        """
        특징 벡터를 전력 정규화합니다.
        
        각 채널별로 독립적으로 전력을 1로 정규화합니다.
        채널 전송 시 일정한 전력으로 전송하기 위함입니다.
        
        Args:
            x (torch.Tensor): 입력 특징 텐서 (B, Seq, C)
        
        Returns:
            torch.Tensor: 전력 정규화된 특징 텐서
        """
        # Sequence dimension (dim=1)에 대해 평균 전력 계산
        # 각 Channel (dim=2)별로 독립적인 Power Normalization 수행
        power = torch.mean(x ** 2, dim=1, keepdim=True)
        x_norm = x / torch.sqrt(power)
        return x_norm

    def forward(self, input_image, snr, force_progressive=False):
        """
        Forward pass.
        
        Args:
            input_image (torch.Tensor): 입력 이미지 (B, 3, H, W)
            snr (float): Signal-to-Noise Ratio (dB)
            force_progressive (bool): 모든 mode에서 chunk-by-chunk progressive decoding 강제 (validate용)
        
        Returns:
            dict: 다음 키를 포함하는 딕셔너리
                - 'mse': 각 단계의 MSE loss 리스트
                - 'recon_img': 각 단계의 복원 이미지 리스트
        """
        B, _, H, W = input_image.shape
        
        # 해상도 변경 시 Encoder/Decoder 해상도 업데이트
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            ds_H = H // (2 ** self.downsample_ratio)
            ds_W = W // (2 ** self.downsample_ratio)
            self.decoder.update_resolution(ds_H, ds_W)
            self.H, self.W = H, W

        # 1. Feature Extraction: 이미지를 특징 벡터로 인코딩 (B, Seq, C_total)
        feat_all = self.encoder(input_image)
        B, Seq, C_total = feat_all.shape

        # Random mask를 위한 패킷 인덱스 (rand_mask 모드용)
        total_packets = (C_total + self.packet_size - 1) // self.packet_size
        random_idx = np.random.randint(0, total_packets)
        
        # 전력 정규화 및 채널 통과 (정렬 없이 원본 사용)
        tx_norm = self.power_normalize(feat_all)
        rx_all = self.channel(tx_norm, snr, avg_pwr=True)

        # Random mask 종료 인덱스 계산
        c_end = (random_idx + 1) * self.packet_size
        c_end = min(c_end, C_total)
        
        # ============================================================================
        # Progressive Mode에 따른 처리
        # ============================================================================
        
        # Force progressive: 모든 mode에서 chunk-by-chunk progressive decoding (validate용)
        if force_progressive:
            feat_received_buffer = torch.zeros_like(feat_all)
            mse_losses = []
            recon_imgs = []

            # 순차 전송 (원래 순서)
            for c_idx in range(0, C_total, self.packet_size):
                c_start = c_idx
                c_end = min(c_idx + self.packet_size, C_total)
                
                # Buffer 업데이트: 새로운 패킷 추가
                feat_received_buffer = feat_received_buffer.clone()
                feat_received_buffer[:, :, c_start:c_end] = rx_all[:, :, c_start:c_end]
                
                # 디코딩
                recon = self.decoder(feat_received_buffer)
                mse_val = nn.MSELoss()(recon, input_image)
                mse_losses.append(mse_val)
                recon_imgs.append(recon)

            return {
                'mse': mse_losses,
                'recon_img': recon_imgs
            }
        
        # [Fast Path] Mode: 'off' (Non-progressive Model)
        # 전체 특징을 한 번에 전송하고 복원
        # 테스트는 force_progressive 블록에서 처리되므로 여기서는 학습 모드만 처리
        if self.args.progressive_mode == 'off':
            # 학습 모드: 전체 한 번에 전송
            recon = self.decoder(rx_all)
            mse_val = nn.MSELoss()(recon, input_image)
            return {'mse': [mse_val], 'recon_img': [recon]}
        
        # [Optimized Path] Mode: 'rand_mask_1'
        # 랜덤으로 선택된 패킷까지만 전송
        elif self.args.progressive_mode == 'rand_mask_1':
            # 뒷부분을 0으로 마스킹
            feat_received_buffer = torch.zeros_like(rx_all)
            feat_received_buffer[:, :, :c_end] = rx_all[:, :, :c_end]
            
            # 디코딩 및 Loss 계산
            recon = self.decoder(feat_received_buffer)
            mse_val = nn.MSELoss()(recon, input_image)
            
            return {'mse': [mse_val], 'recon_img': [recon]}
        
        # [Optimized Path] Mode: 'rand_mask_2'
        # Full 디코딩 1회 + Partial 마스킹 후 디코딩 1회
        elif self.args.progressive_mode == 'rand_mask_2':
            # 1. Full Path: 전체 특징으로 복원
            recon_full = self.decoder(rx_all)
            mse_full = nn.MSELoss()(recon_full, input_image)

            # 2. Partial Path: 마스킹된 특징으로 복원
            feat_received_buffer = torch.zeros_like(rx_all)
            feat_received_buffer[:, :, :c_end] = rx_all[:, :, :c_end]
            
            recon_partial = self.decoder(feat_received_buffer)
            mse_partial = nn.MSELoss()(recon_partial, input_image)

            # [Partial, Full] 순서로 반환
            return {
                'mse': [mse_partial, mse_full],
                'recon_img': [recon_partial, recon_full]
            }
        else:
            raise ValueError(
                f"Unsupported progressive_mode: {self.args.progressive_mode}. "
                f"Supported modes: 'off', 'rand_mask_1', 'rand_mask_2'"
            )