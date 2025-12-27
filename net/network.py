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
from loss.distortion import MS_SSIM

# Global flag to control recon_img saving in force_progressive mode
SAVE_RECON_IMG = False

def set_save_recon_img(enabled):
    """Set global flag to enable/disable saving recon_img in force_progressive mode"""
    global SAVE_RECON_IMG
    SAVE_RECON_IMG = enabled

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
    
    Learning Mode:
    - 'non-progressive': Non-progressive 모드 (전체 특징을 한 번에 전송)
    - 'rand_mask_1': 랜덤 마스킹 기반 학습 (단일 패킷)
    - 'rand_mask_2': 랜덤 마스킹 기반 학습 (부분 + 전체)
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

        self.total_channels = config.encoder_kwargs['embed_dims'][-1] # C_total
        
        # Log network configuration
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info(f"Encoder: {encoder_kwargs}")
            config.logger.info(f"Decoder: {decoder_kwargs}")

        # Progressive Steps: packet_size from config
        self.packet_size = config.packet_size

        # Resolution Info
        self.H = self.W = 0
        # downsample_ratio는 encoder의 최종 feature map 해상도를 계산하는 데 사용
        # encoder 초기화 후 encoder.H, encoder.W를 직접 사용하도록 변경
        # 여기서는 임시값 설정 (실제로는 forward에서 encoder의 해상도 사용)
        self.downsample_ratio = None  # encoder 초기화 후 업데이트

    def power_normalize(self, x, target_frob_norm_sq=None):
        """
        특징 벡터를 배치별 전력 정규화합니다.
        
        각 배치별로 Frobenius norm^2을 target_frob_norm_sq로 정규화합니다.
        target_frob_norm_sq가 None이면 Seq * C로 정규화합니다.
        
        Args:
            x (torch.Tensor): 입력 특징 텐서 (B, Seq, C)
            target_frob_norm_sq (float, optional): 목표 Frobenius norm^2. None이면 Seq * C 사용
        
        Returns:
            torch.Tensor: 전력 정규화된 특징 텐서
        """
        B, Seq, C = x.shape
        
        # 목표 Frobenius norm^2 설정
        if target_frob_norm_sq is None:
            target_frob_norm_sq = Seq * C
        
        # 각 배치별 Frobenius norm^2 계산
        # (B, Seq, C) -> 각 배치 b에 대해 (b, :, :)의 Frob norm^2
        frob_norm_sq = torch.sum(x ** 2, dim=(1, 2), keepdim=True)  # (B, 1, 1)
        
        # 정규화: 각 배치를 sqrt(target_frob_norm_sq / actual_frob_norm_sq)로 스케일링
        scale = torch.sqrt(target_frob_norm_sq / frob_norm_sq)  # (B, 1, 1)
        x_norm = x * scale
        
        return x_norm

    def forward(self, input_image, snr, force_progressive=False):
        """
        Forward pass.
        Args:
            input_image (torch.Tensor): 입력 이미지 (B, 3, H, W)
            snr (float): Signal-to-Noise Ratio (dB)
            force_progressive (bool): 모든 mode에서 chunk-by-chunk progressive decoding 강제 (validate용)
            target_cbr (float, optional): Target Channel Bandwidth Ratio. If provided, uses this CBR to determine number of channels.
        
        Returns:
            dict: 다음 키를 포함하는 딕셔너리
                - 'mse': 각 단계의 MSE loss 리스트
                - 'recon_img': 각 단계의 복원 이미지 리스트
        """
        B, _, H, W = input_image.shape
        
        # 해상도 변경 시 Encoder/Decoder 해상도 업데이트
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            # encoder의 최종 feature map 해상도 직접 사용
            ds_H = self.encoder.H
            ds_W = self.encoder.W
            self.decoder.update_resolution(ds_H, ds_W)
            self.H, self.W = H, W

        # 1. Feature Extraction: 이미지를 특징 벡터로 인코딩 (B, Seq, C_total)
        feat_all = self.encoder(input_image)
        B, Seq, C_total = feat_all.shape

        # 전력 정규화: 배치별로 Frobenius norm^2 = Seq * C_total로 정규화
        tx_norm = self.power_normalize(feat_all, target_frob_norm_sq=Seq * C_total)
        rx_all = self.channel(tx_norm, snr, avg_pwr=True)

        # Training mode에서 랜덤 채널 선택 (rand_mask_1, rand_mask_2용)
        if self.training and not force_progressive:
            # 패킷 단위로만 선택 (8, 16, 24, ..., C_total)
            # 더 안정적인 학습과 실제 사용 시나리오와의 일관성을 위해
            possible_sizes = list(range(self.packet_size, C_total + 1, self.packet_size))
            if C_total % self.packet_size != 0:
                possible_sizes.append(C_total)
            num_selected = np.random.choice(possible_sizes)
            # 0부터 num_selected-1까지의 채널 인덱스
            random_channel_indices = np.arange(num_selected)
        else:
            # Validation/Test: progressive transmission (8, 16, 24, ..., 320)
            random_channel_indices = np.arange(0, C_total, self.packet_size)

        # ============================================================================
        # Learning Mode에 따른 처리
        # ============================================================================
        # Force progressive: 모든 mode에서 chunk-by-chunk progressive decoding (validate/test용)
        if force_progressive:
            feat_received_buffer = torch.zeros_like(feat_all)
            mse_losses = []
            ms_ssim_losses = []
            recon_imgs = [] if SAVE_RECON_IMG else None
            
            # Loss weights from args
            loss_weights = [float(x.strip()) for x in self.args.loss_weights.split(',')]
            if len(loss_weights) != 2:
                raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
            w_mse, w_ms_ssim = loss_weights
            
            # MS-SSIM module (initialize once)
            ms_ssim_module = None
            if w_ms_ssim > 0:
                ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(input_image.device)

            # 순차 전송 (8, 16, 24, ..., 320)
            for c_idx in range(0, C_total, self.packet_size):
                c_start = c_idx
                c_end = min(c_idx + self.packet_size, C_total)
                
                # Buffer 업데이트: 새로운 패킷 추가
                feat_received_buffer = feat_received_buffer.clone()
                feat_received_buffer[:, :, c_start:c_end] = rx_all[:, :, c_start:c_end]
                
                # 현재 사용된 채널 수
                num_channels_used = c_end
                
                # 배치별 normalization: Frobenius norm^2 = Seq * num_channels_used
                feat_received_buffer_normalized = self.power_normalize(
                    feat_received_buffer, 
                    target_frob_norm_sq=Seq * num_channels_used
                )
                
                # 디코딩
                recon = self.decoder(feat_received_buffer_normalized)
                
                # MSE loss
                mse_val = nn.MSELoss()(recon, input_image)
                mse_losses.append(mse_val)
                
                # MS-SSIM loss (1 - ms_ssim, because ms_ssim is similarity, higher is better)
                if w_ms_ssim > 0 and ms_ssim_module is not None:
                    ms_ssim_sim = ms_ssim_module(recon, input_image)
                    # Ensure scalar (MS_SSIM may return batch-averaged value)
                    if ms_ssim_sim.dim() > 0:
                        ms_ssim_sim = ms_ssim_sim.mean()
                    ms_ssim_loss = 1.0 - ms_ssim_sim
                    ms_ssim_losses.append(ms_ssim_loss)
                else:
                    ms_ssim_losses.append(torch.tensor(0.0, device=input_image.device))
                
                # Save recon_img only if global flag is enabled
                if SAVE_RECON_IMG:
                    recon_imgs.append(recon)

            result = {
                'mse': mse_losses,
                'ms_ssim': ms_ssim_losses
            }
            if SAVE_RECON_IMG:
                result['recon_img'] = recon_imgs
            return result
        
        # Mode: 'non-progressive'
        # 모든 channel을 다 전송하고 복원
        if self.args.learning_mode == 'non-progressive':
            # 배치별 normalization: Frobenius norm^2 = Seq * C_total
            rx_all_normalized = self.power_normalize(rx_all, target_frob_norm_sq=Seq * C_total)
            recon = self.decoder(rx_all_normalized)
            
            # Loss weights from args
            loss_weights = [float(x.strip()) for x in self.args.loss_weights.split(',')]
            if len(loss_weights) != 2:
                raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
            w_mse, w_ms_ssim = loss_weights
            
            # MSE loss
            mse_val = nn.MSELoss()(recon, input_image)
            
            # MS-SSIM loss (1 - ms_ssim, because ms_ssim is similarity, higher is better)
            ms_ssim_loss = torch.tensor(0.0, device=input_image.device)
            if w_ms_ssim > 0:
                ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(input_image.device)
                ms_ssim_sim = ms_ssim_module(recon, input_image)
                # Ensure scalar (MS_SSIM may return batch-averaged value)
                if ms_ssim_sim.dim() > 0:
                    ms_ssim_sim = ms_ssim_sim.mean()
                ms_ssim_loss = 1.0 - ms_ssim_sim
            
            return {
                'mse': [mse_val],
                'ms_ssim': [ms_ssim_loss],
                'recon_img': [recon]
            }
        
        # Mode: 'rand_mask_1'
        # 랜덤하게 1부터 C_total 중 하나를 선택하고, 1부터 randi까지의 channel로 decoding
        elif self.args.learning_mode == 'rand_mask_1':
            # random_channel_indices는 이미 0부터 num_selected-1까지 설정됨
            num_selected = len(random_channel_indices)
            
            # 선택된 채널만 남기고 나머지는 0으로 마스킹
            feat_received_buffer = torch.zeros_like(rx_all)
            feat_received_buffer[:, :, random_channel_indices] = rx_all[:, :, random_channel_indices]
            
            # 배치별 normalization: Frobenius norm^2 = Seq * num_selected
            feat_received_buffer_normalized = self.power_normalize(
                feat_received_buffer,
                target_frob_norm_sq=Seq * num_selected
            )
            
            # 디코딩 및 Loss 계산
            recon = self.decoder(feat_received_buffer_normalized)
            
            # Loss weights from args
            loss_weights = [float(x.strip()) for x in self.args.loss_weights.split(',')]
            if len(loss_weights) != 2:
                raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
            w_mse, w_ms_ssim = loss_weights
            
            # MSE loss
            mse_val = nn.MSELoss()(recon, input_image)
            
            # MS-SSIM loss (1 - ms_ssim, because ms_ssim is similarity, higher is better)
            ms_ssim_loss = torch.tensor(0.0, device=input_image.device)
            if w_ms_ssim > 0:
                ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(input_image.device)
                ms_ssim_sim = ms_ssim_module(recon, input_image)
                # Ensure scalar (MS_SSIM may return batch-averaged value)
                if ms_ssim_sim.dim() > 0:
                    ms_ssim_sim = ms_ssim_sim.mean()
                ms_ssim_loss = 1.0 - ms_ssim_sim
            
            return {
                'mse': [mse_val],
                'ms_ssim': [ms_ssim_loss],
                'recon_img': [recon]
            }
        
        # Mode: 'rand_mask_2'
        # loss_total (모든 feature를 다 receive) + loss_rand (랜덤 선택된 feature)
        elif self.args.learning_mode == 'rand_mask_2':
            # Loss weights from args
            loss_weights = [float(x.strip()) for x in self.args.loss_weights.split(',')]
            if len(loss_weights) != 2:
                raise ValueError(f"loss_weights must have 2 values [MSE, MS-SSIM]. Got {len(loss_weights)}")
            w_mse, w_ms_ssim = loss_weights
            
            # MS-SSIM module (initialize once)
            ms_ssim_module = None
            if w_ms_ssim > 0:
                ms_ssim_module = MS_SSIM(data_range=1.0, levels=4, channel=3).to(input_image.device)
            
            # 1. Full Path: 모든 feature를 다 receive (non-progressive와 동일)
            # 배치별 normalization: Frobenius norm^2 = Seq * C_total
            rx_all_normalized = self.power_normalize(rx_all, target_frob_norm_sq = Seq * C_total)
            recon_full = self.decoder(rx_all_normalized)
            mse_full = nn.MSELoss()(recon_full, input_image)
            
            # MS-SSIM for full (1 - ms_ssim)
            ms_ssim_full_loss = torch.tensor(0.0, device=input_image.device)
            if w_ms_ssim > 0 and ms_ssim_module is not None:
                ms_ssim_full_sim = ms_ssim_module(recon_full, input_image)
                # Ensure scalar (MS_SSIM may return batch-averaged value)
                if ms_ssim_full_sim.dim() > 0:
                    ms_ssim_full_sim = ms_ssim_full_sim.mean()
                ms_ssim_full_loss = 1.0 - ms_ssim_full_sim

            # 2. Partial Path: 랜덤 선택된 채널만으로 복원 (rand_mask_1과 동일)
            num_selected = len(random_channel_indices)
            feat_received_buffer = torch.zeros_like(rx_all)
            feat_received_buffer[:, :, random_channel_indices] = rx_all[:, :, random_channel_indices]
            
            # 배치별 normalization: Frobenius norm^2 = Seq * num_selected
            feat_received_buffer_normalized = self.power_normalize(
                feat_received_buffer,
                target_frob_norm_sq=Seq * num_selected
            )
            
            recon_partial = self.decoder(feat_received_buffer_normalized)
            mse_partial = nn.MSELoss()(recon_partial, input_image)
            
            # MS-SSIM for partial (1 - ms_ssim)
            ms_ssim_partial_loss = torch.tensor(0.0, device=input_image.device)
            if w_ms_ssim > 0 and ms_ssim_module is not None:
                ms_ssim_partial_sim = ms_ssim_module(recon_partial, input_image)
                # Ensure scalar (MS_SSIM may return batch-averaged value)
                if ms_ssim_partial_sim.dim() > 0:
                    ms_ssim_partial_sim = ms_ssim_partial_sim.mean()
                ms_ssim_partial_loss = 1.0 - ms_ssim_partial_sim

            # [Partial, Full] 순서로 반환
            return {
                'mse': [mse_partial, mse_full],
                'ms_ssim': [ms_ssim_partial_loss, ms_ssim_full_loss],
                'recon_img': [recon_partial, recon_full]
            }
        else:
            raise ValueError(f"Unknown learning_mode: {self.args.learning_mode}")