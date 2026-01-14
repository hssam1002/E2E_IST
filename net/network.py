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

        # Transmitted dimension (C') - encoder/decoder 내부에서 MLP 처리
        self.transmitted_dim = config.transmitted_dim
        
        # For backward compatibility
        self.total_channels = self.transmitted_dim
        
        # Log network configuration
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info(f"Encoder: {encoder_kwargs}")
            config.logger.info(f"Decoder: {decoder_kwargs}")

        # Progressive Steps
        self.packet_size = args.packet_size
        # Random masking (error) probability (Config에서 arg를 통해 설정됨)
        self.mask_prob = config.mask_prob

        # Chunk 관련 파라미터는 transmitted_dim과 packet_size에만 의존하므로
        # 한 번만 계산해두고 forward에서 재사용 (속도/안정성 향상)
        self.L = (self.transmitted_dim + self.packet_size - 1) // self.packet_size  # 총 chunk 수
        self.chunk_ranges = []
        for i in range(self.L):
            c_start = i * self.packet_size
            c_end = min((i + 1) * self.packet_size, self.transmitted_dim)
            self.chunk_ranges.append((c_start, c_end))

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

        # 1. Feature Extraction: 이미지를 특징 벡터로 인코딩 및 MLP Projection
        # Encoder 내부에서 MLP가 처리됨: (B, Seq, C) -> (B, Seq, C')
        feat_transmitted = self.encoder(input_image)  # (B, Seq, C')
        B, Seq, C_prime = feat_transmitted.shape

        # 전력 정규화 및 채널 통과 (정렬 없이 원본 사용)
        tx_norm = self.power_normalize(feat_transmitted)
        rx_all = self.channel(tx_norm, snr, avg_pwr=True)  # (B, Seq, C')

        # ------------------------------------------------------------------
        # Chunk 정의
        #   - C' 차원을 packet_size 기준으로 나눔
        #   - self.L: 총 chunk 개수 (초기화 시 계산)
        #   - ℓ: {1, ..., self.L} 중 랜덤 샘플 (partial 전송 시 사용)
        # ------------------------------------------------------------------
        # 안전을 위해 encoder 출력 채널 수와 transmitted_dim이 일치하는지 확인
        if C_prime != self.transmitted_dim:
            raise ValueError(
                f"Encoder output dim (C'={C_prime}) must match transmitted_dim ({self.transmitted_dim})"
            )

        L = self.L
        # ℓ 선택 (1~L)
        ell = np.random.randint(1, L + 1)
        ell_end = self.chunk_ranges[ell - 1][1]

        # ------------------------------------------------------------------
        # rmask(·): chunk 단위로 10% 확률로 0으로 만드는 랜덤 마스킹 함수
        #   - mask_prob: 각 chunk가 "에러"로 0이 될 확률 (기본 0.1)
        #   - rmask(z_{≤L}): L개 모든 chunk에 대해 독립적으로 적용
        #   - rmask(z_{≤ℓ}): 1~ℓ chunk에만 적용 (그 이후는 원래도 0)
        # ------------------------------------------------------------------
        def make_packet_mask(num_active_chunks):
            """
            Args:
                num_active_chunks (int): rmask를 적용할 chunk 수 (예: L 또는 ℓ)
            Returns:
                torch.Tensor: (1, 1, C') shape의 float mask (0 또는 1)
            """
            # 기본은 모두 0 (전송 안 된 chunk)
            mask_chunks = np.zeros(L, dtype=np.float32)
            # 활성화된 chunk(1~num_active_chunks)에 대해 rmask 적용
            if num_active_chunks > 0:
                # 각 chunk마다 1 - mask_prob 확률로 정상 전송(1), mask_prob 확률로 0
                active = (np.random.rand(num_active_chunks) > self.mask_prob).astype(np.float32)
                mask_chunks[:num_active_chunks] = active

            # chunk mask를 channel 차원으로 펼치기
            mask_full = np.repeat(mask_chunks, self.packet_size)
            mask_full = mask_full[:C_prime]  # 마지막 chunk는 잘릴 수 있음
            mask_full = torch.from_numpy(mask_full).to(rx_all.device).view(1, 1, C_prime)
            return mask_full

        def apply_mask(x, num_active_chunks):
            """rmask()를 적용한 특징 벡터 반환"""
            mask = make_packet_mask(num_active_chunks)
            return x * mask
        
        # ============================================================================
        # Progressive Mode에 따른 처리
        # ============================================================================
        
        # Force progressive: 모든 mode에서 chunk-by-chunk progressive decoding (validate용)
        if force_progressive:
            feat_received_buffer = torch.zeros_like(rx_all)  # (B, Seq, C')
            mse_losses = []
            recon_imgs = []

            # 순차 전송 (원래 순서)
            for c_idx in range(0, C_prime, self.packet_size):
                c_start = c_idx
                c_end = min(c_idx + self.packet_size, C_prime)
                
                # Buffer 업데이트: 새로운 패킷 추가
                feat_received_buffer = feat_received_buffer.clone()
                feat_received_buffer[:, :, c_start:c_end] = rx_all[:, :, c_start:c_end]
                
                # Decoding (Decoder 내부에서 MLP가 처리됨: C' -> C)
                recon = self.decoder(feat_received_buffer)
                mse_val = nn.MSELoss()(recon, input_image)
                mse_losses.append(mse_val)
                recon_imgs.append(recon)

            return {
                'mse': mse_losses,
                'recon_img': recon_imgs
            }
        
        # ------------------------------------------------------------------
        # Training strategy / progressive_mode에 따른 조합
        #
        # 표의 Objective에 맞게 decoder를 여러 번 호출해서
        #   d_s(z_L), d_s(rmask(z_L)), d_s(z_ℓ), d_s(rmask(z_ℓ))
        # 에 해당하는 reconstruction들을 모두 반환한다.
        # compute_loss()는 이 리스트에 대해 동일한 손실을 계산해 합산한다.
        # ------------------------------------------------------------------

        # Canonical mode 이름 정리 (legacy alias 지원)
        mode = self.args.progressive_mode
        if mode == 'off':
            mode = 'full'          # a) Full
        elif mode == 'rand_mask_2':
            mode = 'full_part'     # d) Full_part (과거 rand_mask_2와 유사)

        # Decoder 호출 횟수를 줄이기 위해 lazy evaluation + 캐시 사용
        criterion = nn.MSELoss()
        recon_cache = {}  # key -> (mse, recon)

        # 공통적으로 사용할 feature들 (Decoder 입력)
        #   - z_L        : rx_all
        #   - z_ℓ        : feat_partial
        #   - z_1        : feat_chunk1 (첫 번째 chunk만 받은 경우)
        #   - rmask(z_L) : feat_full_m
        #   - rmask(z_ℓ) : feat_partial_m
        feat_partial = torch.zeros_like(rx_all)
        feat_partial[:, :, :ell_end] = rx_all[:, :, :ell_end]
        # chunk 1개만 받은 경우 (첫 번째 chunk)
        chunk1_end = self.chunk_ranges[0][1]
        feat_chunk1 = torch.zeros_like(rx_all)
        feat_chunk1[:, :, :chunk1_end] = rx_all[:, :, :chunk1_end]
        feat_full_m = apply_mask(rx_all, num_active_chunks=L)
        feat_partial_m = apply_mask(feat_partial, num_active_chunks=ell)

        def get_recon(key, feat):
            """Decoder를 필요한 경우에만 한 번 호출하고, (mse, recon)을 캐시에서 재사용"""
            if key not in recon_cache:
                recon = self.decoder(feat)
                mse = criterion(recon, input_image)
                recon_cache[key] = (mse, recon)
            return recon_cache[key]

        # 결과 리스트
        mse_list = []
        recon_list = []

        # ----------------------
        # a) Full
        #   Objective: min d_s(z_L)
        # ----------------------
        if mode == 'full':
            mse_full, recon_full = get_recon('full', rx_all)
            mse_list.append(mse_full)
            recon_list.append(recon_full)
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # b) Full_m
        #   Objective: min d_s(rmask(z_L))
        # ----------------------
        if mode == 'full_m':
            mse_full_m, recon_full_m = get_recon('full_m', feat_full_m)
            mse_list.append(mse_full_m)
            recon_list.append(recon_full_m)
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # c) Full_dual
        #   Objective: min d_s(z_L) + d_s(rmask(z_L))
        # ----------------------
        if mode == 'full_dual':
            mse_full, recon_full = get_recon('full', rx_all)
            mse_full_m, recon_full_m = get_recon('full_m', feat_full_m)
            mse_list.extend([mse_full, mse_full_m])
            recon_list.extend([recon_full, recon_full_m])
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # d) Full_part
        #   Objective: min d_s(z_L) + d_s(z_ℓ)
        # ----------------------
        if mode == 'full_part':
            mse_full, recon_full = get_recon('full', rx_all)
            mse_partial, recon_partial = get_recon('partial', feat_partial)
            mse_list.extend([mse_full, mse_partial])
            recon_list.extend([recon_full, recon_partial])
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # e) Full_part_m
        #   Objective: min d_s(z_L) + d_s(rmask(z_ℓ))
        # ----------------------
        if mode == 'full_part_m':
            mse_full, recon_full = get_recon('full', rx_all)
            mse_partial_m, recon_partial_m = get_recon('partial_m', feat_partial_m)
            mse_list.extend([mse_full, mse_partial_m])
            recon_list.extend([recon_full, recon_partial_m])
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # f) Mask_only
        #   Objective: min d_s(rmask(z_L)) + d_s(rmask(z_ℓ))
        # ----------------------
        if mode == 'mask_only':
            mse_full_m, recon_full_m = get_recon('full_m', feat_full_m)
            mse_partial_m, recon_partial_m = get_recon('partial_m', feat_partial_m)
            mse_list.extend([mse_full_m, mse_partial_m])
            recon_list.extend([recon_full_m, recon_partial_m])
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # g) hybrid_all
        #   Objective: min d_s(z_L) + d_s(rmask(z_L)) + d_s(rmask(z_ℓ))
        # ----------------------
        if mode == 'hybrid_all':
            mse_full, recon_full = get_recon('full', rx_all)
            mse_full_m, recon_full_m = get_recon('full_m', feat_full_m)
            mse_partial_m, recon_partial_m = get_recon('partial_m', feat_partial_m)
            mse_list.extend([mse_full, mse_full_m, mse_partial_m])
            recon_list.extend([recon_full, recon_full_m, recon_partial_m])
            return {'mse': mse_list, 'recon_img': recon_list}

        # ----------------------
        # h) chunk1_full
        #   Objective: min d_s(z_1) + d_s(z_L)
        # ----------------------
        if mode == 'chunk1_full':
            mse_chunk1, recon_chunk1 = get_recon('chunk1', feat_chunk1)
            mse_full, recon_full = get_recon('full', rx_all)
            mse_list.extend([mse_chunk1, mse_full])
            recon_list.extend([recon_chunk1, recon_full])
            return {'mse': mse_list, 'recon_img': recon_list}

        raise ValueError(
            f"Unsupported progressive_mode: {self.args.progressive_mode} (canonical: {mode}). "
            f"Supported modes: 'full', 'full_m', 'full_dual', 'full_part', 'full_part_m', 'mask_only', 'hybrid_all', 'chunk1_full' "
            f"(plus legacy aliases 'off', 'rand_mask_2')."
        )