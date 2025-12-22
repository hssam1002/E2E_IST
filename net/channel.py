"""
채널 모델 모듈

무선 통신 채널의 효과를 시뮬레이션합니다.
지원하는 채널 타입: AWGN, Rayleigh, Noiseless
"""

import torch
import torch.nn as nn
import numpy as np


class Channel(nn.Module):
    """
    무선 통신 채널 모델.
    
    현재 지원하는 채널 타입:
    - 'noiseless': 오류 없는 채널 (테스트용)
    - 'awgn': Additive White Gaussian Noise 채널
    - 'rayleigh': Rayleigh 페이딩 채널
    """
    
    def __init__(self, args, config):
        """
        Args:
            args: 명령행 인자 (channel_type 속성 필요)
            config: 설정 객체 (device 속성 필요)
        """
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device

    def gaussian_noise_layer(self, input_layer, std):
        """
        AWGN (Additive White Gaussian Noise) 채널 효과를 적용합니다.
        
        Args:
            input_layer (torch.Tensor): 입력 복소수 텐서
            std (float): 노이즈의 표준편차
        
        Returns:
            torch.Tensor: 노이즈가 추가된 출력 텐서
        """
        device = input_layer.device
        
        # 복소수 가우시안 노이즈 생성
        noise_real = torch.normal(
            mean=0.0,
            std=std,
            size=input_layer.shape,
            device=device
        )
        noise_imag = torch.normal(
            mean=0.0,
            std=std,
            size=input_layer.shape,
            device=device
        )
        noise = noise_real + 1j * noise_imag
        
        return input_layer + noise
    
    def rayleigh_noise_layer(self, input_layer, std):
        """
        Rayleigh 페이딩 채널 효과를 적용합니다.
        
        Rayleigh 페이딩은 다중 경로 전파로 인한 신호 감쇠를 모델링합니다.
        
        Args:
            input_layer (torch.Tensor): 입력 복소수 텐서
            std (float): 노이즈의 표준편차
        
        Returns:
            torch.Tensor: 페이딩과 노이즈가 적용된 출력 텐서
        """
        device = input_layer.device
        
        # 복소수 가우시안 노이즈 생성
        noise_real = torch.normal(
            mean=0.0,
            std=std,
            size=input_layer.shape,
            device=device
        )
        noise_imag = torch.normal(
            mean=0.0,
            std=std,
            size=input_layer.shape,
            device=device
        )
        noise = noise_real + 1j * noise_imag

        # Rayleigh 페이딩 계수 생성
        # h = sqrt(X^2 + Y^2) / sqrt(2), where X, Y ~ N(0, 1)
        h = torch.sqrt(
            torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device) ** 2 +
            torch.normal(mean=0.0, std=1, size=input_layer.shape, device=device) ** 2
        ) / np.sqrt(2)
        
        # 페이딩 적용: y = h * x + n
        return input_layer * h + noise

    def complex_normalize(self, x, power):
        """
        복소수 신호를 지정된 전력으로 정규화합니다.
        
        Args:
            x (torch.Tensor): 입력 복소수 텐서
            power (float): 목표 전력
        
        Returns:
            tuple: (정규화된 텐서, 원본 전력)
        """
        pwr = torch.mean(x ** 2) * 2  # 복소수 전력 계산
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, chan_param, avg_pwr=False):
        """
        채널 효과를 적용합니다.
        
        입력 텐서를 복소수 형태로 변환하고, 채널 타입에 따라
        노이즈나 페이딩을 적용한 후 다시 실수 형태로 변환합니다.
        
        Args:
            input (torch.Tensor): 입력 특징 텐서 (B, Seq, C)
            chan_param (float): 채널 파라미터 (SNR in dB)
            avg_pwr (bool): 평균 전력 정규화 여부 (사용되지 않음)
        
        Returns:
            torch.Tensor: 채널을 통과한 출력 텐서 (B, Seq, C)
        """
        original_shape = input.shape  # (B, Seq, C)
        B = original_shape[0]

        # (B, Seq, C) -> (B, Seq * C) 변환
        channel_in = input.reshape(B, -1)
        num_features = channel_in.shape[1]

        # 복소수 변환: 실수 부분과 허수 부분으로 분할
        half_len = num_features // 2
        x_real = channel_in[:, :half_len]
        x_imag = channel_in[:, half_len:]
        channel_complex = x_real + 1j * x_imag
        
        # 채널 효과 적용
        if self.chan_type == 'awgn':
            # AWGN 채널: y = x + n
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            channel_output = self.gaussian_noise_layer(channel_complex, std=sigma)
            
        elif self.chan_type == 'rayleigh':
            # Rayleigh 페이딩 채널: y = h * x + n
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            channel_output = self.rayleigh_noise_layer(channel_complex, std=sigma)
        
        else:  # 'noiseless'
            # 오류 없는 채널: y = x
            channel_output = channel_complex

        # 실수 형태로 재변환
        channel_output = torch.cat([
            torch.real(channel_output),
            torch.imag(channel_output)
        ], dim=1)
        channel_output = channel_output.reshape(original_shape)
        
        return channel_output
