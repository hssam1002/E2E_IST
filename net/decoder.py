"""
Decoder 모듈

Swin Transformer 기반 Decoder를 구현합니다.
특징 벡터를 이미지로 디코딩하는 역할을 합니다.
"""

from net.modules import *
import torch
import torch.nn as nn
from net.encoder import SwinTransformerBlock
import torch.utils.checkpoint as checkpoint

# ============================================================================
# Basic Layer (Decoder용 - Upsample 포함)
# ============================================================================
class BasicLayer(nn.Module):
    """
    Decoder용 기본 레이어 (Stage).
    
    여러 개의 SwinTransformerBlock과 Patch Reverse Merging 레이어로 구성됩니다.
    Encoder와 달리 Upsampling을 수행합니다.
    
    Args:
        dim (int): 입력 채널 수
        out_dim (int): 출력 채널 수
        input_resolution (tuple[int]): 입력 해상도
        depth (int): SwinTransformerBlock의 개수
        num_heads (int): Attention head 수
        window_size (int): Window 크기
        mlp_ratio (float): MLP expansion ratio
        qkv_bias (bool): QKV bias 사용 여부
        qk_scale (float | None): QK scale
        norm_layer: Normalization 레이어
        upsample: Upsampling 레이어 (PatchReverseMerging)
        use_checkpoint (bool): Gradient checkpointing 사용 여부
    """
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # SwinTransformerBlock 생성
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Patch Reverse Merging 레이어 (해상도 증가 및 채널 감소)
        if upsample is not None:
            self.upsample = upsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer
            )
        else:
            self.upsample = None

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): 입력 텐서 (B, L, C)
        
        Returns:
            torch.Tensor: 출력 텐서
        """
        # SwinTransformerBlock 통과
        for blk in self.blocks:
            if self.use_checkpoint:
                # Gradient checkpointing: 메모리 절약
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        # Patch Reverse Merging (해상도 증가)
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def update_resolution(self, H, W):
        """
        해상도 변경 시 내부 블록들의 해상도를 업데이트합니다.
        
        Args:
            H (int): 새로운 높이
            W (int): 새로운 너비
        """
        self.input_resolution = (H, W)
        for blk in self.blocks:
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops

class SwinJSCC_Decoder(nn.Module):
    """
    Swin Transformer 기반 Decoder.
    
    전송된 특징 벡터(C')를 MLP로 변환한 후 이미지로 디코딩합니다.
    
    Args:
        img_size (tuple[int]): 복원 이미지의 최종 해상도 (예: (256, 256))
        embed_dims (list[int]): 각 stage의 채널 차원 리스트 (예: [320, 256, 192, 128])
        depths (list[int]): 각 stage의 Swin Transformer block 깊이 (예: [2, 6, 2, 2])
        num_heads (list[int]): 각 stage의 attention head 수
        window_size (int): Window Multi-head Self Attention의 window 크기. Default: 8
        mlp_ratio (float): MLP feed-forward layer의 expansion ratio. Default: 4.0
        qkv_bias (bool): Query, Key, Value에 learnable bias 추가 여부. Default: True
        qk_scale (float | None): 기본 qk scale 오버라이드. Default: None
        norm_layer: 블록에서 사용할 normalization 레이어. Default: nn.LayerNorm
        patch_norm (bool): Patch merging 후 normalization 추가 여부. Default: True
        model (str): 모델 타입 식별자 (예: 'E2E'). Default: None
        patch_size (int): 초기 embedding의 patch 크기. Default: 2
        in_chans (int): 출력 채널 수 (RGB 이미지의 경우 3). Default: 3
        use_checkpoint (bool): Gradient checkpointing 사용 여부. Default: False
        transmitted_dim (int, optional): 전송 차원 (C'). None이면 decoder input dimension 사용. Default: None
        **kwargs: 추가 인자
    """
    def __init__(self, 
                 img_size,
                 embed_dims,
                 depths,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 model=None,
                 patch_size=2,
                 in_chans=3,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        self.use_checkpoint = use_checkpoint
        
        # Transmitted dimension (C'): if None, use decoder input dimension (first embed_dim)
        self.transmitted_dim = kwargs.get('transmitted_dim', None)
        if self.transmitted_dim is None:
            self.transmitted_dim = embed_dims[0]
        
        # Build reconstruction layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging,
                               use_checkpoint=self.use_checkpoint)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        
        # MLP for decoder input: C' -> C (first decoder dimension)
        if self.transmitted_dim != embed_dims[0]:
            self.head_list = nn.Sequential(nn.Linear(self.transmitted_dim, embed_dims[0]))
        else:
            self.head_list = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): 전송된 특징 벡터 (B, L, C')
        
        Returns:
            torch.Tensor: 복원된 이미지 (B, 3, H, W)
        """
        # MLP projection: C' -> C (first decoder dimension)
        x_recon = self.head_list(x)
        
        # Swin Transformer layers (Upsampling 포함)
        for layer in self.layers:
            x_recon = layer(x_recon)
        
        # (B, L, 3) -> (B, 3, H, W) 변환
        B, L, Ch = x_recon.shape
        x_recon = x_recon.view(B, self.H, self.W, Ch).permute(0, 3, 1, 2)
        
        return x_recon
    
    def update_resolution(self, H, W):
        """
        입력 해상도 변경 시 내부 레이어들의 해상도를 업데이트합니다.
        
        Args:
            H (int): 새로운 높이
            W (int): 새로운 너비
        """
        self.H = H * (2 ** len(self.layers))
        self.W = W * (2 ** len(self.layers))
        self.patches_resolution = (H, W)
        
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(
                H * (2 ** i_layer),
                W * (2 ** i_layer)
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

def create_decoder(**kwargs):
    """
    Decoder 인스턴스를 생성합니다.
    
    Args:
        **kwargs: SwinJSCC_Decoder 생성자에 전달할 인자
    
    Returns:
        SwinJSCC_Decoder: 생성된 decoder 인스턴스
    """
    model = SwinJSCC_Decoder(**kwargs)
    return model

