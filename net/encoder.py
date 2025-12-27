"""
Encoder 모듈

Swin Transformer 기반 Encoder를 구현합니다.
이미지를 특징 벡터로 인코딩하는 역할을 합니다.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from net.modules import *

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    
    Window-based Multi-head Self-Attention (W-MSA)와 
    Shifted Window-based Multi-head Self-Attention (SW-MSA)를 포함합니다.
    
    Args:
        dim (int): 입력 채널 수
        input_resolution (tuple[int]): 입력 해상도 (H, W)
        num_heads (int): Attention head 수
        window_size (int): Window 크기
        shift_size (int): SW-MSA를 위한 shift 크기
        mlp_ratio (float): MLP hidden dimension과 embedding dimension의 비율
        qkv_bias (bool, optional): Query, Key, Value에 learnable bias 추가 여부. Default: True
        qk_scale (float | None, optional): 기본 qk scale (head_dim ** -0.5) 오버라이드
        act_layer (nn.Module, optional): Activation 함수. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization 레이어. Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Window size 조정 (input resolution보다 크면 shift 안 함)
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # --- 1. Attention Part ---
        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        # --- 2. MLP Part ---
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        # --- 3. Attention Mask Setup (기존과 동일) ---
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            raise AssertionError(
                f"input feature has wrong size: expected L={H*W} (H={H}, W={W}), "
                f"but got L={L}. Actual feature shape: {x.shape}"
            )
        assert L == H * W, f"input feature has wrong size: expected L={H*W} (H={H}, W={W}), but got L={L}"

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition Windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, 
                                 add_token=False, 
                                 mask=self.attn_mask)

        # Merge Windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, L, C)

        # FFN (Add Residual)
        x = shortcut + x

        # 2. Norm -> MLP
        shortcut = x
        x = self.norm2(x)
        
        x = self.mlp(x)
        
        x = shortcut + x

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    
    def update_mask(self):
        # Resolution이 바뀔 때 마스크 재계산
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass

# --------------------------------------------------------
# 2. Basic Layer (Stage)
# --------------------------------------------------------
class BasicLayer(nn.Module):
    """
    Swin Transformer의 기본 레이어 (Stage).
    
    여러 개의 SwinTransformerBlock과 Patch Merging 레이어로 구성됩니다.
    
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
        downsample: Downsampling 레이어 (PatchMerging)
        use_checkpoint (bool): Gradient checkpointing 사용 여부 (메모리 절약)
    """
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # 참조 코드 (HJSCC)와 동일하게:
        # self.input_resolution은 blocks가 처리할 해상도 (downsample 후)
        # input_resolution 파라미터는 BasicLayer에 입력되는 해상도 (downsample 전)
        if downsample is not None:
            self.input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        else:
            self.input_resolution = input_resolution

        # SwinTransformerBlock 생성 (짝수 인덱스: W-MSA, 홀수 인덱스: SW-MSA)
        # blocks는 downsample 후 해상도 (self.input_resolution)를 처리
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.input_resolution,
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

        # Patch Merging 레이어 (해상도 감소 및 채널 증가)
        # downsample.input_resolution은 BasicLayer에 입력되는 해상도 (downsample 전)
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,  # downsample 전 해상도
                dim=dim, 
                out_dim=out_dim, 
                norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): 입력 텐서 (B, L, C)
        
        Returns:
            torch.Tensor: 출력 텐서
        """
        # Patch Merging (해상도 감소)
        if self.downsample is not None:
            x = self.downsample(x)
        
        # SwinTransformerBlock 통과
        for blk in self.blocks:
            if self.use_checkpoint:
                # Gradient checkpointing: 메모리 절약을 위해 중간 활성화를 저장하지 않음
                # 역전파 시 다시 계산하여 메모리 사용량을 줄임
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        """
        해상도 변경 시 내부 블록들의 해상도를 업데이트합니다.
        
        Args:
            H (int): 새로운 높이 (blocks가 처리할 해상도, downsample 후)
            W (int): 새로운 너비 (blocks가 처리할 해상도, downsample 후)
        
        참조: HJSCC의 BasicLayerEnc.update_resolution
        """
        # self.input_resolution은 blocks가 처리할 해상도 (downsample 후)
        self.input_resolution = (H, W)
        
        # blocks는 downsample 후 해상도 (H, W)를 처리
        for blk in self.blocks:
            blk.input_resolution = (H, W)
            blk.update_mask()
        
        # downsample이 있으면 downsample.input_resolution은 downsample 전 해상도 (H*2, W*2)
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)

class SwinJSCC_Encoder(nn.Module):
    """
    Swin Transformer 기반 Encoder.
    
    이미지를 특징 벡터로 인코딩합니다.
    
    Args:
        img_size (tuple[int]): 입력 이미지 해상도 (예: (256, 256))
        embed_dims (list[int]): 각 stage의 채널 차원 리스트 (예: [128, 192, 256, 320])
        patch_embed_size (int): PatchEmbed의 patch 크기. Default: 2. PatchMerging은 항상 2배 downsampling.
        depths (list[int]): 각 stage의 Swin Transformer block 깊이 (예: [2, 2, 6, 2])
        num_heads (list[int]): 각 stage의 attention head 수
        window_size (int): Window Multi-head Self Attention의 window 크기. Default: 8
        mlp_ratio (float): MLP feed-forward layer의 expansion ratio. Default: 4.0
        qkv_bias (bool): Query, Key, Value에 learnable bias 추가 여부. Default: True
        qk_scale (float | None): 기본 qk scale 오버라이드. Default: None
        norm_layer: 블록에서 사용할 normalization 레이어. Default: nn.LayerNorm
        patch_norm (bool): Patch merging 후 normalization 추가 여부. Default: True
        model (str): 모델 타입 식별자 (예: 'E2E'). Default: None
        in_chans (int): 입력 채널 수 (RGB 이미지의 경우 3). Default: 3
        use_checkpoint (bool): Gradient checkpointing 사용 여부. Default: False
        **kwargs: 추가 인자
    """
    def __init__(self,
                 img_size,
                 embed_dims,
                 depths,
                 num_heads,
                 patch_embed_size = 2,  # PatchEmbed의 patch 크기 (기본값 2)
                 window_size = 8,  # Can be int or list[int]
                 mlp_ratio = 4.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 model=None,
                 in_chans=3,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_embed_size = patch_embed_size  # PatchEmbed에만 사용
        self.input_resolution = img_size  # 초기 입력 해상도 저장 (tuple로 유지)
        
        # Stage 0: PatchEmbed 후 해상도 = img_size // patch_embed_size
        # 예: patch_embed_size=4, img_size=256 -> 256 // 4 = 64
        #     256*256*3 -> 64*64*embed_dims[0]
        patches_resolution_after_embed = (img_size[0] // patch_embed_size, img_size[1] // patch_embed_size)
        self.patches_resolution = [patches_resolution_after_embed]
        
        # 이후 stages는 PatchMerging으로 2배씩 downsampling (항상 2배, patch_size는 고정)
        for i in range(1, self.num_layers):
            prev_h, prev_w = self.patches_resolution[i-1]
            self.patches_resolution.append((prev_h // 2, prev_w // 2))

        
        # 최종 feature map 해상도
        self.H = self.patches_resolution[-1][0]
        self.W = self.patches_resolution[-1][1]
        
        # PatchEmbed 초기화 (patch_embed_size 사용)
        # 예: patch_embed_size = 4 -> 256*256*3 -> 64*64*embed_dims[0]
        self.patch_embed = PatchEmbed(img_size, patch_embed_size, in_chans, embed_dims[0])
        self.use_checkpoint = use_checkpoint
        
        # Handle window_size: can be int (single value) or list (per stage)
        if isinstance(window_size, (list, tuple)):
            if len(window_size) != self.num_layers:
                raise ValueError(f"window_size list length ({len(window_size)}) must match num_layers ({self.num_layers})")
            window_sizes = window_size
        else:
            window_sizes = [window_size] * self.num_layers
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # BasicLayer의 input_resolution은 BasicLayer에 입력되는 해상도 (downsample 전)
            # Stage 0: PatchEmbed 후 해상도 = patches_resolution[0]
            # Stage 1+: 이전 stage의 output 해상도 = patches_resolution[i_layer-1]
            if i_layer == 0:
                # Stage 0: PatchEmbed 후 해상도
                layer_input_h, layer_input_w = self.patches_resolution[0]
            else:
                # Stage 1+: 이전 stage의 output 해상도 (downsample 전)
                layer_input_h, layer_input_w = self.patches_resolution[i_layer - 1]
            
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(layer_input_h, layer_input_w),
                               depth = depths[i_layer],
                               num_heads = num_heads[i_layer],
                               window_size = window_sizes[i_layer],
                               mlp_ratio = self.mlp_ratio,
                               qkv_bias = qkv_bias, qk_scale=qk_scale,
                               norm_layer = norm_layer,
                               downsample = PatchMerging if i_layer != 0 else None,
                               use_checkpoint = self.use_checkpoint)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
            
        self.norm = norm_layer(embed_dims[-1])       

        self.head_list = nn.Identity()     
        self.apply(self._init_weights)

    def forward(self, x, model=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): 입력 이미지 (B, C, H, W)
            model (str, optional): 모델 타입 (사용되지 않음)
        
        Returns:
            torch.Tensor: 인코딩된 특징 벡터 (B, L, C)
        """
        B, C, H, W = x.size()
        
        # 입력 해상도가 변경되었으면 update_resolution 호출
        if not hasattr(self, 'input_resolution') or H != self.input_resolution[0] or W != self.input_resolution[1]:
            self.update_resolution(H, W)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Swin Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        x = self.head_list(x)
        
        return x

    def _init_weights(self, m):
        """
        가중치 초기화 함수.
        
        Args:
            m (nn.Module): 초기화할 모듈
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Weight decay를 적용하지 않을 파라미터 이름 반환"""
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Weight decay를 적용하지 않을 파라미터 키워드 반환"""
        return {'relative_position_bias_table'}

    def flops(self):
        """
        모델의 FLOPs (Floating Point Operations) 계산.
        
        Returns:
            int: 총 FLOPs 수
        """
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        """
        입력 해상도 변경 시 내부 레이어들의 해상도를 업데이트합니다.
        
        Args:
            H (int): 새로운 높이 (원본 이미지 해상도)
            W (int): 새로운 너비 (원본 이미지 해상도)
        """
        self.input_resolution = (H, W)
        
        # PatchEmbed 해상도 업데이트 (patch_embed_size 사용)
        patch_embed_size = self.patch_embed_size
        patches_resolution_after_embed = (H // patch_embed_size, W // patch_embed_size)

        # 각 stage별 patches_resolution 업데이트
        # Stage 0: PatchEmbed 후 해상도
        self.patches_resolution = [patches_resolution_after_embed]
        # 이후 stages는 PatchMerging으로 2배씩 downsampling (항상 2배)
        for i in range(1, self.num_layers):
            prev_h, prev_w = self.patches_resolution[i-1]
            self.patches_resolution.append((prev_h // 2, prev_w // 2))
        
        # 최종 feature map 해상도 업데이트
        self.H = self.patches_resolution[-1][0]
        self.W = self.patches_resolution[-1][1]
        
        # 각 layer의 해상도 업데이트
        for i_layer, layer in enumerate(self.layers):
            # BasicLayer.update_resolution은 blocks가 처리할 해상도 (downsample 후)를 받음
            # 즉, patches_resolution[i_layer]를 전달
            blocks_h, blocks_w = self.patches_resolution[i_layer]
            layer.update_resolution(blocks_h, blocks_w)

def create_encoder(**kwargs):
    """
    Encoder 인스턴스를 생성합니다.
    
    Args:
        **kwargs: SwinJSCC_Encoder 생성자에 전달할 인자
    
    Returns:
        SwinJSCC_Encoder: 생성된 encoder 인스턴스
    """
    model = SwinJSCC_Encoder(**kwargs)
    return model

def build_model(config):
    input_image = torch.ones([1, 256, 256]).to(config.device)
    model = create_encoder(**config.encoder_kwargs)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
