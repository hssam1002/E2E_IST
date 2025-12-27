#!/bin/bash

# [설명]
# --training: 학습 모드 켜기
# --channel_type awgn: AWGN 채널 사용
# --train_snr 0: SNR 0dB로 학습
# --learning_mode rand_mask_2: rand_mask_2 전략 사용
# --use_checkpoint: Gradient checkpointing 활성화 (메모리 절약)
# Architecture (기본값 사용):
#   --emb_dim: 128,192,256,320
#   --patch_embed_size: 2 (PatchEmbed용, PatchMerging은 항상 2x downsampling)
#   --num_heads: 4,6,8,10
#   --depths: 2,2,6,2
#   --window_size: 8,8,8,8

python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --train_snr 0 \
    --learning_mode rand_mask_2 \
    --use_checkpoint \
    --emb_dim 128,192,256,320 \
    --patch_embed_size 2 \
    --num_heads 4,6,8,10 \
    --depths 2,2,6,2 \
    --window_size 8,8,8,8
