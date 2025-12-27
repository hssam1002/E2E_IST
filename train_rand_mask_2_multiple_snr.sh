#!/bin/bash

# [설명]
# 여러 SNR 값(0dB, 3dB, 6dB, 9dB)에서 rand_mask_2 모드로 학습
# --training: 학습 모드 켜기
# --channel_type awgn: AWGN 채널 사용
# --learning_mode rand_mask_2: rand_mask_2 전략 사용
# Architecture (기본값 사용):
# Training 설정:
#   --batch_size: 4 (per GPU)
#   --gradient_accumulation_steps: 2 (효과적인 배치 크기 = 4 * 2 = 8)
#   --emb_dim: 128,192,256,320
#   --patch_embed_size: 2 (PatchEmbed용, PatchMerging은 항상 2x downsampling)
#   --num_heads: 4,6,8,10
#   --depths: 2,2,6,2
#   --window_size: 8,8,8,8

SNR_LIST=(0 3 6 9)

for SNR in "${SNR_LIST[@]}"; do
    echo "=========================================="
    echo "Training with SNR: ${SNR} dB"
    echo "=========================================="
    
    python main.py \
        --training \
        --trainset DIV2K \
        --testset Kodak \
        --channel_type awgn \
        --train_snr ${SNR} \
        --learning_mode rand_mask_2 \
        --emb_dim 128,192,256,320 \
        --patch_embed_size 2 \
        --num_heads 4,6,8,10 \
        --depths 2,2,6,2 \
        --window_size 8,8,8,8 \
        --batch_size 4 \
        --gradient_accumulation_steps 2
    echo "Completed training for SNR: ${SNR} dB"
    echo ""
done

echo "All training completed!"

