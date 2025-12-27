#!/bin/bash

# [설명]
# SNR 10dB에서 두 가지 아키텍처를 비교
# Architecture 1: Default trial (1)
# Architecture 2: Default trial (2)
# --training: 학습 모드 켜기
# --channel_type awgn: AWGN 채널 사용
# --learning_mode rand_mask_2: rand_mask_2 전략 사용
# Training 설정:
#   --batch_size: 4 (per GPU)
#   --gradient_accumulation_steps: 2 (효과적인 배치 크기 = 4 * 2 = 8)

SNR=10

echo "=========================================="
echo "Architecture Comparison at SNR: ${SNR} dB"
echo "=========================================="

# Architecture 1: Default trial (1)
echo ""
echo "------------------------------------------"
echo "Architecture 1: Default trial (1)"
echo "emb_dim: 128,192,256,320"
echo "patch_embed_size: 2"
echo "num_heads: 4,6,8,10"
echo "depths: 2,2,6,2"
echo "window_size: 8,8,8,8"
echo "------------------------------------------"

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

echo "Completed Architecture 1"
echo ""

# Architecture 2: Default trial (2)
echo ""
echo "------------------------------------------"
echo "Architecture 2: Default trial (2)"
echo "emb_dim: 256,256,256,256"
echo "patch_embed_size: 8"
echo "num_heads: 4,4,4,4"
echo "depths: 2,2,4,2"
echo "window_size: 32,16,8,4"
echo "------------------------------------------"

python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --train_snr ${SNR} \
    --learning_mode rand_mask_2 \
    --emb_dim 256,256,256,256 \
    --patch_embed_size 8 \
    --num_heads 4,4,4,4 \
    --depths 2,2,4,2 \
    --window_size 32,16,8,4 \
    --batch_size 4 \
    --gradient_accumulation_steps 2

echo "Completed Architecture 2"
echo ""

echo "All architecture comparison completed!"

