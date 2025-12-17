#!/bin/bash

# [설명]
# --training: 학습 모드 켜기
# --channel_type noiseless: 노이즈 없는 깨끗한 채널 사용
# --alpha_mode base: 점진적 학습 끄기 (한 번에 통으로 학습)

python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type noiseless \
    --alpha_mode exponential \
    --progressive_mode progressive \
    --packet_size   32\
    --train_snr 10 \
    --save_start 0 \
    --save_end 0