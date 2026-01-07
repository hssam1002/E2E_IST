#!/bin/bash

# [설명]
# --training: 학습 모드 켜기
# --channel_type noiseless: 노이즈 없는 깨끗한 채널 사용

python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2