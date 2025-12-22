#!/bin/bash

# Progressive Performance Test Script (Noiseless)
# 여러 progressive_mode와 packet_size에 대해 성능을 테스트합니다.
# 여러 모드를 테스트하려면 이 스크립트를 여러 번 실행하거나 루프를 사용하세요.

# 모델 디렉토리 (모델들을 하나의 폴더에 모아두고 파일명 규칙을 따르세요)
# 파일명 규칙: {progressive_mode}_{alpha_mode}.pth 또는 {progressive_mode}.pth
MODEL_DIR="/data4/hongsik/E2E_IST/test_models"

# 또는 단일 모델 경로 지정
# PRETRAINED_MODEL="/data4/hongsik/E2E_IST/results/.../models/best_model_base.pth"

# 테스트 설정
PROGRESSIVE_MODE="alm"  # off, alm, adaptive-alm, adaptive-mrl, mrl, rand_mask_1, rand_mask_2
PACKET_SIZE=32
ALPHA_MODE="base"

# 여러 모드를 테스트하는 예제 (주석 해제하여 사용)
# for mode in alm mrl rand_mask_1 rand_mask_2; do
#     python main.py \
#         --testset Kodak \
#         --channel_type noiseless \
#         --alpha_mode ${ALPHA_MODE} \
#         --progressive_mode ${mode} \
#         --packet_size ${PACKET_SIZE} \
#         --model_dir ${MODEL_DIR}
# done

# 단일 테스트 실행
python main.py \
    --testset Kodak \
    --channel_type noiseless \
    --alpha_mode ${ALPHA_MODE} \
    --progressive_mode ${PROGRESSIVE_MODE} \
    --packet_size ${PACKET_SIZE} \
    --model_dir ${MODEL_DIR}
    # --pretrained ${PRETRAINED_MODEL} \  # 단일 모델 사용 시 주석 해제
