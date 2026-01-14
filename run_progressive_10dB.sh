#!/bin/bash

# 10 dB 단일 SNR에서 7가지 progressive_mode를 학습시키는 스크립트
# 사용 예:
#   bash run_progressive_10dB.sh
#
# - train_snr_list 는 "10"으로 고정
# - validation 시에도 평균 SNR이 10 dB 이므로, val_snr = 10 dB 로 동작

COMMON_ARGS="--train_snr_list 10 --testset Kodak --packet_size 16 --mask_prob 0.1"

# 7가지 모드 전체 학습
MODES=("full" "full_m" "full_dual" "full_part" "full_part_m" "mask_only" "hybrid_all")

for MODE in "${MODES[@]}"; do
  echo "====================================================="
  echo " Training with progressive_mode = ${MODE} at 10 dB"
  echo "====================================================="
  python main.py --training --progressive_mode "${MODE}" ${COMMON_ARGS}
done

