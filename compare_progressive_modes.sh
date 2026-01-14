#!/bin/bash

# 여러 progressive_mode에 대해 학습을 순차적으로 실행하는 간단한 쉘 스크립트.
# 사용 예:
#   bash compare_progressive_modes.sh "--trainset DIV2K --testset Kodak --packet_size 16 --mask_prob 0.1"

COMMON_ARGS=$1

MODES=("full" "full_m" "full_dual" "full_part" "full_part_m" "mask_only" "hybrid_all")

for MODE in "${MODES[@]}"; do
  echo "====================================================="
  echo " Training with progressive_mode = ${MODE}"
  echo "====================================================="
  python main.py --training --progressive_mode "${MODE}" ${COMMON_ARGS}
done

