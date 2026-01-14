#!/bin/bash

# chunk1_full 모드를 학습시키는 스크립트
# 사용 예:
#   bash train_chunk1_full.sh "--trainset DIV2K --testset Kodak --packet_size 16 --mask_prob 0.1"

COMMON_ARGS=$1

echo "====================================================="
echo " Training with progressive_mode = chunk1_full"
echo "====================================================="
python main.py --training --progressive_mode "chunk1_full" ${COMMON_ARGS}
