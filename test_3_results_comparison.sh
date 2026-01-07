#!/bin/bash

# Compare 3 different experiment results
# Results directories:
# 1. 20260101_221704
# 2. 20260102_060933
# 3. 20260102_154715

echo "=========================================="
echo "3개 결과 비교 테스트"
echo "=========================================="
echo ""

BASE_DIR="/data4/hongsik/E2E_IST/results/DIV2K_SNR-10to10_rand_mask_2"
PROGRESSIVE_MODE="rand_mask_2"
TESTSET="Kodak"
CHANNEL_TYPE="awgn"

# Result directories
RESULT_DIRS=(
    "${BASE_DIR}/20260101_221704"
    "${BASE_DIR}/20260102_060933"
    "${BASE_DIR}/20260102_154715"
)

# Result names for identification
RESULT_NAMES=(
    "Result_1_20260101_221704"
    "Result_2_20260102_060933"
    "Result_3_20260102_154715"
)

echo "테스트 설정:"
echo "  Testset: ${TESTSET}"
echo "  Channel Type: ${CHANNEL_TYPE}"
echo "  Progressive Mode: ${PROGRESSIVE_MODE}"
echo ""

# Test each result
for i in "${!RESULT_DIRS[@]}"; do
    RESULT_DIR="${RESULT_DIRS[$i]}"
    RESULT_NAME="${RESULT_NAMES[$i]}"
    MODEL_DIR="${RESULT_DIR}/models"
    
    echo "=========================================="
    echo "[$((i+1))/3] Testing: ${RESULT_NAME}"
    echo "  Model Directory: ${MODEL_DIR}"
    echo "=========================================="
    
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "  ERROR: Model directory not found: ${MODEL_DIR}"
        echo ""
        continue
    fi
    
    if [ ! -f "${MODEL_DIR}/best_model.pth" ]; then
        echo "  ERROR: best_model.pth not found in ${MODEL_DIR}"
        echo ""
        continue
    fi
    
    python main.py \
        --testset ${TESTSET} \
        --channel_type ${CHANNEL_TYPE} \
        --progressive_mode ${PROGRESSIVE_MODE} \
        --model_dir ${MODEL_DIR}
    
    echo ""
done

echo "=========================================="
echo "3개 결과 비교 테스트 완료!"
echo "=========================================="
echo ""
echo "각 결과의 로그 파일을 확인하여 성능을 비교하세요:"
for i in "${!RESULT_DIRS[@]}"; do
    echo "  ${RESULT_NAMES[$i]}: ${RESULT_DIRS[$i]}/Log_*.log"
done

