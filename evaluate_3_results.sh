#!/bin/bash

# Evaluate 3 specific result directories and create CBR comparison plot
# Usage: ./evaluate_3_results.sh

echo "=========================================="
echo "3개 결과 CBR 비교 평가"
echo "=========================================="
echo ""

BASE_DIR="/data4/hongsik/E2E_IST/results/DIV2K_SNR-10to10_rand_mask_2"

# Result directories (full paths to result directories, not models directories)
RESULT_DIRS=(
    "${BASE_DIR}/20260101_221704"
    "${BASE_DIR}/20260102_060933"
    "${BASE_DIR}/20260102_154715"
)

echo "평가할 결과 디렉토리:"
for dir in "${RESULT_DIRS[@]}"; do
    echo "  - ${dir}"
done
echo ""

# Run evaluation
python evaluate_all_models.py \
    --result_dirs "${RESULT_DIRS[@]}" \
    --testset Kodak

echo ""
echo "=========================================="
echo "평가 완료!"
echo "결과 파일:"
echo "  - /data4/hongsik/E2E_IST/results/3_results_comparison.json"
echo "  - /data4/hongsik/E2E_IST/results/cbr_comparison_3_results.png"
echo "=========================================="

