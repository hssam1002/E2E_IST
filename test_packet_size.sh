#!/bin/bash

# Transmitted Chunk (Packet Size) Performance Test Script
# results 폴더에 있는 모델들을 사용하여 다양한 packet_size에 따른 성능을 테스트합니다.
# 기존 validate 함수를 사용하여 각 packet_size마다 테스트를 실행합니다.

# 테스트 설정
TESTSET="Kodak"
CHANNEL_TYPE="noiseless"

# packet_size는 32로 고정 (validate 함수가 자동으로 chunk-by-chunk 성능 측정)
PACKET_SIZE=32

# 각 모델 설정 (progressive_mode, alpha_mode, model_dir)
declare -A MODELS
MODELS[base_off]="off|base|/data4/hongsik/E2E_IST/results/DIV2K_SNR10_base_off/20251218_203023/models"
MODELS[base_mrl]="mrl|base|/data4/hongsik/E2E_IST/results/DIV2K_SNR10_base_mrl/20251220_132302/models"
MODELS[base_rand_mask_1]="rand_mask_1|base|/data4/hongsik/E2E_IST/results/DIV2K_SNR10_base_rand_mask_1/20251219_074305/models"
MODELS[base_rand_mask_2]="rand_mask_2|base|/data4/hongsik/E2E_IST/results/DIV2K_SNR10_base_rand_mask_2/20251219_185756/models"
MODELS[exponential_alm]="alm|exponential|/data4/hongsik/E2E_IST/results/DIV2K_SNR10_exponential_alm/20251217_205623/models"

# 각 모델에 대해 테스트 실행
for MODEL_NAME in "${!MODELS[@]}"; do
    IFS='|' read -r PROGRESSIVE_MODE ALPHA_MODE MODEL_DIR <<< "${MODELS[$MODEL_NAME]}"
    
    echo "=========================================="
    echo "Testing Model: ${MODEL_NAME}"
    echo "Progressive Mode: ${PROGRESSIVE_MODE}"
    echo "Alpha Mode: ${ALPHA_MODE}"
    echo "=========================================="
    
    # packet_size=32로 테스트 (validate 함수가 자동으로 chunk-by-chunk 성능 측정)
    echo "--- Packet Size: ${PACKET_SIZE} (chunk-by-chunk: 32, 64, 96, ...) ---"
    
    python main.py \
        --testset ${TESTSET} \
        --channel_type ${CHANNEL_TYPE} \
        --alpha_mode ${ALPHA_MODE} \
        --progressive_mode ${PROGRESSIVE_MODE} \
        --packet_size ${PACKET_SIZE} \
        --model_dir ${MODEL_DIR}
    
    echo ""
    
    echo "=========================================="
    echo "Completed testing for ${MODEL_NAME}"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "=========================================="

