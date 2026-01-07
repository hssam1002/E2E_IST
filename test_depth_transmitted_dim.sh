#!/bin/bash

# Test different depth and transmitted_dim (C') configurations
# Base architecture: embed_dims=128,192,256,320, num_heads=4,6,8,10
# Depths: 2,2,2,2 / 2,2,6,2 / 2,2,8,2
# Transmitted_dim (C'): 64, 128, 192, 256, 320

echo "=========================================="
echo "Depth and Transmitted Dimension Test"
echo "Base: embed_dims=128,192,256,320, num_heads=4,6,8,10"
echo "Depths: 2,2,2,2 / 2,2,6,2 / 2,2,8,2"
echo "Transmitted_dim (C'): 64, 128, 192, 256, 320"
echo "=========================================="
echo ""

# Depth configurations
DEPTHS=(
    "2,2,2,2"
    "2,2,6,2"
    "2,2,8,2"
)

# Transmitted dimension configurations
TRANSMITTED_DIMS=(64 128 192 256 320)

# Base configuration
EMBED_DIMS="128,192,256,320"
NUM_HEADS="4,6,8,10"

# Counters
total_tests=$((${#DEPTHS[@]} * ${#TRANSMITTED_DIMS[@]}))
test_num=0

# Test each combination
for depth in "${DEPTHS[@]}"; do
    for transmitted_dim in "${TRANSMITTED_DIMS[@]}"; do
        test_num=$((test_num + 1))
        
        echo "=========================================="
        echo "[$test_num/$total_tests] Testing Configuration"
        echo "  Depth: $depth"
        echo "  Transmitted_dim (C'): $transmitted_dim"
        echo "  Embed_dims: $EMBED_DIMS"
        echo "  Num_heads: $NUM_HEADS"
        echo "=========================================="
        
        python main.py \
            --training \
            --trainset DIV2K \
            --testset Kodak \
            --channel_type awgn \
            --progressive_mode rand_mask_2 \
            --embed_dims $EMBED_DIMS \
            --depths $depth \
            --num_heads $NUM_HEADS \
            --transmitted_dim $transmitted_dim
        
        echo ""
        echo "Configuration [$test_num/$total_tests] completed!"
        echo ""
    done
done

echo "=========================================="
echo "All Depth and Transmitted Dimension Tests Completed!"
echo "Total: $total_tests configurations tested"
echo "=========================================="

