#!/bin/bash

# Compare different depth configurations
# Embed_dims fixed at 128,192,256,320
# Testing: D2,2,2,2 vs D2,2,6,2 vs D2,2,8,2

echo "=========================================="
echo "Depth Configuration Comparison Test"
echo "Embed_dims: 128,192,256,320 (fixed)"
echo "=========================================="
echo ""

# Depth 2,2,2,2
echo "[1/3] Testing Depth Configuration: 2,2,2,2"
echo "  embed_dims: 128,192,256,320"
echo "  depths: 2,2,2,2"
echo "  num_heads: 4,6,8,10"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 128,192,256,320 \
    --depths 2,2,2,2 \
    --num_heads 4,6,8,10
echo ""

# Depth 2,2,6,2 (Default)
echo "[2/3] Testing Depth Configuration: 2,2,6,2 (Default)"
echo "  embed_dims: 128,192,256,320"
echo "  depths: 2,2,6,2"
echo "  num_heads: 4,6,8,10"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 128,192,256,320 \
    --depths 2,2,6,2 \
    --num_heads 4,6,8,10
echo ""

# Depth 2,2,8,2
echo "[3/3] Testing Depth Configuration: 2,2,8,2"
echo "  embed_dims: 128,192,256,320"
echo "  depths: 2,2,8,2"
echo "  num_heads: 4,6,8,10"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 128,192,256,320 \
    --depths 2,2,8,2 \
    --num_heads 4,6,8,10
echo ""

echo "=========================================="
echo "Depth Configuration Comparison Completed!"
echo "=========================================="


