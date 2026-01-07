#!/bin/bash

# Test all architectures sequentially
# This script runs all architecture variations for comparison

echo "=========================================="
echo "Starting Architecture Comparison Tests"
echo "=========================================="
echo ""

# Default Architecture
echo "[1/9] Testing Default Architecture"
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

# Small Architecture 1
echo "[2/9] Testing Small Architecture 1"
echo "  embed_dims: 64,96,128,160"
echo "  depths: 2,2,4,2"
echo "  num_heads: 2,3,4,5"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 64,96,128,160 \
    --depths 2,2,4,2 \
    --num_heads 2,3,4,5
echo ""

# Small Architecture 2
echo "[3/9] Testing Small Architecture 2"
echo "  embed_dims: 96,128,160,192"
echo "  depths: 2,2,4,2"
echo "  num_heads: 3,4,5,6"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 96,128,160,192 \
    --depths 2,2,4,2 \
    --num_heads 3,4,5,6
echo ""

# Small Architecture 3
echo "[4/9] Testing Small Architecture 3"
echo "  embed_dims: 80,120,160,200"
echo "  depths: 2,2,4,2"
echo "  num_heads: 2,4,5,5"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 80,120,160,200 \
    --depths 2,2,4,2 \
    --num_heads 2,4,5,5
echo ""

# Small Architecture 4
echo "[5/9] Testing Small Architecture 4"
echo "  embed_dims: 112,144,176,208"
echo "  depths: 2,2,4,2"
echo "  num_heads: 4,4,4,4"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 112,144,176,208 \
    --depths 2,2,4,2 \
    --num_heads 4,4,4,4
echo ""

# Depth Variation: depth=2
echo "[6/9] Testing Depth Variation: 3rd layer depth=2"
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

# Depth Variation: depth=4
echo "[7/9] Testing Depth Variation: 3rd layer depth=4"
echo "  embed_dims: 128,192,256,320"
echo "  depths: 2,2,4,2"
echo "  num_heads: 4,6,8,10"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 128,192,256,320 \
    --depths 2,2,4,2 \
    --num_heads 4,6,8,10
echo ""

# Depth Variation: depth=8
echo "[8/9] Testing Depth Variation: 3rd layer depth=8"
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

# Depth Variation: depth=10
echo "[9/9] Testing Depth Variation: 3rd layer depth=10"
echo "  embed_dims: 128,192,256,320"
echo "  depths: 2,2,10,2"
echo "  num_heads: 4,6,8,10"
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 128,192,256,320 \
    --depths 2,2,10,2 \
    --num_heads 4,6,8,10
echo ""

echo "=========================================="
echo "All Architecture Tests Completed!"
echo "=========================================="

