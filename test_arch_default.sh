#!/bin/bash

# Default Architecture Test
# embed_dims: 128,192,256,320
# depths: 2,2,6,2
# num_heads: 4,6,8,10

python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --progressive_mode rand_mask_2 \
    --embed_dims 128,192,256,320 \
    --depths 2,2,6,2 \
    --num_heads 4,6,8,10

