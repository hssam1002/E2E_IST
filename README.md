# E2E-IST: End-to-End Image Semantic Transmission

End-to-End Progressive Image Transmission using Swin Transformer-based Joint Source-Channel Coding (JSCC) with Augmented Lagrangian Method (ALM).

## Introduction

This project implements a progressive image transmission system that sends image features in chunks, allowing incremental reconstruction at the receiver. The system uses:

- **Swin Transformer** as the backbone for JSCC encoder/decoder
- **Augmented Lagrangian Method (ALM)** for progressive transmission control
- **Adaptive fine-tuning** with Scale & Shift Features (SSF)
- **Multiple progressive strategies** for different use cases

## Project Structure

```
E2E_IST/
├── main.py              # Main entry point
├── config.py            # Configuration and argument parsing
├── model_utils.py       # Model utilities (loading, setup, etc.)
├── train.py             # Training functions
├── test.py              # Test and validation functions
├── utils_plot.py        # Plotting and result saving utilities
├── utils.py             # General utilities
├── net/
│   ├── network.py       # E2E_SwinJSCC network
│   ├── encoder.py       # Swin Transformer encoder
│   ├── decoder.py       # Swin Transformer decoder
│   ├── channel.py       # Channel model (AWGN, Rayleigh, etc.)
│   └── modules.py       # Swin Transformer modules
├── loss/
│   └── distortion.py    # Loss functions (MS-SSIM, etc.)
└── data/
    └── datasets.py      # Data loaders
```

## Progressive Modes

The system supports multiple progressive transmission strategies:

- **`off`**: Non-progressive mode (all features transmitted at once)
- **`alm`**: ALM-based progressive transmission with constraint control
- **`adaptive-alm`**: SSF + ALM for adaptive fine-tuning
- **`adaptive-mrl`**: SSF + Mean Rate-Distortion Loss
- **`mrl`**: Mean Rate-Distortion Loss (sum of all step losses)
- **`rand_mask_1`**: Random masking (single random chunk)
- **`rand_mask_2`**: Random masking (partial + full)

## Installation

Requirements:
- Python 3.8+
- PyTorch 1.9+ (recommended <= 1.12 for consistency)
- Other dependencies: numpy, matplotlib, pandas

## Usage

### Training

Train a model with specified progressive mode and hyperparameters:

```bash
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --train_snr 10 \
    --progressive_mode alm \
    --alpha_mode exponential \
    --packet_size 32 \
    --lr 1e-4
```

**Key arguments:**
- `--training`: Enable training mode
- `--trainset`: Training dataset (default: DIV2K)
- `--testset`: Test dataset (Kodak, CLIC2021, DIV2K)
- `--channel_type`: Channel model (awgn, rayleigh, noiseless)
- `--train_snr`: Training SNR in dB (default: 10)
- `--progressive_mode`: Progressive strategy (see above)
- `--alpha_mode`: Alpha weight sequence mode (linear, inverse, square, exponential, uniform)
- `--packet_size`: Number of features per transmission step (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--pretrained`: Path to pretrained model for fine-tuning
- `--patience`: Early stopping patience in epochs (default: 100)

**Adaptive mode options:**
- `--ssf_target`: SSF activation location (enc, dec, both) for adaptive-* modes

### Testing

Test a trained model:

```bash
python main.py \
    --testset Kodak \
    --channel_type noiseless \
    --progressive_mode alm \
    --packet_size 32 \
    --model_dir /path/to/models
```

**Test mode arguments:**
- `--model_dir`: Directory containing trained models (auto-discovered by progressive_mode)
- Models are automatically found using naming: `{progressive_mode}_{alpha_mode}.pth` or `{progressive_mode}.pth`

**Test features:**
- Chunk-by-chunk performance evaluation (all modes)
- For `off` mode: Sequential vs. variance-sorted transmission comparison
- Performance metrics: PSNR, MS-SSIM

### Examples

**1. Train ALM-based model:**
```bash
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --train_snr 10 \
    --progressive_mode alm \
    --alpha_mode exponential \
    --packet_size 32
```

**2. Train adaptive-alm model:**
```bash
python main.py \
    --training \
    --trainset DIV2K \
    --testset Kodak \
    --channel_type awgn \
    --train_snr 10 \
    --progressive_mode adaptive-alm \
    --alpha_mode exponential \
    --packet_size 32 \
    --ssf_target both
```

**3. Test trained model:**
```bash
python main.py \
    --testset Kodak \
    --channel_type noiseless \
    --progressive_mode alm \
    --alpha_mode exponential \
    --packet_size 32 \
    --model_dir ./results/.../models
```

## Model Architecture

- **Encoder**: Swin Transformer with 4 stages [128, 192, 256, 320] channels
- **Decoder**: Symmetric Swin Transformer decoder
- **Channel**: AWGN, Rayleigh fading, or noiseless
- **Progressive transmission**: Features transmitted in chunks of `packet_size`

## Training Details

- **Loss functions**: 
  - ALM modes: Final MSE + Lagrangian penalty
  - MRL modes: Sum of all step MSEs
  - Random mask modes: Partial step MSEs
  
- **Optimization**: 
  - Optimizer: AdamW
  - Learning rate scheduler: ReduceLROnPlateau
  - Early stopping based on validation PSNR

- **ALM parameters**:
  - Rho (penalty parameter) adaptively updated
  - Lambda (Lagrange multiplier) updated per epoch

## Validation

During validation, the system:
- Evaluates chunk-by-chunk performance regardless of progressive mode
- Logs PSNR and MS-SSIM for each chunk
- Saves reconstructed images periodically

## Results

Results are saved in:
- Training: `results/{trainset}_SNR{snr}_{alpha_mode}_{progressive_mode}/{timestamp}/`
- Testing: `results/test_{testset}_{progressive_mode}_packet{size}/{timestamp}/`

Each directory contains:
- `models/`: Saved model checkpoints
- `samples/`: Reconstructed images
- `Log_*.log`: Training/testing logs

## Citation

If you use this code, please cite the original SwinJSCC paper:

```bibtex
@ARTICLE{10589474,
  author={Yang, Ke and Wang, Sixian and Dai, Jincheng and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Adaptation models;Signal to noise ratio;Convolutional neural networks;Wireless communication;Vectors;Image coding;Joint source-channel coding;Swin Transformer;attention mechanism;image communications},
  doi={10.1109/TCCN.2024.3424842}
}
```

## Acknowledgement

- Based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- Dataset links:
  - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  - [Kodak](http://r0k.us/graphics/kodak/)
  - [CLIC2021](http://compression.cc)