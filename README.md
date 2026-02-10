# Brontes

**Synthesis-first waveform enhancement for neural codec repair and bandwidth extension.**

Brontes is a time-domain audio enhancement model that upsamples and repairs speech degraded by neural codec compression. Unlike conventional Wave U-Net approaches that rely on dense skip connections, Brontes uses a synthesis-first architecture with selective deep skips, forcing the model to actively reconstruct rather than copy degraded input details.

## Pretrained Models

| Model | Dataset | Sample Rate | Parameters | Download |
|-------|---------|-------------|------------|----------|
| Brontes-Neucodec24-48-30M | Internal paired data | 48 kHz | 30M | [HuggingFace](https://huggingface.co/ZDisket/Brontes-General-NeuCodec24-48) / [ModelScope](https://www.modelscope.ai/models/ZDisket/Brontes-General-NeuCodec24-48) |
| Brontes-Neucodec24-48-60M | Internal paired data | 48 kHz | 60M | *Coming soon* |

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** PyTorch ≥ 2.0, torchaudio, pesto-pitch (for pitch loss)

## Quick Start

### Training

```bash
python train_brontes.py \
  --config configs/config_brontes_48khz_demucs.yaml \
  --train_input_dir <path/to/degraded_audio> \
  --train_output_dir <path/to/target_audio>
```

Training pairs files by basename between input and target directories. The default config uses:
- 10,000 pretrain steps (generator-only) before adversarial training
- Multi-scale mel loss + pitch loss
- MPD + multi-band spectral discrimination with hinge loss
- BF16 mixed precision

### Checkpoint Loading & Fine-tuning

Brontes supports three checkpoint loading modes with intelligent priority handling:

#### 1. Fine-tuning from Pretrained Models (`--pretrained`)


The pretrained model is trained on a variety of audio data. **To extract maximum performance, it is recommended to fine-tune the model on your specific dataset**. Load model weights without optimizer state for fine-tuning on new data:

```bash
python train_brontes.py \
  --config configs/config_brontes_48khz_demucs.yaml \
  --train_input_dir <path/to/new_degraded_audio> \
  --train_output_dir <path/to/new_target_audio> \
  --pretrained <path/to/pretrained_checkpoint_dir>
```

**What gets loaded:** Model weights + discriminator weights (if available)  
**What doesn't:** Optimizer states, training step/epoch counters  
**Use case:** Adapting a pretrained model to a new dataset or domain

#### 2. Resuming Training (`--checkpoint_path`)

Explicitly resume from a specific checkpoint directory with full training state:

```bash
python train_brontes.py \
  --config configs/config_brontes_48khz_demucs.yaml \
  --train_input_dir <path/to/degraded_audio> \
  --train_output_dir <path/to/target_audio> \
  --checkpoint_path <path/to/checkpoint_dir>
```

**What gets loaded:** Model + discriminator + optimizers + step/epoch counters  
**Use case:** Continuing interrupted training runs

#### 3. Automatic Checkpoint Resumption

If your checkpoint directory already contains checkpoints, training automatically resumes from the latest one:

```bash
# First run - starts from scratch
python train_brontes.py \
  --config configs/config_brontes_48khz_demucs.yaml \
  --train_input_dir <path/to/degraded_audio> \
  --train_output_dir <path/to/target_audio>

# Subsequent runs - automatically resumes from latest checkpoint
# (same command, no additional flags needed)
python train_brontes.py \
  --config configs/config_brontes_48khz_demucs.yaml \
  --train_input_dir <path/to/degraded_audio> \
  --train_output_dir <path/to/target_audio>
```

**Priority order:** `--checkpoint_path` > `--pretrained` > auto-resume > from scratch

### Inference

```bash
python infer_brontes.py \
  --config configs/config_brontes_48khz_demucs.yaml \
  --checkpoint <path/to/checkpoint.pt> \
  --input <path/to/input.wav> \
  --output <path/to/output.wav>
```

## Project Structure

```
Brontes/
├── brontes.py              # Main model (Brontes → UNet1DRefiner)
├── unet1d.py               # Core U-Net implementation
├── discriminators.py       # Unified MPD/MSD/MBSD discriminator stack
├── audio_loss.py           # Mel, MR-STFT, and pitch losses
├── audio_pair_dataset.py   # Paired audio dataset loader
├── train_brontes.py        # Training script with GAN support
├── infer_brontes.py        # Inference with overlap-add chunking
├── infer_brontes.ipynb     # Notebook for inference/export
├── configs/
│   └── config_brontes_48khz_demucs.yaml
├── tests/                  # Unit tests for U-Net, discriminators, losses
└── requirements.txt
```

## Architecture Overview

Brontes uses a 6-stage encoder-decoder with stride 4 per stage (4096× total compression):

```
Input (24kHz) → Encoder [6 stages] → LSTM Bottleneck → Decoder [6 stages] → Output (48kHz)
                    ↓                                        ↑
                    └──────── Deep skips only [-1, -2] ──────┘
```

**Why selective skips?** Standard U-Net skips assume input details are useful for reconstruction. For codec repair, input details *are* the artifacts we're trying to remove. Restricting skips to deep layers provides semantic guidance without leaking degraded fine structure.

## Acknowledgments

Compute resources provided by Hot Aisle and AI at AMD.

## License

For both repo and model weights, MIT.