# PCA-Guided Latent Manipulation for StyleGAN3

## Project Overview

本项目基于 NVIDIA StyleGAN3 开源代码，探索在 W 空间中引入
PCA（Principal Component Analysis）作为潜空间引导机制，
以实现可解释、可控的语义形变与连续插值。
项目聚焦于潜空间分析、语义控制以及面向交互系统的生成流程设计。

This repository is a derivative research project built on top of StyleGAN3.
It focuses on PCA-guided latent manipulation for semantic control,
temporal morphing, and interactive deployment.
The project is intended strictly for non-commercial research and educational purposes.

## Contributions

This project extends StyleGAN3 with a PCA-guided latent manipulation pipeline.
The main contributions are as follows:

- **W-space PCA Direction Extraction**  
  Implemented a tool to sample latent codes in StyleGAN3’s W space and perform PCA,
  producing reusable semantic directions (mean, eigenvalues, eigenvectors).
  This enables data-driven discovery of dominant variation axes without manual annotation.

- **Layer-aware PCA Editing and Sweep Visualization**  
  Developed a flexible editor to apply PCA directions to selected synthesis layers,
  supporting strength sweeps, layer masking, and grid-based visual diagnostics
  for analyzing semantic locality across the network.

- **Multi-Knob PCA Morphing and Animation**  
  Designed a multi-knob latent morphing system where each knob corresponds to
  a PCA component combined with a specific layer range.
  The system supports smooth interpolation, frame-wise parameter logging,
  and optional video encoding for reproducible semantic transitions.

- **Persistent GPU Morph Engine for Interactive Systems**  
  Implemented a lightweight, file-based rendering engine that keeps StyleGAN3
  and PCA directions resident on GPU, enabling low-latency single-frame generation.
  The system is designed for integration with real-time environments
  such as TouchDesigner or interactive installations.

Together, these components form a modular PCA-guided control framework for StyleGAN3,
bridging latent space analysis, semantic editing, temporal morphing,
and interactive deployment.

All contributions focus on latent manipulation and tooling.
The underlying GAN architecture and training pipeline are inherited from StyleGAN3.

## Usage / Quickstart

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended)
- StyleGAN3 (official implementation)

This project assumes that you already have a trained StyleGAN3 network

### Step A: Compute PCA Directions in W Space

This step samples latent codes in W space and computes PCA directions.
 The output `.npz` file will be reused by all subsequent steps.

```
python tools/pca_w_directions.py \
  --network=network-snapshot-002482.pkl \
  --out=directions/pca_w.npz \
  --n=20000 \
  --k=50 \
  --seed=0 \
  --trunc=1.0
```

Key parameters:

- `--n`: number of W samples (larger = more stable PCA)
- `--k`: number of PCA components to retain
- Output `.npz` contains `mean`, `evals`, `sigma`, and `dirs`

### Step B: PCA-Based Editing and Sweep Visualization

#### B1. Single-Component Editing

Apply a single PCA component to selected synthesis layers and generate
 a set of images with different strengths.

```
python tools/apply_pca_w.py \
  --network=network-snapshot-006646.pkl \
  --pca=directions/pca_w.npz \
  --seed=0 \
  --comp=3 \
  --layers="0-6" \
  --strengths="-5,-3,-1,0,1,3,5" \
  --outdir=out/edit
```

- `--comp`: PCA component index (0 = strongest)
- `--layers`: synthesis layer indices (ws index)
- `--strengths`: list of editing magnitudes

#### B2. Sweep Mode (Analysis / Diagnostics)

Generate grid visualizations to analyze how a PCA component affects
 different layers across multiple strengths.

```
python tools/apply_pca_w.py \
  --network=network-snapshot-006646.pkl \
  --pca=directions/pca_w.npz \
  --seed=0 \
  --sweep \
  --comp-range="0-5" \
  --layer-range="0-15" \
  --strengths="-5,-3,-1,0,1,3,5" \
  --outdir=out/sweep
```

This mode is useful for studying semantic locality and layer sensitivity.

### Step C: Multi-Knob Latent Morphing and Animation

This script defines multiple semantic "knobs",
 where each knob corresponds to a PCA component and a layer range.
 It generates a smooth morphing sequence between two latent states.

```
python tools/pca_morph_4knobs.py \
  --network=network-snapshot-006646.pkl \
  --pca=directions/pca_w.npz \
  --seed-a=0 --seed-b=20 \
  --knob-layers="0|1-6|7-9|10-13|14|15" \
  --knob-comps="1" \
  --a="0,2,4,2,1,0" \
  --b="0,-2,-4,2,-1,10" \
  --steps=120 --fps=30 --smooth \
  --outdir=out/morph \
  --mp4=demo.mp4
```

Notes:

- The number of knobs is defined by the number of `|`-separated layer groups
- `--knob-comps` can be a single integer (broadcast) or one per knob
- `--a` and `--b` must provide one value per knob

The script automatically saves:

- PNG frame sequence
- `morph_config.json`
- `params.jsonl` (one record per frame for reproducibility)


### Step D: Interactive GPU Render Server (Optional)

This mode keeps StyleGAN3 and PCA directions resident on GPU
 and renders a single frame on demand via a file-based IPC interface.

#### Start the server

```
python tools/pca_morph_server_b1.py \
  --network=network-snapshot-006646.pkl \
  --pca=directions/pca_w.npz \
  --knob-layers="0|1-6|7-9|10-13|14|15" \
  --knob-comps="1" \
  --ipc-dir=ipc
```

#### Example `ipc/request.json`

```
{
  "seed": 0,
  "trunc": 1.0,
  "noise_mode": "const",
  "knobs": [0, 2, 4, 2, 1, 0],
  "frame_id": 1
}
```

The server outputs:

- `ipc/latest.png`
- `ipc/latest_meta.json`

This mode is designed for integration with real-time tools
 such as TouchDesigner or custom interactive systems.

## Acknowledgements

This project is based on the official implementation of:

StyleGAN3  
Copyright (c) 2021, NVIDIA Corporation  
https://github.com/NVlabs/stylegan3

All original StyleGAN3 code is subject to the
NVIDIA Source Code License (Non-Commercial).

## License

This repository contains code derived from NVIDIA StyleGAN3,
which is licensed under the NVIDIA Source Code License (Non-Commercial).

All modifications and additional code in this repository are released
for non-commercial research and educational use only.

Please refer to the original StyleGAN3 license for details.

## Dependencies

This project depends on StyleGAN3:
https://github.com/NVlabs/stylegan3