# World Model-driven Process Industry Operations: An Offline Reinforcement Learning Solution

This repository provides an implementation of a three-stage model-based offline reinforcement learning framework for industrial process optimization, inspired by recent advances in diffusion-based world models and hybrid data-driven policy learning.

The overall pipeline adheres to a structured paradigm: **Training World Models→ Constructing a Hybrid Experience Space → Offline RL Optimization paradigm**. This approach is particularly suited to process industry applications, where online exploration is often prohibitively expensive or practically infeasible.

---

## Framework Overview

<p align="center">
  <img src="fig/framework.png" width="900"/>
</p>
<p align="center">
  <em>Figure 1: Three-stage framework: (1) diffusion-based world model learning, (2) hybrid replay buffer generation (real + model-generated), and (3) offline RL policy optimization using the hybrid buffer.</em>
</p>

---

## Key Features

- **Stage 1 (World Model Learning)**  
  Trains a diffusion-based dynamics model for next-state prediction and a reward/quality predictor using offline process trajectories.

- **Stage 2 (Hybrid Buffer Construction)**  
  Uses the trained world model to generate imagined transitions and constructs a hybrid replay buffer mixing **real** and **model-generated** data.  
  The hybrid buffer is saved in **HDF5 (`.h5`)** format for efficient downstream training.

- **Stage 3 (Offline Reinforcement Learning)**  
  Trains an offline actor–critic agent (TD3 / TD3+BC style) using the saved hybrid replay buffer, without interacting with the real system.

---

## Data Availability (Confidentiality Notice)

 **Important:** Due to industrial data confidentiality and privacy constraints, **we are not able to publicly release the complete dataset** used in our experiments.

- This repository includes only **a small subset / partial example dataset** to demonstrate the required data format and to allow end-to-end execution.
- The example data is **not sufficient** to reproduce the full experimental results reported in the paper.
- To reproduce full results, please replace the example dataset with your own proprietary or public dataset following the same structure.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---
## Quick Start

All stages are executed via main.py by setting --stage.

### Stage 1: Train the World Model (Multi-step & Autoregressive Training)

Stage 1 trains the diffusion-based world model and the reward/quality predictor.
This implementation supports **multi-step unrolled prediction** and optional
**autoregressive (self-feeding) training**.

- **Multi-step prediction:** controlled by `--k_unroll` (K-step unroll length).  
- **Autoregressive training (optional):** enabled by `--enable_self_feeding`.

#### (A) Multi-step prediction (K-step unroll)
```bash
python main.py --stage 1 --k_unroll 5
```
#### (B) Autoregressive training (self-feeding) on top of K-step unroll
```bash
python main.py --stage 1 --k_unroll 5 --enable_self_feeding
```

**Outputs:**

- Checkpoints: ./ckpt/<folder_name>/eps_model_*.pth, pred_reward_model_*.pth
- Training logs: ./results/excel/<folder_name>/

### Stage 2: Build Hybrid Replay Buffer (.h5)
```bash
python main.py --stage 2
```
**Outputs:**
- Hybrid buffer: ./results/h5/<folder_name>/hybrid_buffer.h5

**The saved .h5 contains:**

- trajectorys: shape [N, L, state_dim]
- actions: shape [N, action_dim]
- rewards: shape [N, 1]
- next_states: shape [N, state_dim]

### Stage 3: Offline RL Training
```bash
python main.py --stage 3
```
**By default, Stage 3 reads:**
- ./results/h5/<folder_name>/hybrid_buffer.h5

**You can also specify a custom path**
```bash
python main.py --stage 3 --h5_path ./results/h5/<folder_name>/hybrid_buffer.h5

```

**Outputs:**

- RL checkpoints: ./ckpt/rl/<folder_name>/

**Citing Our Work:**

[1] Y Yin, R. Chiong, C. Deng, L. Wang, D. Niyato and W. Wang, “World Model-driven Process Industry Operations: An Offline Reinforcement Learning Solution based on Conditional Diffusion,”  to appear in Computers in Industry, 2026.


