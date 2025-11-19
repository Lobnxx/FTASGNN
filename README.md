# FTA-SGNN: Encrypted Traffic Detection with Timestamp Encoding and Inductive Graph Learning

Official implementation of **"FTA-SGNN: Encrypted Traffic Detection with Timestamp Encoding and Inductive Graph Learning"**



## 🎯 Overview

FTA-SGNN is a security-oriented encrypted traffic detection framework that addresses three critical limitations in existing approaches:

1. **Overlooked malicious traffic family attributes**
2. **Underutilized payload-header semantic differences**
3. **Limited generalization to unknown threats**

## ✨ Key Features

### 🔧 Traffic Preprocessing Pipeline

- **Traffic Router**: Bidirectional session extraction from raw pcap files
- **Multi-Process Feature Extraction**: Parallel processing with frame-level feature alignment
- **Frame-Level Representation**: First 64 bytes per frame with timestamp preservation

### 🧠 Model Components

- **Timestamp-aware Semantic Encoder (TSE)**: Transformer-based encoder with relative timestamp encoding
- **Masked Graph Aggregation Module (MGAM)**: GraphSAGE with inductive learning capability
- **KNN-based Edge Enhancement**: Graph augmentation for capturing family-level similarities

### 🛡️ Adversarial Robustness

- Resilient to timing jitter attacks (±1s perturbation)
- Robust against packet padding (up to 50%)
- Adversarial training integration for enhanced defense

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raw PCAP Traffic Input                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Traffic Preprocessing Pipeline                     │
│  ┌────────────────┐         ┌──────────────────────┐            │
│  │ Traffic Router │────────▶│ Multi-Process Feature│            │
│  │                │         │ Extraction Algorithm │            │
│  └────────────────┘         └──────────────────────┘            │
│         │                              │                        │
│         ▼                              ▼                        │
│  Session Flows              Features + Timestamps + Labels      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Timestamp-aware Semantic Encoder (TSE)                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │  Multi-Head Attention + Timestamp Encoding       │           │
│  │  Feed Forward + Layer Normalization              │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            KNN-based Graph Enhancement                          │
│  Original Five-tuple Graph + Similarity-based Edges             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│       Masked Graph Aggregation Module (MGAM)                    │
│  ┌──────────────────────────────────────────────────┐           │
│  │  GraphSAGE with Masking Strategy                 │           │
│  │  Neighborhood Sampling & Aggregation             │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                  Classification Output
              (Malicious / Benign / Multi-class)
```

## 🚀 Installation

### Requirements

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)

### Setup

bash

```bash
# Clone the repository
git clone https://github.com/chengxian611/Models-and-Raw-Traffic-Processing.git
cd Models-and-Raw-Traffic-Processing
```

## 🛡️ Adversarial Robustness

### Attack Types

1. Timing Jitter Attack
   - Perturbs packet timestamps: `t'ᵢ = tᵢ + ϵᵢ`
   - Tested with Δt ∈ {50, 100, 200, 500, 1000} ms
2. Packet Padding Attack
   - Adds random padding: `s'ᵢ = min(sᵢ + pᵢ, MTU)`
   - Padding ratios: ρ ∈ {10%, 30%, 50%}
3. Benign Mimicry Attack
   - Blends malicious and benign patterns: `Fₐdᵥ = α·Fₘₐₗ + (1-α)·Fᵦₑₙ`
   - Mixing ratios: α ∈ {0.25, 0.5, 0.75}

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We would like to thank:

- **Network Security Laboratory of USTC** for providing the USTC-TFC2016 dataset
- **ATG group at CTU** for maintaining the MCFP dataset
- The open-source community for PyTorch Geometric and related tools

------

**Note**: This repository is under active development. Some features mentioned in the paper may not yet be fully implemented. We are working to complete the codebase and will update this README accordingly.
