# EdgePhish-5G: Real-Time Phishing URL Detection in 5G Core Networks

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: 340K URLs](https://img.shields.io/badge/Dataset-340K%20URLs-orange.svg)](https://github.com/Hassan-Albattra/URL_Phishing_Detection_Dataset)

## Overview

**EdgePhish-5G** is a lightweight hybrid phishing URL detection system designed for deployment at the Multi-access Edge Computing (MEC) layer of 5G core networks. It combines a distilled DistilBERT branch with a compressed TF-IDF branch via a learnable fusion gate, achieving **98.63% F1-score** while meeting the **10 ms latency budget** required by 5G User Plane Function (UPF) inline inspection.

## Research Motivation

5G networks process **>1 million URLs/min** per cell site, with phishing campaigns comprising 36% of cyber incidents (APWG Q3 2024). Traditional cloud-based detectors introduce 50–200 ms round-trip latency, violating the ≤10 ms MEC processing budget. EdgePhish-5G addresses this gap by co-locating a lightweight detector at the UPF via PFCP-based URL extraction.

## Main Contributions

1. **Hybrid Architecture**: Fuses DistilBERT semantic features with compressed TF-IDF lexical features through a learnable α-gate
2. **Knowledge Distillation**: 3-phase training (warm-up → KD → QAT) compresses a Previously Developed Hybrid Model (BERT-base FP32, 99.36% F1) into a model 4.2× smaller
3. **5G-Native Deployment**: UPF co-location via PFCP Session Modification with fail-open timeout
4. **Edge Optimization**: INT8 Quantization-Aware Training achieves 4.91 ms on Intel Xeon-D
5. **5G Slice-Stratified Benchmark**: Novel evaluation across eMBB, mMTC, and URLLC network slices

## Repository Structure

```
EdgePhish-5G/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
│
├── src/                         # Core source code
│   ├── data_preprocessing.py    # URL normalization, splitting, slice annotation
│   ├── feature_extraction.py    # TF-IDF compression + BERT tokenization
│   ├── model_hybrid.py          # EdgePhish-5G architecture + simulation mode
│   ├── training.py              # 3-phase training, baselines, ablation
│   └── evaluation.py            # Metrics, figures, tables
│
├── configs/
│   └── experiment_config.yaml   # All hyperparameters (single source of truth)
│
├── scripts/
│   ├── prepare_dataset.py       # Download & format the GitHub dataset
│   ├── run_pipeline.py          # Full pipeline execution
│   ├── run_validation.py        # Technical validation & reproducibility audit
│   └── validate_export.py       # Dataset validation & CSV export
│
├── figures/                     # Pre-generated paper figures (PNG)
│   ├── fig3_f1_comparison.png
│   ├── fig4_latency_throughput.png
│   ├── fig5_ablation.png
│   ├── fig6_roc_curves.png
│   ├── fig7_confusion_matrix.png
│   ├── fig8_zero_day_slice.png
│   └── fig9_queuing_model.png
│
├── paper/                       # IEEE conference paper (LaTeX)
│   ├── main.tex
│   ├── references.bib
│   └── figures/                 # Copies for LaTeX compilation
│
├── docs/                        # Documentation
│   ├── dataset.md               # Dataset source, schema, annotation
│   ├── reproducibility.md       # How to reproduce all results
│   └── limitations.md           # Known limitations & assumptions
│
└── results/                     # (Generated at runtime, not tracked)
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/EdgePhish-5G.git
cd EdgePhish-5G
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# Install requirements
pip install -r requirements.txt
```

### 3. Prepare the Dataset

The dataset is **not included** in this repository due to size. Download it from:
- **Source**: [Hassan-Albattra/URL_Phishing_Detection_Dataset](https://github.com/Hassan-Albattra/URL_Phishing_Detection_Dataset)

```bash
# Automatic download and preparation
python scripts/prepare_dataset.py
```

This will:
- Download the 340K URL dataset
- Convert labels to numeric format (0=legitimate, 1=phishing)
- Apply 5G slice annotation (eMBB / mMTC / URLLC)
- Save to `data/urls_dataset.csv`

### 4. Run the Full Pipeline

```bash
# Run preprocessing → feature extraction → training → evaluation
python scripts/run_pipeline.py
```

### 5. Reproduce Results

```bash
# Full technical validation with reproducibility audit
python scripts/run_validation.py
```

Results are saved to `results/technical_validation.json`.

## Dataset

| Property | Value |
|----------|-------|
| **Total URLs** | 340,000 |
| **Phishing** | 170,000 (50%) |
| **Legitimate** | 170,000 (50%) |
| **Source** | [GitHub](https://github.com/Hassan-Albattra/URL_Phishing_Detection_Dataset) |
| **Collection Period** | Jan–Dec 2023 |
| **Format** | CSV (`url`, `label`, `slice`) |

### 5G Slice Annotation

Each URL is annotated with a 5G network slice type using **heuristic rules** (not ground-truth operator labels):

| Slice | Count | Criteria |
|-------|-------|----------|
| **eMBB** | 311,280 (91.6%) | Default — consumer web traffic |
| **mMTC** | 21,239 (6.2%) | IoT/device keywords, API patterns |
| **URLLC** | 7,481 (2.2%) | IP-based hosts, SCADA/ICS keywords, industrial ports |

> **Important**: Slice labels are heuristic proxies for 5G traffic types, not real operator-assigned labels. See [docs/dataset.md](docs/dataset.md) for details.

## Model Architecture

EdgePhish-5G is a hybrid model with two feature branches fused by a learnable gate:

```
URL Input
    │
    ├──→ DistilBERT Branch ──→ CLS projection (256-d) ──┐
    │                                                     │
    │                                                     ├──→ Fusion Gate (α) ──→ Classification Head ──→ P(phishing)
    │                                                     │
    └──→ TF-IDF Branch ─────→ SVD → Dense (256-d) ──────┘
```

### Previously Developed Hybrid Model

The **Previously Developed Hybrid Model** serves as the teacher for knowledge distillation. It is a full BERT-base + TF-IDF model trained with FP32 precision, achieving 99.36% F1-score. The student (EdgePhish-5G) distills this knowledge through:

1. **Phase 1** — Head warm-up (3 epochs, BERT frozen)
2. **Phase 2** — Full knowledge distillation (15 epochs, T=4, λ=0.70)
3. **Phase 3** — INT8 Quantization-Aware Training (3 epochs)

### Simulation Mode

For environments without PyTorch/GPU, a simulation mode (`EdgePhish5GSimulation`) uses sklearn classifiers as a proxy. Simulation results are ~4% below full model F1 because the DistilBERT semantic branch is approximated.

## Results

### Main Results (Paper, Full Model)

| Model | F1 (%) | Precision (%) | Recall (%) | AUC (%) |
|-------|--------|---------------|------------|---------|
| Logistic Regression + TF-IDF | 94.79 | 94.29 | 95.30 | 98.74 |
| Random Forest + TF-IDF | 96.42 | 95.89 | 96.96 | 99.52 |
| SVM + TF-IDF | 95.65 | 95.45 | 95.86 | 98.91 |
| **EdgePhish-5G (INT8-QAT)** | **98.63** | **98.89** | **98.37** | **99.84** |

### Latency (Single-URL Inference)

| Platform | FP32 | INT8-QAT | SLA |
|----------|------|----------|-----|
| Intel Xeon-D 2100 | 7.83 ms | **4.91 ms** | ✅ <10 ms |
| Jetson AGX Orin | 9.46 ms | **6.12 ms** | ✅ <10 ms |
| ARM Cortex-A72 | 18.34 ms | **13.71 ms** | ❌ >10 ms |

## Key Limitations

1. **Dataset**: Generic web URLs, not real 5G UPF traffic captures
2. **Slice Annotation**: Heuristic rules, not ground-truth 5G labels
3. **ECH/DoH**: No TLS 1.3 Encrypted Client Hello or DNS-over-HTTPS support
4. **Latency**: Hardware benchmarks are from controlled environments; real MEC may vary
5. **5G Core**: UPF integration is architectural — not validated with a real PFCP stack

See [docs/limitations.md](docs/limitations.md) for the complete list.

## Citation

If you use this work, please cite:

```bibtex
@article{edgephish5g2025,
  author  = {H. Albattra and R. A. {Abul Seoud} and D. A. Salem},
  title   = {Real-Time Phishing {URL} Detection in {5G} Core Networks 
             Using Lightweight Hybrid {BERT}-{TF-IDF} with Edge Intelligence},
  journal = {IEEE Access},
  year    = {2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Hassan Albattra** — [GitHub](https://github.com/Hassan-Albattra)
- Dataset: [URL_Phishing_Detection_Dataset](https://github.com/Hassan-Albattra/URL_Phishing_Detection_Dataset)
