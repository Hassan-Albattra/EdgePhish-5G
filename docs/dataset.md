# Dataset Documentation

## Source

## Source

The EdgePhish-5G project uses the **EdgePhish-5G Dataset** published by Hassan AlBattra:

- **Zenodo (DOI)**: https://doi.org/10.5281/zenodo.19371661
- **Original Sources**: PhishTank, OpenPhish, Kaggle, Alexa Top Sites, Common Crawl
- **Collection Period**: January–March 2025
- **License**: CC BY 4.0

## Dataset Overview

| Property | Value |
|----------|-------|
| Total URLs | 340,000 |
| Phishing URLs | 170,000 (50.0%) |
| Legitimate URLs | 170,000 (50.0%) |
| Format | CSV |
| Encoding | UTF-8 |
| Duplicates | 277 (0.08%) |
| URL Length (mean ± std) | 79.5 ± 56.5 characters |

## Schema

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `url` | string | The full URL string |
| `label` | int | `0` = legitimate, `1` = phishing |

### Annotation Columns (Added by Pipeline)

| Column | Type | Description |
|--------|------|-------------|
| `slice` | string | 5G network slice: `eMBB`, `mMTC`, or `URLLC` |
| `URLLC_score` | int | URLLC heuristic score (for auditing) |
| `mMTC_score` | int | mMTC heuristic score (for auditing) |

## Obtaining the Dataset

The dataset is **not included** in this repository. To obtain it:

### Option 1: Automatic Download

```bash
python scripts/prepare_dataset.py
```
### Download Dataset

The dataset is publicly available on Zenodo:

https://doi.org/10.5281/zenodo.19371661
---

### Option 1: Manual Download (Recommended)

1. Open the Zenodo link above  
2. Download the dataset file (`urls_dataset.csv`)  
3. Place it in:


---

### Option 2: Quick Test (Sample Data)

For quick experimentation, you can use the included sample dataset:

```bash
cp dataset/sample/urls_sample_1000.csv data/urls_dataset.csv
## 5G Slice Annotation

### What Are Slices?

In 5G networks, **network slicing** partitions the physical network into logical slices optimized for different traffic types:

- **eMBB** (Enhanced Mobile Broadband): Consumer web browsing, streaming, social media
- **mMTC** (Massive Machine-Type Communications): IoT devices, sensors, firmware updates
- **URLLC** (Ultra-Reliable Low-Latency Communications): Industrial control, SCADA, real-time systems

### Why Annotate URLs with Slices?

EdgePhish-5G deploys at the UPF, where traffic is already classified by slice. Different slices have different:
- **Latency budgets** (URLLC < 1ms vs. eMBB < 10ms)
- **URL patterns** (consumer vs. IoT vs. industrial)
- **Phishing attack surfaces** (URLLC has different threats than eMBB)

Evaluating detection performance per slice reveals whether the model degrades on underrepresented traffic types.

### Annotation Method

The slice annotation uses a **score-based heuristic system**, implemented in `src/data_preprocessing.py` (`SliceAnnotator` class). This is **NOT** ground-truth labeling from a 5G operator — it is a rule-based proxy for benchmarking.

#### URLLC Scoring (threshold ≥ 3)

| Signal | Points |
|--------|--------|
| IP-based hostname (e.g., `192.168.1.100`) | +3 |
| Industrial ports: 502 (Modbus), 102 (S7), 44818 (EtherNet/IP), 20000 (DNP3) | +4 |
| Strong ICS keywords: `scada`, `plc`, `hmi`, `ics`, `modbus`, `dnp3`, `opc`, `rtu` | +2 each |
| Weak industrial keywords: `factory`, `plant`, `control`, `industrial` | +1 each |

#### mMTC Scoring (threshold ≥ 2)

| Signal | Points |
|--------|--------|
| Strong IoT keywords: `iot`, `device`, `sensor`, `gateway`, `telemetry`, `mqtt`, `coap`, `firmware` | +2 each |
| Weak IoT keywords: `update`, `meter`, `router`, `node`, `cam`, `camera`, `embedded`, `edge` | +1 each |
| High numeric ratio in URL path (>40%) | +1 |
| Device-like IDs or MAC address patterns | +1 |
| REST API device patterns (`api/v1/device`) | +2 |

#### Priority Rule

```
if URLLC_score >= 3 → URLLC
elif mMTC_score >= 2 → mMTC  
else → eMBB (default)
```

If both URLLC and mMTC thresholds are met, **URLLC takes priority** (highest criticality).

### Resulting Distribution

| Slice | Count | Percentage | Phishing Ratio |
|-------|-------|-----------|---------------|
| eMBB | 311,280 | 91.55% | 46.8% |
| mMTC | 21,239 | 6.25% | 79.6% |
| URLLC | 7,481 | 2.20% | 99.2% |

### Known Anomalies

> **URLLC is 99.2% phishing.** This occurs because IP-based hostnames (the primary URLLC signal at +3 points) are overwhelmingly used by phishing URLs, not legitimate ones. Only 58 legitimate URLs in the entire 340K dataset have URLLC scores ≥ 3. This imbalance is a direct consequence of applying industrial traffic heuristics to a general web URL dataset.

### Critical Disclaimer

The slice annotation is a **heuristic approximation** for research benchmarking purposes. It does NOT represent:
- Actual 5G network slice assignments from an operator
- NSSAI (Network Slice Selection Assistance Information) labels
- Real UPF traffic classification

In a production 5G deployment, the UPF would provide the actual slice ID via PFCP session context, making heuristic annotation unnecessary.

## Dataset Splitting

The pipeline uses the following split strategy:

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% (238,000) | Model training |
| Validation | 15% (51,000) | Hyperparameter tuning, early stopping |
| Test | 15% (51,000) | Final evaluation |
| Zero-Day | 5,000 (held out) | Post-training campaign simulation |

Splitting is performed by **temporal index** (preserving insertion order) to simulate the real scenario where models must detect campaigns that emerge after training.

## Data Quality Notes

1. **277 duplicate URLs** (0.08%) exist in the raw dataset — these are not removed to maintain dataset integrity
2. **No non-ASCII URLs** — all URLs use standard ASCII encoding
3. **URL lengths range from 4 to 2,337 characters** — URLs beyond 512 characters are truncated
4. The dataset does not include HTTP headers, page content, or WHOIS features — only raw URL strings
