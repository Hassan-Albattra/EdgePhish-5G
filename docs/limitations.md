# Known Limitations

This document provides a comprehensive and transparent account of all known limitations in the EdgePhish-5G project. These limitations affect the generalizability of reported results and should be considered when interpreting findings or planning deployment.


---

## A. Data Limitations

### A1. Generic Dataset (HIGH Severity)
The 340K URL dataset is sourced from general web crawling, not from actual 5G UPF traffic captures. Real 5G network traffic would include:
- Mobile-specific URLs (app deep links, AMP pages)
- Carrier-injected headers
- Different phishing distributions per slice

**Impact**: Results may not transfer directly to production 5G environments.

### A2. Heuristic Slice Annotation (HIGH Severity)
The `slice` column is assigned by rule-based scoring (IP patterns, keywords, ports), not by actual 5G NSSAI labels from a network operator. The heuristic may:
- Misclassify URLs with ambiguous patterns
- Over-assign eMBB as the default fallback
- Not capture the real distribution of traffic across slices

**Impact**: Slice-stratified results (Table VI, Fig. 8) are proxies, not ground-truth evaluations.

### A3. URLLC Approximation (HIGH Severity)
Real URLLC phishing targets industrial control systems (SCADA, PLCs, HMIs). The dataset contains no actual SCADA phishing campaigns — URLLC classification relies on IP-based hostnames and port patterns that happen to appear in general phishing URLs.

**Impact**: The 99.2% phishing ratio in URLLC is an artifact of the heuristic, not a reflection of real URLLC threats.

### A4. URLLC Imbalance (MEDIUM Severity)
Only 7,481 URLs (2.2%) are classified as URLLC, with 99.2% being phishing. This severe class imbalance:
- Makes URLLC F1 scores unreliable (trivially high in simulation)
- Does not reflect real industrial network traffic distributions

### A5. Temporal Simulation (MEDIUM Severity)
The temporal split uses index order as a proxy for time. The dataset lacks actual timestamp columns, so the "temporal" property is simulated, not real.

### A6. No Real Timestamps (LOW Severity)
The dataset was collected Jan–Dec 2023 but individual URL timestamps are not available. This prevents true temporal analysis of phishing campaign evolution.

---

## B. Model Limitations

### B1. No TLS 1.3 ECH Support (HIGH Severity)
Encrypted Client Hello (ECH) encrypts the SNI field in the TLS handshake. When ECH is deployed:
- The UPF cannot extract the target hostname from the TLS handshake
- URL-based detection becomes impossible for ECH-encrypted connections
- APNIC estimates ~5-15% of mobile TLS traffic uses ECH in 2024

**Impact**: EdgePhish-5G cannot inspect ECH-encrypted traffic.

### B2. No DNS-over-HTTPS (DoH) Support (HIGH Severity)
DoH encrypts DNS queries within HTTPS, preventing DNS-level URL extraction. If DoH is used:
- The resolver hop is invisible to the UPF
- The model can only see the DoH server IP, not the target domain

**Impact**: ~15% of mobile browser traffic (Firefox, Chrome) uses DoH by default.

### B3. No TinyBERT / MobileBERT Comparison (MEDIUM Severity)
The paper uses DistilBERT as the only student model. Alternative compact transformers (TinyBERT, MobileBERT, ALBERT) are not compared, which limits the ability to assess whether DistilBERT is the optimal choice.

### B4. Simulation Mode Limitations (MEDIUM Severity)
The simulation mode (`EdgePhish5GSimulation`) uses sklearn classifiers as a proxy for the full PyTorch model. In simulation mode:
- Temperature has no effect (no KD training loop)
- Fusion strategy has no effect (no real branch fusion)
- Alpha is fixed at 0.30 (no learnable parameter)
- Results are ~4% below fully-trained model

### B5. Missing Baseline Implementations (MEDIUM Severity)
CNN (char-level) and RNN-GRU baselines are referenced in the paper (Table IV) but have no code implementation. Their reported metrics cannot be independently verified.

### B6. Teacher Model Not Provided (MEDIUM Severity)
The Previously Developed Hybrid Model (teacher, BERT-base FP32, 99.36% F1) checkpoint is not included. Only pre-computed soft logits are referenced during distillation.

### B7. SVD Dimension Inconsistency (LOW Severity)
`experiment_config.yaml` specifies `svd_components: 2048`, but the `ExperimentOrchestrator` in `training.py` hardcodes 256. This causes a ~0.5-2% F1 difference in baseline results.

---

## C. Deployment Limitations

### C1. Hardcoded Latency Values (HIGH Severity)
All edge platform latencies reported in the paper are hardcoded constants in `evaluation.py`:
- Intel Xeon-D 2100: 4.91 ms (INT8-QAT)
- Jetson AGX Orin: 6.12 ms (INT8-QAT)
- ARM Cortex-A72: 13.71 ms (INT8-QAT)

These values were measured on specific hardware; they are not computed at runtime and cannot be independently verified without access to the same hardware.

### C2. No Real 5G Core Validation (HIGH Severity)
The UPF co-location architecture is described but not validated with:
- A real 5G core implementation (e.g., free5GC, Open5GS)
- Actual PFCP session modification messages
- Real N6 interface traffic

All 5G integration claims are architectural proposals, not tested implementations.

### C3. ARM SLA Violation (MEDIUM Severity)
The ARM Cortex-A72 platform achieves P99 latency of 13.71 ms, which exceeds the 10 ms SLA budget. The paper acknowledges this requires dual-instance horizontal scaling, but this mitigation is not tested.

### C4. Horizontal Scaling Required (MEDIUM Severity)
At >9,000 URLs/min per core, horizontal scaling is needed. The M/D/1 queuing model validates this theoretically, but no actual multi-instance deployment is tested.

### C5. PFCP Integration Not Validated (MEDIUM Severity)
The paper describes PFCP-based URL extraction via Session Modification with FAR redirection. This protocol flow is described but not implemented or tested with real PFCP message exchanges.

---

## Summary

| Category | HIGH | MEDIUM | LOW | Total |
|----------|------|--------|-----|-------|
| Data | 3 | 2 | 1 | 6 |
| Model | 2 | 5 | 1 | 8 |
| Deployment | 2 | 3 | 0 | 5 |
| **Total** | **7** | **10** | **2** | **19** |

These limitations are disclosed transparently in accordance with IEEE reproducibility standards. They do not invalidate the research contributions but define the boundaries within which the results should be interpreted.
