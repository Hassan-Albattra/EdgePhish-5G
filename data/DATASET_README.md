# EdgePhish-5G Dataset

##  Official Dataset Access

The full dataset is publicly available on Zenodo:

https://doi.org/10.5281/zenodo.19371661
## Overview

The EdgePhish-5G experiments use the URL Phishing Detection Dataset developed
in the associated prior work (Albattra et al., 2025).

The dataset was originally constructed from multiple public sources and is now officially released via Zenodo for reproducibility and citation purposes.

| Property | Value |
|---|---|
| Total URLs | 340,000 |
| Phishing | 170,000 (50%) |
| Legitimate | 170,000 (50%) |
| Collection period | January–March 2025 |
| Format | CSV (`url`, `label`) |
| Zenodo (DOI) |  https://doi.org/10.5281/zenodo.19371661|

---

## Sources

| Source | Count | Class | Verification |
|---|---|---|---|
| PhishTank† | 75,000 | Phishing | Community multi-verification |
| OpenPhish | 45,000 | Phishing | Algorithm-verified |
| Kaggle Phishing Dataset | 50,000 | Phishing | Crowd-sourced |
| Alexa Top 1M | 100,000 | Legitimate | Reputation-ranked |
| Common Crawl | 70,000 | Legitimate | Structural diversity |

†PhishTank data collected via archived snapshots (Jan–Mar 2025); timestamps verified.

---

##### Download Instructions

Download the dataset from Zenodo:

https://doi.org/10.5281/zenodo.19371661

Then place the CSV file in the `data/` directory:

```bash
cp <downloaded_path>/urls_dataset.csv data/urls_dataset.csv

---

## Dataset Statistics

```
URL Length:
  Phishing:   mean=75.3 chars, std=28.6
  Legitimate: mean=52.8 chars, std=19.4

Domain Length:
  Phishing:   mean=15.7 chars, std=6.2
  Legitimate: mean=11.3 chars, std=4.1

Special Characters per URL:
  Phishing:   mean=6.8, std=3.5
  Legitimate: mean=4.2, std=2.3

Subdomains Present:
  Phishing:   63.7%
  Legitimate: 41.2%
```

---

## Temporal Split (for reproducing results)

```
Training:   238,000 URLs  (rows 0–237,999)    Jan–mid Feb 2025
Validation:  51,000 URLs  (rows 238,000–288,999) mid Feb 2025
Test:        51,000 URLs  (rows 289,000–334,999) Mar 2025
Zero-day:     5,000 URLs  (rows 335,000–339,999) post-Mar 2025
```

**Critical:** Use the temporal split (not random split) to reproduce reported results.
Random splitting inflates performance by allowing campaign patterns to leak across splits.

---

## 5G Slice Annotation

The test set is annotated with 5G slice labels using `dataset/slice_annotation_rules.py`.

Slice definitions:
- **eMBB**: Consumer web (standard TLDs, consumer keywords, length > 30)
- **mMTC**: IoT (IoT keywords, numeric-heavy paths, IoT ports {1883, 5683, 8080})
- **URLLC**: Industrial (IP-only hosts, SCADA paths/ports {502, 102, 44818})

---

## Sample Dataset (Included)

`dataset/sample/urls_sample_1000.csv` — 1,000 URLs (500 phishing, 500 legitimate)
representing the distribution of the full dataset. Suitable for:
- Pipeline testing without GPU
- Feature extraction validation
- Model architecture verification

Column format:
```csv
url,label
https://paypal-secure.malicious.com/login,1
https://www.google.com/search,0
```

---

## Preprocessing Instructions

See `src/data_preprocessing.py` for the full pipeline:

1. **Deduplication** — SHA-256 hash on normalized URL string
2. **Normalization** — Lowercase, URL decode, trailing slash removal
3. **Validation** — Regex format check, max 512 characters
4. **Temporal split** — By row order (preserving collection timeline)
5. **Slice annotation** — Rule-based 5G slice labeling
6. **Synthetic URLLC** — 500 ICS-CERT hostname patterns + 1,500 generated variants

---

## License and Usage

The dataset is released for academic research purposes under CC BY 4.0.
Commercial use requires written permission from the authors.
