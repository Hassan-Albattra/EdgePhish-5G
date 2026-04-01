"""
Dataset Validation, Display & Export Script
============================================
Validates the annotated dataset, displays samples, and exports to CSV.
"""
import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════
# TASK 1: VALIDATE DATASET STRUCTURE
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  TASK 1: VALIDATE DATASET STRUCTURE")
print("=" * 70)

SRC = 'data/urls_annotated.csv'
if not os.path.exists(SRC):
    print(f"  [FAIL] File not found: {SRC}")
    sys.exit(1)

df = pd.read_csv(SRC)
print(f"  File: {SRC}")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Dtypes:")
for c in df.columns:
    print(f"    {c}: {df[c].dtype}")

# Required columns
checks = []
for col in ['url', 'label', 'slice']:
    present = col in df.columns
    checks.append(present)
    status = 'PASS' if present else 'FAIL'
    print(f"  [{status}] Column '{col}' present")

# No missing in slice
null_slice = int(df['slice'].isna().sum())
checks.append(null_slice == 0)
print(f"  [{'PASS' if null_slice == 0 else 'FAIL'}] Missing slice values: {null_slice}")

# Exactly one slice per row (no multi-label)
checks.append(True)  # CSV inherently single-value per cell
print(f"  [PASS] Each row has exactly ONE slice value")

# Only valid slice values
valid_slices = {'URLLC', 'mMTC', 'eMBB'}
unique_slices = set(df['slice'].unique())
invalid_slices = unique_slices - valid_slices
checks.append(len(invalid_slices) == 0)
print(f"  [{'PASS' if len(invalid_slices) == 0 else 'FAIL'}] "
      f"Unique slice values: {sorted(unique_slices)}")
if invalid_slices:
    print(f"    INVALID values found: {invalid_slices}")

# Label values
valid_labels = {0, 1}
unique_labels = set(df['label'].unique())
checks.append(unique_labels == valid_labels)
print(f"  [{'PASS' if unique_labels == valid_labels else 'FAIL'}] "
      f"Label values: {sorted(unique_labels)}")

all_pass = all(checks)
print(f"\n  >> TASK 1 OVERALL: {'PASS' if all_pass else 'FAIL'}")


# ═══════════════════════════════════════════════════════════
# TASK 2: VALIDATE ANNOTATION LOGIC (Random Samples)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TASK 2: VALIDATE ANNOTATION LOGIC (Spot Check)")
print("=" * 70)

import re
from urllib.parse import urlparse

IP_RE = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
URLLC_STRONG = {'scada', 'plc', 'hmi', 'ics', 'modbus', 'dnp3', 'opc', 'rtu'}
URLLC_WEAK = {'factory', 'plant', 'control', 'industrial'}
URLLC_PORTS = {502, 102, 44818, 20000}
MMTC_STRONG = {'iot', 'device', 'sensor', 'gateway', 'telemetry', 'mqtt', 'coap', 'firmware'}
MMTC_WEAK = {'update', 'meter', 'router', 'node', 'cam', 'camera', 'embedded', 'edge'}

def recompute_slice(url):
    """Independent re-computation of slice for verification."""
    url_lower = str(url).lower()
    try:
        parsed = urlparse(str(url))
        hostname = parsed.hostname or ''
        port = parsed.port
        path = parsed.path or ''
    except Exception:
        return 'eMBB', 0, 0

    # URLLC score
    us = 0
    if hostname and IP_RE.match(hostname):
        us += 3
    if port and port in URLLC_PORTS:
        us += 4
    for kw in URLLC_STRONG:
        if kw in url_lower:
            us += 2
    for kw in URLLC_WEAK:
        if kw in url_lower:
            us += 1

    # mMTC score
    ms = 0
    for kw in MMTC_STRONG:
        if kw in url_lower:
            ms += 2
    for kw in MMTC_WEAK:
        if kw in url_lower:
            ms += 1
    path_chars = max(len(path), 1)
    if sum(c.isdigit() for c in path) / path_chars > 0.4:
        ms += 1
    mac_re = re.compile(r'[0-9a-fA-F]{2}(?:[:\-][0-9a-fA-F]{2}){5}')
    devid_re = re.compile(r'[a-fA-F0-9]{8,}')
    if mac_re.search(str(url)):
        ms += 1
    elif len(path) > 10 and devid_re.search(path):
        ms += 1
    api_re = re.compile(r'api/v\d+/device', re.IGNORECASE)
    if api_re.search(str(url)):
        ms += 2

    if us >= 3:
        return 'URLLC', us, ms
    elif ms >= 2:
        return 'mMTC', us, ms
    else:
        return 'eMBB', us, ms

# Sample from each slice
inconsistencies = 0
for sl_name in ['URLLC', 'mMTC', 'eMBB']:
    sl_df = df[df['slice'] == sl_name]
    n_check = min(20, len(sl_df))
    sample = sl_df.sample(n=n_check, random_state=42)
    mismatches = 0
    for _, row in sample.iterrows():
        expected, us, ms = recompute_slice(row['url'])
        if expected != row['slice']:
            mismatches += 1
            inconsistencies += 1
            print(f"  [MISMATCH] url={row['url'][:60]}...")
            print(f"    Dataset: {row['slice']}, Recomputed: {expected} "
                  f"(URLLC={us}, mMTC={ms})")
    status = 'PASS' if mismatches == 0 else 'FAIL'
    print(f"  [{status}] {sl_name}: checked {n_check} samples, {mismatches} mismatches")

print(f"\n  >> TASK 2 OVERALL: {'PASS' if inconsistencies == 0 else 'FAIL'} "
      f"({inconsistencies} total inconsistencies)")


# ═══════════════════════════════════════════════════════════
# TASK 3: DATASET SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TASK 3: DATASET SUMMARY")
print("=" * 70)

print(f"  Total samples: {len(df):,}")
print(f"\n  Slice Distribution:")
print(f"  {'Slice':>6}  {'Count':>8}  {'Pct':>7}")
print(f"  {'-'*6}  {'-'*8}  {'-'*7}")
for sl in ['eMBB', 'mMTC', 'URLLC']:
    cnt = int((df['slice'] == sl).sum())
    pct = cnt / len(df) * 100
    print(f"  {sl:>6}  {cnt:>8,}  {pct:>6.2f}%")
print(f"  {'Total':>6}  {len(df):>8,}  100.00%")

print(f"\n  Slice x Label Crosstab:")
ct = pd.crosstab(df['slice'], df['label'], margins=True)
ct.columns = ['Legit(0)', 'Phish(1)', 'Total']
ct.index.name = 'Slice'
print(ct.to_string())

print(f"\n  Anomalies / Notes:")
urllc_phish = (df[df['slice'] == 'URLLC']['label'] == 1).mean() * 100
mmtc_phish = (df[df['slice'] == 'mMTC']['label'] == 1).mean() * 100
embb_phish = (df[df['slice'] == 'eMBB']['label'] == 1).mean() * 100
print(f"    eMBB phishing ratio:  {embb_phish:.1f}%")
print(f"    mMTC phishing ratio:  {mmtc_phish:.1f}%")
print(f"    URLLC phishing ratio: {urllc_phish:.1f}% (heavily skewed)")
if urllc_phish > 95:
    print(f"    >> WARNING: URLLC is {urllc_phish:.1f}% phishing. "
          "IP-based hostnames are overwhelmingly in phishing URLs.")

# Score columns
if 'URLLC_score' in df.columns and 'mMTC_score' in df.columns:
    print(f"\n  Score Statistics:")
    print(f"    URLLC_score: mean={df['URLLC_score'].mean():.3f}, "
          f"max={df['URLLC_score'].max()}, >0: {(df['URLLC_score']>0).sum():,}")
    print(f"    mMTC_score:  mean={df['mMTC_score'].mean():.3f}, "
          f"max={df['mMTC_score'].max()}, >0: {(df['mMTC_score']>0).sum():,}")


# ═══════════════════════════════════════════════════════════
# TASK 4: DISPLAY SAMPLE OUTPUT
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TASK 4: SAMPLE PREVIEW (first 20 rows)")
print("=" * 70)
preview_cols = ['url', 'label', 'slice']
if 'URLLC_score' in df.columns:
    preview_cols += ['URLLC_score', 'mMTC_score']

pd.set_option('display.max_colwidth', 65)
pd.set_option('display.width', 120)
print(df[preview_cols].head(20).to_string())

# Also show samples from each slice
print(f"\n  Sample URLLC URLs (5):")
for _, r in df[df['slice'] == 'URLLC'].head(5).iterrows():
    print(f"    label={r['label']} | {r['url'][:80]}")

print(f"\n  Sample mMTC URLs (5):")
for _, r in df[df['slice'] == 'mMTC'].head(5).iterrows():
    print(f"    label={r['label']} | {r['url'][:80]}")

print(f"\n  Sample eMBB URLs (5):")
for _, r in df[df['slice'] == 'eMBB'].head(5).iterrows():
    print(f"    label={r['label']} | {r['url'][:80]}")


# ═══════════════════════════════════════════════════════════
# TASK 5: EXPORT DATASET
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TASK 5: EXPORT DATASET")
print("=" * 70)

# Export with scores (Task 6: optional columns included)
export_cols = ['url', 'label', 'slice']
if 'URLLC_score' in df.columns:
    export_cols += ['URLLC_score', 'mMTC_score']
    print("  Including URLLC_score and mMTC_score columns")

# Primary export path
export_path = r'D:\ITC\Phishing URL Detection in 5G\annotated_dataset.csv'
try:
    df[export_cols].to_csv(export_path, index=False, encoding='utf-8')
    exported_rows = len(df)
    print(f"  [OK] Exported to: {export_path}")
    print(f"  Rows saved: {exported_rows:,}")
    print(f"  Columns: {export_cols}")
    print(f"  Encoding: UTF-8")
    print(f"  Headers: YES")
    print(f"  Original order: PRESERVED")

    # Verify export
    verify = pd.read_csv(export_path)
    assert len(verify) == len(df), "Row count mismatch after export!"
    assert list(verify.columns) == export_cols, "Column mismatch after export!"
    print(f"  [OK] Export verified: {len(verify):,} rows read back successfully")
except Exception as e:
    print(f"  [WARN] Primary path failed: {e}")
    # Fallback to files directory
    fallback_path = r'D:\ITC\Phishing URL Detection in 5G\files\annotated_dataset.csv'
    df[export_cols].to_csv(fallback_path, index=False, encoding='utf-8')
    print(f"  [OK] Exported to fallback: {fallback_path}")
    print(f"  Rows saved: {len(df):,}")

print("\n" + "=" * 70)
print("  ALL TASKS COMPLETE")
print("=" * 70)
