"""
Prepare the GitHub dataset for EdgePhish-5G pipeline.
Converts label format and adds required columns.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

print("Loading GitHub dataset...")
df = pd.read_csv('data/github_dataset/balanced_urls.csv')
print(f"  Raw shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Label values: {df['label'].unique()[:5]}")

# Convert labels: phishing->1, legitimate->0
df['label'] = df['label'].map({'phishing': 1, 'legitimate': 0})
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Reorder columns to match expected format: url, label, source, slice
df = df[['url', 'label']]

# Add source column
df['source'] = df['label'].map({1: 'phishing_sample', 0: 'legitimate_sample'})

# Add slice column (eMBB default for web URLs)
df['slice'] = 'eMBB'

# Shuffle with seed for reproducibility
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"  Processed shape: {df.shape}")
print(f"  Label dist: phishing={df['label'].sum()}, legitimate={len(df)-df['label'].sum()}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Head:")
print(df.head(3).to_string())

# Save
out_path = 'data/urls_dataset.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path} ({len(df)} rows)")
