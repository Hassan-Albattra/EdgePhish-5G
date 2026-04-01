"""
Practical Pipeline: 50K subset from 340K dataset, paper-consistent params
"""
import sys, os, time, warnings, json
import numpy as np
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  EdgePhish-5G Pipeline — 50K Subset + Paper Params")
print("=" * 70)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                              recall_score, roc_auc_score, confusion_matrix)

# Load full dataset, sample 50K
print("\n[STEP 1] Loading & sampling dataset...")
df = pd.read_csv('data/urls_dataset.csv')
print(f"  Full dataset: {len(df):,} URLs")

# Stratified sample of 50K
phish_df = df[df['label'] == 1].sample(n=25000, random_state=42)
legit_df = df[df['label'] == 0].sample(n=25000, random_state=42)
df_sample = pd.concat([phish_df, legit_df], ignore_index=True)
df_sample = df_sample.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"  Sampled: {len(df_sample):,} (25K phish + 25K legit)")
print(f"  Columns: {df_sample.columns.tolist()}")
print(f"  Labels: {df_sample['label'].value_counts().to_dict()}")

# Split: 70/15/15
train_df, temp_df = train_test_split(df_sample, test_size=0.30, random_state=42, stratify=df_sample['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

train_urls = train_df['url'].tolist()
train_labels = train_df['label'].values
val_urls = val_df['url'].tolist()
val_labels = val_df['label'].values
test_urls = test_df['url'].tolist()
test_labels = test_df['label'].values

print(f"  train={len(train_urls):,}, val={len(val_urls):,}, test={len(test_urls):,}")
print(f"  Train phishing: {train_labels.sum():,}/{len(train_labels):,}")

# [STEP 2] Feature Extraction
print("\n[STEP 2] Feature Extraction...")
t0 = time.time()
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor(
    tfidf_config={
        'ngram_range': (2, 4),
        'max_features': 20000,
        'chi2_k': 3000,
        'svd_components': 256
    },
    seed=42
)
extractor.fit_tfidf(train_urls, train_labels)
X_train = extractor.extract_tfidf(train_urls)
X_val = extractor.extract_tfidf(val_urls)
X_test = extractor.extract_tfidf(test_urls)
tok_train = extractor.extract_bert_tokens(train_urls)
tok_val = extractor.extract_bert_tokens(val_urls)
tok_test = extractor.extract_bert_tokens(test_urls)
print(f"  TF-IDF: {X_train.shape}, SVD variance captured")
print(f"  Time: {time.time()-t0:.1f}s")

# [STEP 3] Baselines + EdgePhish-5G Simulation
print("\n[STEP 3] Training baselines + EdgePhish-5G simulation...")
t0 = time.time()

from model_hybrid import EdgePhish5GSimulation, EdgePhishConfig, SklearnBaselineFactory
from training import compute_metrics

# Baselines
baselines = SklearnBaselineFactory.get_all()
all_results = {}

for name, model in baselines.items():
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)
    m = compute_metrics(test_labels, y_pred, y_prob)
    all_results[name] = m
    print(f"  {name}: F1={m['f1_score']*100:.2f}%, ACC={m['accuracy']*100:.2f}%, AUC={m['roc_auc']*100:.2f}%")

# EdgePhish-5G Simulation
config = EdgePhishConfig(tfidf_input_dim=X_train.shape[1])
sim = EdgePhish5GSimulation(config=config, seed=42)
sim.fit(X_train, tok_train, train_labels)
test_probs, test_preds, alpha_val = sim.predict(X_test, tok_test)
m_ep = compute_metrics(test_labels, test_preds, test_probs)
all_results['EdgePhish-5G (Simulation)'] = m_ep

print(f"\n  EdgePhish-5G (Sim): F1={m_ep['f1_score']*100:.2f}%, ACC={m_ep['accuracy']*100:.2f}%, "
      f"AUC={m_ep['roc_auc']*100:.2f}%, Alpha={alpha_val:.4f}")
print(f"    Precision: {m_ep['precision']*100:.2f}%")
print(f"    Recall:    {m_ep['recall']*100:.2f}%")
print(f"    FPR:       {m_ep['fpr']*100:.2f}%")
print(f"    FNR:       {m_ep['fnr']*100:.2f}%")

cm = confusion_matrix(test_labels, test_preds)
print(f"    CM: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")
print(f"  Time: {time.time()-t0:.1f}s")

# [STEP 4] Latency
print("\n[STEP 4] Latency measurement...")
single_tfidf = X_test[0:1]
single_tok = {k: v[0:1] for k, v in tok_test.items()}
for _ in range(200):
    sim.predict_proba(single_tfidf, single_tok)
t_lat = time.perf_counter()
for _ in range(2000):
    sim.predict_proba(single_tfidf, single_tok)
latency_ms = (time.perf_counter() - t_lat) / 2000 * 1000
print(f"  Simulation Latency: {latency_ms:.3f} ms/URL")
print(f"  Simulation Throughput: {1000.0/latency_ms:.0f} URL/s")

# [STEP 5] Temperature Ablation
print("\n[STEP 5] Temperature Ablation...")
n_abl = min(5000, len(train_urls))
X_abl_tr = X_train[:n_abl]
y_abl_tr = train_labels[:n_abl]
tok_abl_tr = {k: v[:n_abl] for k, v in tok_train.items()}
temp_results = []
for T in [1, 2, 4, 6, 8, 10]:
    cfg = EdgePhishConfig(tfidf_input_dim=X_train.shape[1])
    cfg.temperature = float(T)
    sm = EdgePhish5GSimulation(config=cfg, seed=42)
    sm.fit(X_abl_tr, tok_abl_tr, y_abl_tr)
    vp, vpred, va = sm.predict(X_val, tok_val)
    vm = compute_metrics(val_labels, vpred, vp)
    temp_results.append({'T': T, 'f1': vm['f1_score']*100, 'acc': vm['accuracy']*100})
    print(f"  T={T}: F1={vm['f1_score']*100:.2f}%")

# [STEP 6] Fusion Ablation
print("\n[STEP 6] Fusion Ablation...")
fusion_results = []
for strat in ['alpha_gate', 'concat', 'equal_weight']:
    cfg = EdgePhishConfig(tfidf_input_dim=X_train.shape[1])
    cfg.fusion_strategy = strat
    sm = EdgePhish5GSimulation(config=cfg, seed=42)
    sm.fit(X_abl_tr, tok_abl_tr, y_abl_tr)
    vp, vpred, va = sm.predict(X_val, tok_val)
    vm = compute_metrics(val_labels, vpred, vp)
    fusion_results.append({'strategy': strat, 'f1': vm['f1_score']*100, 'alpha': va})
    print(f"  {strat}: F1={vm['f1_score']*100:.2f}%, alpha={va:.3f}")

# [STEP 7] Synthetic URLLC
print("\n[STEP 7] URLLC Slice (Synthetic)...")
from data_preprocessing import EdgePhishDataset
ds = EdgePhishDataset(data_path='data/urls_dataset.csv', seed=42)
ds.load_and_preprocess()
slices = ds.get_slice_subsets()
for sn, sdf in slices.items():
    if len(sdf) >= 10:
        su = sdf['url'].tolist()
        sl = sdf['label'].values
        Xs = extractor.extract_tfidf(su)
        ts = extractor.extract_bert_tokens(su)
        sp, spred, sa = sim.predict(Xs, ts)
        sm_m = compute_metrics(sl, spred, sp)
        print(f"  {sn}: n={len(sdf):,}, F1={sm_m['f1_score']*100:.2f}%, alpha={sa:.3f}")

# Save
def convert_np(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_np(i) for i in obj]
    return obj

os.makedirs('results', exist_ok=True)
json.dump(convert_np({
    'dataset_size': 50000,
    'split': {'train': len(train_urls), 'val': len(val_urls), 'test': len(test_urls)},
    'all_results': all_results,
    'alpha': alpha_val,
    'latency_ms': latency_ms,
    'temperature_ablation': temp_results,
    'fusion_ablation': fusion_results,
}), open('results/pipeline_results.json', 'w'), indent=2)

# COMPARISON TABLE
print("\n" + "=" * 70)
print("  PAPER vs SIMULATION — COMPARISON TABLE")
print("=" * 70)
paper = {
    'Logistic Regression': 94.79,
    'Random Forest': 96.42,
    'SVM': 95.65,
    'EdgePhish-5G (INT8-QAT)': 98.63,
}
sim_map = {
    'Logistic Regression': 'Logistic Regression + TF-IDF',
    'Random Forest': 'Random Forest + TF-IDF',
    'SVM': 'SVM + TF-IDF',
    'EdgePhish-5G (INT8-QAT)': 'EdgePhish-5G (Simulation)',
}
print(f"  {'Model':<35} {'Paper F1':>10} {'Sim F1':>10} {'Delta':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
for name, paper_f1 in paper.items():
    sk = sim_map.get(name, name)
    if sk in all_results:
        sf1 = all_results[sk]['f1_score'] * 100
    else:
        sf1 = float('nan')
    d = sf1 - paper_f1
    print(f"  {name:<35} {paper_f1:>9.2f}% {sf1:>9.2f}% {d:>+7.2f}%")
print("=" * 70)
print("  DONE. Results in results/pipeline_results.json")
