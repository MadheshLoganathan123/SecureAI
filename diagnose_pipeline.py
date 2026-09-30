import sys, time, json, copy, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef, balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import (
    load_data, make_partitions, WeakMLP, df_to_tensors,
    get_flat_params, set_flat_params, local_train,
    aggregate_fedavg, aggregate_trimmed_clip,
    seed_everything, SEED, FEATURE_COLS
)

def evaluate_probs(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
    yt = y.cpu().numpy().astype(int)
    return yt, probs

def get_metrics_at_thresh(y_true, probs, thresh=0.5):
    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "thresh": round(float(thresh), 3),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def main():
    seed_everything(SEED)
    data_dir = ROOT / 'data'
    train_df, test_df, public_df, encoders, scaler = load_data(data_dir, ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    # Let's inspect the holdout set from train_df (the 15000 rows that were reserved!)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    X_val = torch.tensor(holdout_df[FEATURE_COLS].values, dtype=torch.float32)
    y_val = torch.tensor(holdout_df['binary_label'].values, dtype=torch.float32)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    
    print("Holdout (Validation) set shape:", holdout_df.shape, "Attack rate:", holdout_df['binary_label'].mean())
    print("Test set shape:", test_df.shape, "Attack rate:", test_df['binary_label'].mean())
    
    # Run previous defended model to analyze threshold sensitivity
    tensors = [df_to_tensors(df) for df in noniid_clients]
    seed_everything(SEED)
    defended_model = WeakMLP()
    for r in range(8):
        local_models, sizes = [], []
        global_flat = get_flat_params(defended_model)
        for cid, (X, y0) in enumerate(tensors):
            y = 1 - y0 if cid == 1 else y0
            lm = local_train(defended_model, X, y)
            delta = get_flat_params(lm) - global_flat
            if cid == 1:
                set_flat_params(lm, global_flat + 15.0 * delta)
            local_models.append(lm)
            sizes.append(len(X))
        defended_model = aggregate_trimmed_clip(local_models, sizes, defended_model, clip_norm=1.0, trim=1)
    
    y_val_np, val_probs = evaluate_probs(defended_model, X_val, y_val)
    y_test_np, test_probs = evaluate_probs(defended_model, X_test, y_test)
    
    print("\n--- Decision Threshold Analysis on Previous Defended Model ---")
    best_val_f1 = 0.0
    best_val_thresh = 0.5
    thresh_results = []
    for t in np.arange(0.1, 0.9, 0.05):
        vm = get_metrics_at_thresh(y_val_np, val_probs, t)
        tm = get_metrics_at_thresh(y_test_np, test_probs, t)
        thresh_results.append({"thresh": round(t, 2), "val_f1": vm['f1'], "test_f1": tm['f1'], "test_prec": tm['precision'], "test_rec": tm['recall'], "test_acc": tm['accuracy']})
        if vm['f1'] > best_val_f1:
            best_val_f1 = vm['f1']
            best_val_thresh = round(float(t), 2)
    
    print(pd.DataFrame(thresh_results).to_string())
    print(f"\nBest Validation Threshold: {best_val_thresh} with Val F1: {best_val_f1}")
    tm_best = get_metrics_at_thresh(y_test_np, test_probs, best_val_thresh)
    print(f"Test performance at best val threshold ({best_val_thresh}):", tm_best)
    tm_default = get_metrics_at_thresh(y_test_np, test_probs, 0.5)
    print(f"Test performance at default 0.5 threshold:", tm_default)

if __name__ == '__main__':
    main()
