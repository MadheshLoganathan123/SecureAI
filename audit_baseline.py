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
    seed_everything, SEED, FEATURE_COLS, CAT_COLS, COLS
)

def comprehensive_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(r_curve, p_curve)
    
    return {
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "mcc": round(float(mcc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4)
    }

def audit():
    seed_everything(SEED)
    data_dir = ROOT / 'data'
    train_df, test_df, public_df, encoders, scaler = load_data(data_dir, ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    y_test_np = y_test.numpy().astype(int)
    
    print("=== DATA AUDIT ===")
    print(f"Total train: {len(train_df)}, Test: {len(test_df)}, Public: {len(public_df)}")
    print(f"Train pool: {len(train_pool)}, Holdout: {len(holdout_idx)}")
    print(f"Overall test label balance: Normal={(y_test_np==0).sum()}, Attack={(y_test_np==1).sum()} (Attack rate: {y_test_np.mean():.4f})")
    print(f"Overall train pool label balance: Attack rate: {train_pool['binary_label'].mean():.4f}")
    
    print("\n=== CLIENT PARTITIONS ===")
    for cid, cdf in enumerate(noniid_clients):
        y_c = cdf['binary_label'].values
        print(f"Client {cid}: samples={len(cdf)}, normal={(y_c==0).sum()}, attack={(y_c==1).sum()}, attack_rate={y_c.mean():.4f}")
    
    # 1. Run Baseline Clean FedAvg
    print("\n--- Running Baseline Clean FedAvg (8 rounds) ---")
    seed_everything(SEED)
    clean_model = WeakMLP()
    tensors = [df_to_tensors(df) for df in noniid_clients]
    clean_round_metrics = []
    
    t0 = time.perf_counter()
    for r in range(8):
        local_models, sizes = [], []
        global_flat = get_flat_params(clean_model)
        for cid, (X, y) in enumerate(tensors):
            lm = local_train(clean_model, X, y)
            local_models.append(lm)
            sizes.append(len(X))
        clean_model = aggregate_fedavg(local_models, sizes, clean_model)
        with torch.no_grad():
            probs = torch.sigmoid(clean_model(X_test)).numpy()
        m = comprehensive_metrics(y_test_np, probs)
        clean_round_metrics.append(m)
        print(f"Round {r+1}: F1={m['f1']}, Acc={m['accuracy']}, Prec={m['precision']}, Rec={m['recall']}")
    clean_train_time = time.perf_counter() - t0
    
    # 2. Run Baseline Attacked FedAvg (Malicious client 1, label flip + 15x scale)
    print("\n--- Running Baseline Attacked FedAvg (Client 1 poisoned, 8 rounds) ---")
    seed_everything(SEED)
    attack_model = WeakMLP()
    attack_round_metrics = []
    t0 = time.perf_counter()
    for r in range(8):
        local_models, sizes = [], []
        global_flat = get_flat_params(attack_model)
        for cid, (X, y0) in enumerate(tensors):
            y = 1 - y0 if cid == 1 else y0
            lm = local_train(attack_model, X, y)
            delta = get_flat_params(lm) - global_flat
            if cid == 1:
                set_flat_params(lm, global_flat + 15.0 * delta)
            local_models.append(lm)
            sizes.append(len(X))
        attack_model = aggregate_fedavg(local_models, sizes, attack_model)
        with torch.no_grad():
            probs = torch.sigmoid(attack_model(X_test)).numpy()
        m = comprehensive_metrics(y_test_np, probs)
        attack_round_metrics.append(m)
        print(f"Round {r+1}: F1={m['f1']}, Acc={m['accuracy']}, Prec={m['precision']}, Rec={m['recall']}")
    attack_train_time = time.perf_counter() - t0
    
    # 3. Run Previous Defended Model (E003: ClippedTrimmedPeerRobust)
    print("\n--- Running Previous Defended Model E003 (8 rounds) ---")
    seed_everything(SEED)
    defended_model = WeakMLP()
    defended_round_metrics = []
    t0 = time.perf_counter()
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
        with torch.no_grad():
            probs = torch.sigmoid(defended_model(X_test)).numpy()
        m = comprehensive_metrics(y_test_np, probs)
        defended_round_metrics.append(m)
        print(f"Round {r+1}: F1={m['f1']}, Acc={m['accuracy']}, Prec={m['precision']}, Rec={m['recall']}")
    defended_train_time = time.perf_counter() - t0
    
    # Inference time measurement
    def measure_inference(model, X):
        model.eval()
        # warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(X)
        t_start = time.perf_counter()
        iters = 50
        with torch.no_grad():
            for _ in range(iters):
                _ = model(X)
        total_t = time.perf_counter() - t_start
        return (total_t / iters) * 1000.0 # ms per full test set pass
    
    inf_time_clean = measure_inference(clean_model, X_test)
    inf_time_attack = measure_inference(attack_model, X_test)
    inf_time_defended = measure_inference(defended_model, X_test)
    
    # Model size and param count
    param_count = sum(p.numel() for p in defended_model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in defended_model.parameters())
    client_update_size_bytes = param_bytes # delta has same size as model params
    
    # Per-client metrics on each client's local data
    print("\n=== PER-CLIENT METRICS ON DEFENDED MODEL (LOCAL CLIENT DATA) ===")
    per_client_defended = []
    for cid, (X, y) in enumerate(tensors):
        with torch.no_grad():
            probs = torch.sigmoid(defended_model(X)).numpy()
        pcm = comprehensive_metrics(y.numpy().astype(int), probs)
        per_client_defended.append(pcm)
        print(f"Client {cid} ({len(X)} samples): F1={pcm['f1']}, Acc={pcm['accuracy']}, Prec={pcm['precision']}, Rec={pcm['recall']}, FPR={pcm['fpr']}, FNR={pcm['fnr']}")
    
    audit_results = {
        "clean_baseline": {
            "metrics": clean_round_metrics[-1],
            "train_time": clean_train_time,
            "inf_time_ms": inf_time_clean,
            "round_metrics": clean_round_metrics
        },
        "attack_baseline": {
            "metrics": attack_round_metrics[-1],
            "train_time": attack_train_time,
            "inf_time_ms": inf_time_attack,
            "round_metrics": attack_round_metrics
        },
        "previous_defended": {
            "metrics": defended_round_metrics[-1],
            "train_time": defended_train_time,
            "inf_time_ms": inf_time_defended,
            "round_metrics": defended_round_metrics,
            "per_client": per_client_defended
        },
        "model_stats": {
            "param_count": param_count,
            "param_bytes": param_bytes,
            "client_update_bytes": client_update_size_bytes
        }
    }
    
    with open("baseline_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)
    print("\nSaved baseline_audit.json successfully.")

if __name__ == '__main__':
    audit()
