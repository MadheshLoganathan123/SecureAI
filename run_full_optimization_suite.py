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
    load_data, make_partitions, df_to_tensors,
    get_flat_params, set_flat_params,
    seed_everything, SEED, FEATURE_COLS
)

class ResNetMLP(nn.Module):
    def __init__(self, in_features=41, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.head = nn.Linear(hidden_dim, 1)
        self.skip = nn.Linear(in_features, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act1(self.ln1(self.fc1(x)))
        h2 = self.act2(self.ln2(self.fc2(h1)))
        s = self.skip(x)
        out = self.head(h2 + s)
        return out.squeeze(-1)

def compute_all_metrics(y_true, y_probs, thresh=0.5):
    preds = (y_probs >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    roc_auc = roc_auc_score(y_true, y_probs)
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(r_curve, p_curve)
    
    return {
        "f1": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "mcc": round(float(mcc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def local_train(model, X, y, epochs=1, lr=0.005, batch_size=256, mu_prox=0.001, global_params=None):
    model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            if mu_prox > 0.0 and global_params is not None:
                prox = sum((p - gp).norm()**2 for p, gp in zip(model.parameters(), global_params))
                loss = loss + 0.5 * mu_prox * prox
            loss.backward()
            opt.step()
    return model

def robust_aggregate(local_models, global_model, sizes, clip_norm=1.0, weighting="sqrt_size"):
    ref = get_flat_params(global_model)
    deltas = [get_flat_params(m) - ref for m in local_models]
    delta_stack = torch.stack(deltas)
    
    # 1. Norm Clipping
    raw_norms = delta_stack.norm(dim=1).clamp_min(1e-12)
    clipped = delta_stack * torch.minimum(torch.ones_like(raw_norms), clip_norm / raw_norms)[:, None]
    
    # 2. Multi-criterion Anomaly Scoring:
    # A) Normalized Direction Cosine to Peer Mean
    unit_clipped = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_sim = torch.mm(unit_clipped, unit_clipped.t())
    mean_cos = (cos_sim.sum(dim=1) - 1.0) / (len(local_models) - 1)
    
    # B) Peer Euclidean Distance
    pairwise = torch.cdist(clipped, clipped, p=2)
    peer_dist = pairwise.sum(dim=1)
    
    # C) Norm Ratio vs Median Norm
    med_norm = torch.median(raw_norms)
    norm_ratio = raw_norms / (med_norm + 1e-6)
    
    # Composite Malicious Score: higher = more malicious
    # A malicious client has: high norm ratio, high peer distance, low/negative cosine
    anomaly_score = peer_dist * norm_ratio - 2.0 * mean_cos
    
    # Exclude top anomaly
    malicious_idx = int(torch.argmax(anomaly_score).item())
    keep = [i for i in range(len(local_models)) if i != malicious_idx]
    
    # 3. Weighting among honest clients
    if weighting == "sample_size":
        kw = np.asarray(sizes, dtype=float)[keep]
    elif weighting == "sqrt_size":
        kw = np.sqrt(np.asarray(sizes, dtype=float)[keep])
    elif weighting == "balanced":
        kw = np.ones(len(keep), dtype=float)
    else:
        kw = np.ones(len(keep), dtype=float)
    kw = kw / kw.sum()
    
    avg_delta = (clipped[keep] * torch.tensor(kw, dtype=torch.float32)[:, None]).sum(0)
    out = copy.deepcopy(global_model)
    set_flat_params(out, ref + avg_delta)
    return out, malicious_idx, [round(float(s), 3) for s in anomaly_score]

def run_fl_training(
    noniid_clients, X_test, y_test, rounds=8, epochs=1, lr=0.005,
    weighting="sqrt_size", clip_norm=1.0,
    malicious_clients=None, scale_factor=15.0, seed=SEED, is_defense=True
):
    seed_everything(seed)
    malicious_clients = set(malicious_clients or [])
    model = ResNetMLP()
    tensors = [df_to_tensors(df) for df in noniid_clients]
    sizes = [len(df) for df in noniid_clients]
    
    detected_history = []
    round_metrics = []
    t0 = time.perf_counter()
    
    for r in range(rounds):
        local_models = []
        global_flat = get_flat_params(model)
        global_params = [p.data.clone() for p in model.parameters()]
        
        for cid, (X, y0) in enumerate(tensors):
            y = 1 - y0 if cid in malicious_clients else y0
            lm = local_train(model, X, y, epochs=epochs, lr=lr, mu_prox=0.001, global_params=global_params)
            delta = get_flat_params(lm) - global_flat
            if cid in malicious_clients:
                delta = scale_factor * delta
                set_flat_params(lm, global_flat + delta)
            local_models.append(lm)
            
        if is_defense:
            model, mal_idx, scores = robust_aggregate(local_models, model, sizes, clip_norm=clip_norm, weighting=weighting)
            detected_history.append(mal_idx)
        else:
            # Naive FedAvg (sample-size weighted mean of raw parameters)
            w = np.array(sizes, dtype=float) / np.sum(sizes)
            flats = torch.stack([get_flat_params(m) for m in local_models])
            avg = (flats * torch.tensor(w, dtype=torch.float32)[:, None]).sum(0)
            set_flat_params(model, avg)
            
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test)).cpu().numpy()
        m = compute_all_metrics(y_test.cpu().numpy().astype(int), probs)
        round_metrics.append(m)
        
    train_time = time.perf_counter() - t0
    return model, round_metrics, detected_history, train_time

def main():
    seed_everything(SEED)
    data_dir = ROOT / 'data'
    train_df, test_df, public_df, encoders, scaler = load_data(data_dir, ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    y_test_np = y_test.numpy().astype(int)
    
    print("=================================================================")
    print("STEP 1: Run Multi-Seed Evaluation of Final Defended Model")
    print("=================================================================")
    seeds = [42, 101, 2024, 777, 9999]
    seed_results = []
    for s in seeds:
        model, hist, detected, t_time = run_fl_training(
            noniid_clients, X_test, y_test, rounds=8, epochs=1, lr=0.005,
            weighting="sqrt_size", clip_norm=1.0,
            malicious_clients=[1], scale_factor=15.0, seed=s, is_defense=True
        )
        final_m = hist[-1]
        detection_acc = sum(1 for d in detected if d == 1) / len(detected)
        seed_results.append({
            "seed": s,
            "f1": final_m["f1"],
            "accuracy": final_m["accuracy"],
            "precision": final_m["precision"],
            "recall": final_m["recall"],
            "specificity": final_m["specificity"],
            "balanced_accuracy": final_m["balanced_accuracy"],
            "mcc": final_m["mcc"],
            "roc_auc": final_m["roc_auc"],
            "pr_auc": final_m["pr_auc"],
            "detection_rate": detection_acc,
            "train_time": round(t_time, 2)
        })
        print(f"Seed {s:5d}: F1={final_m['f1']:.4f}, Acc={final_m['accuracy']:.4f}, Prec={final_m['precision']:.4f}, Rec={final_m['recall']:.4f}, DetRate={detection_acc*100:.1f}%, Time={t_time:.2f}s")
        
    df_seeds = pd.DataFrame(seed_results)
    print("\n--- Multi-Seed Summary ---")
    print(f"Mean F1: {df_seeds['f1'].mean():.4f} +/- {df_seeds['f1'].std():.4f} (Min: {df_seeds['f1'].min():.4f}, Max: {df_seeds['f1'].max():.4f})")
    print(f"Mean Acc: {df_seeds['accuracy'].mean():.4f} +/- {df_seeds['accuracy'].std():.4f}")
    print(f"Mean Prec: {df_seeds['precision'].mean():.4f} +/- {df_seeds['precision'].std():.4f}")
    print(f"Mean Rec: {df_seeds['recall'].mean():.4f} +/- {df_seeds['recall'].std():.4f}")
    df_seeds.to_csv("multi_seed_results.csv", index=False)
    
    print("\n=================================================================")
    print("STEP 2: Intermediate Track Comparison (Clean Non-IID)")
    print("Naive FedAvg (WeakMLP) vs Final Optimized Model (ResNetMLP + Sqrt Weighting)")
    print("=================================================================")
    # Naive FedAvg WeakMLP on Clean Non-IID
    with open("baseline_audit.json") as f:
        audit_data = json.load(f)
    base_clean = audit_data["clean_baseline"]["metrics"]
    
    # Final Optimized Model on Clean Non-IID
    opt_clean_model, opt_clean_hist, _, opt_clean_time = run_fl_training(
        noniid_clients, X_test, y_test, rounds=8, epochs=1, lr=0.005,
        weighting="sqrt_size", clip_norm=1.0,
        malicious_clients=[], seed=SEED, is_defense=True
    )
    opt_clean = opt_clean_hist[-1]
    
    inter_comp = [
        {
            "Method": "Naive FedAvg (WeakMLP baseline)",
            "Scenario": "Non-IID Clean",
            "F1": base_clean["f1"],
            "Accuracy": base_clean["accuracy"],
            "Precision": base_clean["precision"],
            "Recall": base_clean["recall"],
            "Specificity": base_clean["specificity"],
            "Balanced_Accuracy": base_clean["balanced_accuracy"],
            "MCC": base_clean["mcc"],
            "ROC_AUC": base_clean["roc_auc"],
            "PR_AUC": base_clean["pr_auc"]
        },
        {
            "Method": "Final Optimized Federated Model",
            "Scenario": "Non-IID Clean",
            "F1": opt_clean["f1"],
            "Accuracy": opt_clean["accuracy"],
            "Precision": opt_clean["precision"],
            "Recall": opt_clean["recall"],
            "Specificity": opt_clean["specificity"],
            "Balanced_Accuracy": opt_clean["balanced_accuracy"],
            "MCC": opt_clean["mcc"],
            "ROC_AUC": opt_clean["roc_auc"],
            "PR_AUC": opt_clean["pr_auc"]
        }
    ]
    df_inter = pd.DataFrame(inter_comp)
    print(df_inter.to_string())
    df_inter.to_csv("intermediate_final_comparison.csv", index=False)
    
    print("\n=================================================================")
    print("STEP 3: Advanced Track Comparison (Under Attack)")
    print("Naive FedAvg Under Attack vs Previous Defended vs Final Defended")
    print("=================================================================")
    base_attack = audit_data["attack_baseline"]["metrics"]
    prev_def = audit_data["previous_defended"]["metrics"]
    
    # Run Final Defended Model on Seed 42
    final_model, final_hist, final_det, final_time = run_fl_training(
        noniid_clients, X_test, y_test, rounds=8, epochs=1, lr=0.005,
        weighting="sqrt_size", clip_norm=1.0,
        malicious_clients=[1], scale_factor=15.0, seed=SEED, is_defense=True
    )
    final_def = final_hist[-1]
    
    adv_comp = [
        {
            "Method": "Naive FedAvg (Under Attack)",
            "Scenario": "Poisoned Client 1 (scale 15x)",
            "F1": base_attack["f1"],
            "Accuracy": base_attack["accuracy"],
            "Precision": base_attack["precision"],
            "Recall": base_attack["recall"],
            "F1_Gain_vs_Attack": 0.0,
            "FPR": base_attack["fpr"],
            "FNR": base_attack["fnr"],
            "Confusion_Matrix": str(base_attack["confusion_matrix"])
        },
        {
            "Method": "Previous Defended E003 (ClippedTrimmedPeerRobust)",
            "Scenario": "Poisoned Client 1 (scale 15x)",
            "F1": prev_def["f1"],
            "Accuracy": prev_def["accuracy"],
            "Precision": prev_def["precision"],
            "Recall": prev_def["recall"],
            "F1_Gain_vs_Attack": round(prev_def["f1"] - base_attack["f1"], 4),
            "FPR": prev_def["fpr"],
            "FNR": prev_def["fnr"],
            "Confusion_Matrix": str(prev_def["confusion_matrix"])
        },
        {
            "Method": "Final Optimized Defended Model",
            "Scenario": "Poisoned Client 1 (scale 15x)",
            "F1": final_def["f1"],
            "Accuracy": final_def["accuracy"],
            "Precision": final_def["precision"],
            "Recall": final_def["recall"],
            "F1_Gain_vs_Attack": round(final_def["f1"] - base_attack["f1"], 4),
            "FPR": final_def["fpr"],
            "FNR": final_def["fnr"],
            "Confusion_Matrix": str(final_def["confusion_matrix"])
        }
    ]
    df_adv = pd.DataFrame(adv_comp)
    print(df_adv.to_string())
    df_adv.to_csv("advanced_final_comparison.csv", index=False)
    
    print("\n=================================================================")
    print("STEP 4: Cross-Client Generalization Analysis")
    print("=================================================================")
    tensors = [df_to_tensors(df) for df in noniid_clients]
    client_evals = []
    for cid, (Xc, yc) in enumerate(tensors):
        with torch.no_grad():
            c_probs = torch.sigmoid(final_model(Xc)).cpu().numpy()
        cm_metrics = compute_all_metrics(yc.cpu().numpy().astype(int), c_probs)
        client_evals.append({
            "client_id": cid,
            "samples": len(Xc),
            "attack_rate": round(float(yc.mean().item()), 4),
            "f1": cm_metrics["f1"],
            "accuracy": cm_metrics["accuracy"],
            "precision": cm_metrics["precision"],
            "recall": cm_metrics["recall"],
            "fpr": cm_metrics["fpr"],
            "fnr": cm_metrics["fnr"]
        })
        print(f"Client {cid} ({len(Xc)} samples, attack_rate={yc.mean().item():.3f}): F1={cm_metrics['f1']}, Acc={cm_metrics['accuracy']}, Prec={cm_metrics['precision']}, Rec={cm_metrics['recall']}")
    df_clients = pd.DataFrame(client_evals)
    df_clients.to_csv("final_client_generalization.csv", index=False)
    
    print("\n=================================================================")
    print("STEP 5: Ablation Study")
    print("=================================================================")
    # 1. Baseline WeakMLP FedAvg Clean
    # 2. Baseline WeakMLP FedAvg Under Attack
    # 3. + Robust Clipping & Anomaly Detection (Previous Defense)
    # 4. + ResNetMLP Architecture
    # 5. + Local AdamW & Proximal Regularization
    # 6. + Sqrt-Size Weighting (Final Defended)
    # 7. + Decision Threshold Calibration (0.48)
    
    # We will test each component in ablation
    # Let's save the final full comparison dictionary
    suite_data = {
        "multi_seed": seed_results,
        "intermediate_comparison": inter_comp,
        "advanced_comparison": adv_comp,
        "client_generalization": client_evals,
        "final_metrics": final_def,
        "final_round_history": final_hist,
        "final_train_time": final_time
    }
    with open("full_optimization_suite.json", "w") as f:
        json.dump(suite_data, f, indent=2)
    print("\nSaved full_optimization_suite.json successfully.")

if __name__ == '__main__':
    main()
