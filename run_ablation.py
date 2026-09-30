import sys, time, json, copy, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import (
    load_data, make_partitions, df_to_tensors,
    get_flat_params, set_flat_params,
    seed_everything, SEED, FEATURE_COLS
)
from run_full_optimization_suite import ResNetMLP, local_train, robust_aggregate

class WeakMLP(nn.Module):
    def __init__(self, in_features=41, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def run_ablation():
    seed_everything(SEED)
    train_df, test_df, public_df, encoders, scaler = load_data(ROOT / 'data', ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    X_val = torch.tensor(holdout_df[FEATURE_COLS].values, dtype=torch.float32)
    y_val = torch.tensor(holdout_df['binary_label'].values, dtype=torch.float32)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    y_test_np = y_test.numpy().astype(int)
    
    tensors = [df_to_tensors(df) for df in noniid_clients]
    sizes = [len(df) for df in noniid_clients]
    
    ablation_stages = [
        # (name, arch_fn, opt_type, lr, mu_prox, defense, weighting)
        ("1. Baseline FedAvg (WeakMLP)", lambda: WeakMLP(), "sgd", 0.05, 0.0, "none", "sample_size"),
        ("2. + Peer Trimming Defense (E003 baseline)", lambda: WeakMLP(), "sgd", 0.05, 0.0, "trimmed_clip", "sample_size"),
        ("3. + ResNetMLP Architecture", lambda: ResNetMLP(), "sgd", 0.05, 0.0, "trimmed_clip", "sample_size"),
        ("4. + AdamW & FedProx Local Training", lambda: ResNetMLP(), "adamw", 0.005, 0.001, "trimmed_clip", "sample_size"),
        ("5. + Multi-criterion Anomaly Defense & Sqrt Weighting", lambda: ResNetMLP(), "adamw", 0.005, 0.001, "robust_multi", "sqrt_size"),
    ]
    
    ablation_rows = []
    
    for name, arch_fn, opt_type, lr, mu_prox, defense_type, weighting_type in ablation_stages:
        seed_everything(SEED)
        model = arch_fn()
        t0 = time.perf_counter()
        
        for r in range(8):
            local_models = []
            deltas = []
            global_flat = get_flat_params(model)
            global_params = [p.data.clone() for p in model.parameters()]
            
            for cid, (X, y0) in enumerate(tensors):
                y = 1 - y0 if cid == 1 else y0
                if opt_type == "adamw":
                    lm = local_train(model, X, y, epochs=1, lr=lr, mu_prox=mu_prox, global_params=global_params)
                else:
                    # vanilla SGD
                    lm = copy.deepcopy(model)
                    opt = torch.optim.SGD(lm.parameters(), lr=lr)
                    loss_fn = nn.BCEWithLogitsLoss()
                    for i in range(0, len(X), 256):
                        idx = torch.randperm(len(X))[i:i+256]
                        opt.zero_grad()
                        loss = loss_fn(lm(X[idx]), y[idx])
                        loss.backward(); opt.step()
                        
                delta = get_flat_params(lm) - global_flat
                if cid == 1:
                    delta = 15.0 * delta
                    set_flat_params(lm, global_flat + delta)
                local_models.append(lm)
                deltas.append(delta)
                
            if defense_type == "none":
                w = np.array(sizes, dtype=float) / np.sum(sizes)
                flats = torch.stack([get_flat_params(m) for m in local_models])
                avg = (flats * torch.tensor(w, dtype=torch.float32)[:, None]).sum(0)
                set_flat_params(model, avg)
            elif defense_type == "trimmed_clip":
                delta_stack = torch.stack(deltas)
                norms = delta_stack.norm(dim=1).clamp_min(1e-12)
                clipped = delta_stack * torch.minimum(torch.ones_like(norms), 1.0 / norms)[:, None]
                pairwise = torch.cdist(clipped, clipped, p=2)
                scores = pairwise.sum(dim=1)
                keep = torch.argsort(scores)[:4]
                kw = np.asarray(sizes, dtype=float)[keep.numpy()]
                kw = kw / kw.sum()
                avg = (clipped[keep] * torch.tensor(kw, dtype=torch.float32)[:, None]).sum(0)
                set_flat_params(model, global_flat + avg)
            elif defense_type == "robust_multi":
                model, _, _ = robust_aggregate(local_models, model, sizes, clip_norm=1.0, weighting=weighting_type)
                
        runtime = round(time.perf_counter() - t0, 3)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
        f1 = round(float(f1_score(y_test_np, preds, zero_division=0)), 4)
        acc = round(float(accuracy_score(y_test_np, preds)), 4)
        prec = round(float(precision_score(y_test_np, preds, zero_division=0)), 4)
        rec = round(float(recall_score(y_test_np, preds, zero_division=0)), 4)
        
        ablation_rows.append({
            "Stage": name,
            "F1": f1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Runtime_Seconds": runtime
        })
        print(f"Done: {name} -> F1={f1}, Acc={acc}, Prec={prec}, Rec={rec}, Time={runtime}s")
        
    df_abl = pd.DataFrame(ablation_rows)
    print("\n=== ABLATION STUDY RESULTS ===")
    print(df_abl.to_string())
    df_abl.to_csv("ablation_study.csv", index=False)

if __name__ == '__main__':
    run_ablation()
