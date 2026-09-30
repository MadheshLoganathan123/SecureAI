import sys, time, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

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
    def forward(self, x):
        h1 = self.act1(self.ln1(self.fc1(x)))
        h2 = self.act2(self.ln2(self.fc2(h1)))
        s = self.skip(x)
        out = self.head(h2 + s)
        return out.squeeze(-1)

class LayerNormMLP(nn.Module):
    def __init__(self, in_features=41, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def local_train(model, X, y, epochs=1, lr=0.005, batch_size=256, opt_type="adamw", mu_prox=0.001, global_params=None):
    model = copy.deepcopy(model)
    if opt_type == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt_type == "sgd_mom":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
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

def analyze_client_updates():
    seed_everything(SEED)
    train_df, test_df, public_df, encoders, scaler = load_data(ROOT / 'data', ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    X_val = torch.tensor(holdout_df[FEATURE_COLS].values, dtype=torch.float32)
    y_val = torch.tensor(holdout_df['binary_label'].values, dtype=torch.float32)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    
    tensors = [df_to_tensors(df) for df in noniid_clients]
    sizes = [len(df) for df in noniid_clients]
    
    for arch_name, arch_class, lr, opt_type in [
        ("LayerNormMLP_AdamW", LayerNormMLP, 0.005, "adamw"),
        ("ResNetMLP_AdamW", ResNetMLP, 0.005, "adamw"),
        ("ResNetMLP_SGDMom", ResNetMLP, 0.02, "sgd_mom"),
    ]:
        print(f"\n=======================================================")
        print(f"Testing {arch_name} (opt={opt_type}, lr={lr})")
        print(f"=======================================================")
        
        for weighting_mode in ["sample_size", "sqrt_size", "balanced"]:
            seed_everything(SEED)
            model = arch_class()
            detected_malicious = []
            
            for r in range(8):
                local_models = []
                deltas = []
                global_flat = get_flat_params(model)
                global_params = [p.data.clone() for p in model.parameters()]
                
                for cid, (X, y0) in enumerate(tensors):
                    y = 1 - y0 if cid == 1 else y0
                    lm = local_train(model, X, y, epochs=1, lr=lr, opt_type=opt_type, mu_prox=0.001, global_params=global_params)
                    delta = get_flat_params(lm) - global_flat
                    if cid == 1:
                        delta = 15.0 * delta
                        set_flat_params(lm, global_flat + delta)
                    local_models.append(lm)
                    deltas.append(delta)
                
                # Defense: Norm clipping + Anomaly Detection
                delta_stack = torch.stack(deltas)
                norms = delta_stack.norm(dim=1).clamp_min(1e-12)
                clip_norm = 1.0
                clipped = delta_stack * torch.minimum(torch.ones_like(norms), clip_norm / norms)[:, None]
                
                # Peer distance scoring
                pairwise = torch.cdist(clipped, clipped, p=2)
                peer_scores = pairwise.sum(dim=1)
                
                # Cosine similarity scoring
                unit_clipped = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(1e-12)
                cos_sim = torch.mm(unit_clipped, unit_clipped.t())
                mean_cos = (cos_sim.sum(dim=1) - 1.0) / (len(tensors) - 1)
                
                # Client with highest peer distance (or lowest mean cosine)
                outlier_idx = int(torch.argmax(peer_scores).item())
                detected_malicious.append(outlier_idx)
                
                # Keep 4 clients
                keep = [i for i in range(len(tensors)) if i != outlier_idx]
                
                if weighting_mode == "sample_size":
                    kw = np.asarray(sizes, dtype=float)[keep]
                elif weighting_mode == "sqrt_size":
                    kw = np.sqrt(np.asarray(sizes, dtype=float)[keep])
                elif weighting_mode == "balanced":
                    # effective balanced weight
                    kw = np.ones(len(keep), dtype=float)
                kw = kw / kw.sum()
                
                avg_delta = (clipped[keep] * torch.tensor(kw, dtype=torch.float32)[:, None]).sum(0)
                set_flat_params(model, global_flat + avg_delta)
            
            # Validation threshold search
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_val)).cpu().numpy()
                test_probs = torch.sigmoid(model(X_test)).cpu().numpy()
            y_val_np = y_val.cpu().numpy().astype(int)
            y_test_np = y_test.cpu().numpy().astype(int)
            
            best_t = 0.5
            best_val_f1 = f1_score(y_val_np, (val_probs >= 0.5).astype(int), zero_division=0)
            for t in np.arange(0.2, 0.8, 0.02):
                vf1 = f1_score(y_val_np, (val_probs >= t).astype(int), zero_division=0)
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    best_t = round(float(t), 2)
            
            def eval_at(t):
                preds = (test_probs >= t).astype(int)
                return {
                    "f1": round(float(f1_score(y_test_np, preds, zero_division=0)), 4),
                    "acc": round(float(accuracy_score(y_test_np, preds)), 4),
                    "prec": round(float(precision_score(y_test_np, preds, zero_division=0)), 4),
                    "rec": round(float(recall_score(y_test_np, preds, zero_division=0)), 4)
                }
            
            res_default = eval_at(0.5)
            res_opt = eval_at(best_t)
            all_detected_correct = (detected_malicious == [1] * 8)
            print(f"Weighting: {weighting_mode:<12} | Detected Client 1: {all_detected_correct} | Default F1: {res_default['f1']} (Acc: {res_default['acc']}) | Opt F1 (t={best_t}): {res_opt['f1']} (Acc: {res_opt['acc']}, Prec: {res_opt['prec']}, Rec: {res_opt['rec']})")

if __name__ == '__main__':
    analyze_client_updates()
