import sys, time, json, copy, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import (
    load_data, make_partitions, df_to_tensors,
    get_flat_params, set_flat_params,
    seed_everything, SEED, FEATURE_COLS
)

# Architectures to evaluate
class M1_WeakMLP(nn.Module):
    def __init__(self, in_features=41, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class M2_MediumMLP(nn.Module):
    def __init__(self, in_features=41, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class M3_TwoLayerMLP(nn.Module):
    def __init__(self, in_features=41, h1=32, h2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class M4_LayerNormMLP(nn.Module):
    def __init__(self, in_features=41, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class M5_ResMLP(nn.Module):
    def __init__(self, in_features=41, hidden=32):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        self.out_proj = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.in_proj(x)
        h = h + self.block(h)
        return self.out_proj(h).squeeze(-1)

class M6_WiderMLP(nn.Module):
    def __init__(self, in_features=41, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def evaluate_f1(model, X, y, thresh=0.5):
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) >= thresh).cpu().numpy().astype(int)
    yt = y.cpu().numpy().astype(int)
    return {
        "f1": round(float(f1_score(yt, preds, zero_division=0)), 4),
        "acc": round(float(accuracy_score(yt, preds)), 4),
        "prec": round(float(precision_score(yt, preds, zero_division=0)), 4),
        "rec": round(float(recall_score(yt, preds, zero_division=0)), 4)
    }

def local_train_advanced(model, X, y, epochs=1, lr=0.05, batch_size=256,
                         opt_name="sgd", momentum=0.0, weight_decay=0.0,
                         mu_prox=0.0, global_params=None):
    model = copy.deepcopy(model)
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(opt_name)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            if mu_prox > 0.0 and global_params is not None:
                prox_term = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox_term = prox_term + (p - gp).norm()**2
                loss = loss + 0.5 * mu_prox * prox_term
            loss.backward()
            opt.step()
    return model

def run_fl_experiment(
    arch_fn, noniid_clients, X_test, y_test, X_val, y_val,
    rounds=8, epochs=1, lr=0.05, opt_name="sgd", momentum=0.0, weight_decay=0.0, mu_prox=0.0,
    weighting="sample_size", # "sample_size", "sqrt_size", "uniform", "balanced"
    defense="robust", # "none", "median", "robust", "trimmed_norm", "cosine_trust"
    clip_norm=1.0, trim=1,
    malicious_clients=None, scale_factor=15.0, seed=SEED
):
    seed_everything(seed)
    malicious_clients = set(malicious_clients or [])
    model = arch_fn()
    tensors = [df_to_tensors(df) for df in noniid_clients]
    sizes = [len(df) for df in noniid_clients]
    
    t0 = time.perf_counter()
    for r in range(rounds):
        local_models, deltas = [], []
        global_flat = get_flat_params(model)
        global_param_list = [p.data.clone() for p in model.parameters()]
        
        for cid, (X, y0) in enumerate(tensors):
            y = 1 - y0 if cid in malicious_clients else y0
            lm = local_train_advanced(
                model, X, y, epochs=epochs, lr=lr,
                opt_name=opt_name, momentum=momentum, weight_decay=weight_decay,
                mu_prox=mu_prox, global_params=global_param_list
            )
            delta = get_flat_params(lm) - global_flat
            if cid in malicious_clients:
                delta = scale_factor * delta
                set_flat_params(lm, global_flat + delta)
            local_models.append(lm)
            deltas.append(delta)
        
        ref = get_flat_params(model)
        delta_stack = torch.stack(deltas) # (N, d)
        
        # Defense logic
        if defense == "none":
            # standard FedAvg
            if weighting == "sample_size":
                w = np.array(sizes, dtype=float) / np.sum(sizes)
            elif weighting == "sqrt_size":
                s = np.sqrt(sizes); w = s / np.sum(s)
            elif weighting == "uniform":
                w = np.ones(len(sizes)) / len(sizes)
            avg_delta = (delta_stack * torch.tensor(w, dtype=torch.float32)[:, None]).sum(0)
            set_flat_params(model, ref + avg_delta)
            
        elif defense == "median":
            flats = torch.stack([get_flat_params(m) for m in local_models])
            med = flats.median(dim=0).values
            set_flat_params(model, med)
            
        elif defense == "robust": # baseline robust (clipped + peer dist trim + size weighted)
            norms = delta_stack.norm(dim=1).clamp_min(1e-12)
            clipped = delta_stack * torch.minimum(torch.ones_like(norms), clip_norm / norms)[:, None]
            pairwise = torch.cdist(clipped, clipped, p=2)
            scores = pairwise.sum(dim=1)
            keep = torch.argsort(scores)[:len(local_models)-trim] if len(local_models) > trim else torch.arange(len(local_models))
            
            if weighting == "sample_size":
                kw = np.asarray(sizes, dtype=float)[keep.numpy()]
            elif weighting == "sqrt_size":
                kw = np.sqrt(np.asarray(sizes, dtype=float)[keep.numpy()])
            elif weighting == "uniform":
                kw = np.ones(len(keep))
            kw = kw / kw.sum()
            avg = (clipped[keep] * torch.tensor(kw, dtype=torch.float32)[:, None]).sum(0)
            set_flat_params(model, ref + avg)
            
        elif defense == "cosine_trust":
            # Clip updates
            norms = delta_stack.norm(dim=1).clamp_min(1e-12)
            clipped = delta_stack * torch.minimum(torch.ones_like(norms), clip_norm / norms)[:, None]
            # Compute normalized cosines
            unit_clipped = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos_matrix = torch.mm(unit_clipped, unit_clipped.t())
            # Malicious client that points opposite has negative / low cosine to peers
            mean_cos = cos_matrix.sum(dim=1) - 1.0 # exclude self
            keep = torch.argsort(mean_cos, descending=True)[:len(local_models)-trim]
            if weighting == "sample_size":
                kw = np.asarray(sizes, dtype=float)[keep.numpy()]
            elif weighting == "sqrt_size":
                kw = np.sqrt(np.asarray(sizes, dtype=float)[keep.numpy()])
            elif weighting == "uniform":
                kw = np.ones(len(keep))
            kw = kw / kw.sum()
            avg = (clipped[keep] * torch.tensor(kw, dtype=torch.float32)[:, None]).sum(0)
            set_flat_params(model, ref + avg)
            
    train_time = time.perf_counter() - t0
    
    # Evaluate at default 0.5 threshold
    test_metrics = evaluate_f1(model, X_test, y_test, thresh=0.5)
    val_metrics = evaluate_f1(model, X_val, y_val, thresh=0.5)
    
    # Search threshold on validation set
    best_t, best_val_f1 = 0.5, val_metrics["f1"]
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_test)).cpu().numpy()
    y_val_np = y_val.cpu().numpy().astype(int)
    y_test_np = y_test.cpu().numpy().astype(int)
    
    for t in np.arange(0.2, 0.8, 0.05):
        vf1 = f1_score(y_val_np, (val_probs >= t).astype(int), zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_t = round(float(t), 2)
            
    opt_test_preds = (test_probs >= best_t).astype(int)
    opt_test_metrics = {
        "opt_thresh": best_t,
        "opt_f1": round(float(f1_score(y_test_np, opt_test_preds, zero_division=0)), 4),
        "opt_acc": round(float(accuracy_score(y_test_np, opt_test_preds)), 4),
        "opt_prec": round(float(precision_score(y_test_np, opt_test_preds, zero_division=0)), 4),
        "opt_rec": round(float(recall_score(y_test_np, opt_test_preds, zero_division=0)), 4)
    }
    
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "model": model,
        "param_count": param_count,
        "train_time": round(train_time, 3),
        "test_default": test_metrics,
        "val_default": val_metrics,
        "opt_test": opt_test_metrics
    }

def main():
    seed_everything(SEED)
    data_dir = ROOT / 'data'
    train_df, test_df, public_df, encoders, scaler = load_data(data_dir, ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    X_val = torch.tensor(holdout_df[FEATURE_COLS].values, dtype=torch.float32)
    y_val = torch.tensor(holdout_df['binary_label'].values, dtype=torch.float32)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    
    print("Testing Architecture Search...")
    archs = [
        ("M1_WeakMLP_8", lambda: M1_WeakMLP()),
        ("M2_MediumMLP_32", lambda: M2_MediumMLP()),
        ("M3_TwoLayer_32_16", lambda: M3_TwoLayerMLP()),
        ("M4_LayerNorm_32", lambda: M4_LayerNormMLP()),
        ("M5_ResMLP_32", lambda: M5_ResMLP()),
        ("M6_WiderMLP_64", lambda: M6_WiderMLP()),
    ]
    
    arch_results = []
    for name, afn in archs:
        res = run_fl_experiment(
            afn, noniid_clients, X_test, y_test, X_val, y_val,
            rounds=8, epochs=1, lr=0.05, defense="robust",
            malicious_clients=[1], scale_factor=15.0
        )
        arch_results.append({
            "arch": name,
            "params": res["param_count"],
            "train_time": res["train_time"],
            "f1_0.5": res["test_default"]["f1"],
            "acc_0.5": res["test_default"]["acc"],
            "prec_0.5": res["test_default"]["prec"],
            "rec_0.5": res["test_default"]["rec"],
            "opt_thresh": res["opt_test"]["opt_thresh"],
            "opt_f1": res["opt_test"]["opt_f1"],
            "opt_rec": res["opt_test"]["opt_rec"]
        })
        print(f"Arch {name} ({res['param_count']}p): Default F1={res['test_default']['f1']}, Opt F1={res['opt_test']['opt_f1']} (th={res['opt_test']['opt_thresh']})")
    
    print("\n--- Architecture Comparison Table ---")
    df_arch = pd.DataFrame(arch_results)
    print(df_arch.to_string())
    df_arch.to_csv("arch_search_results.csv", index=False)
    
    print("\nTesting Optimizer & Local Training Optimization (using M2_MediumMLP)...")
    opt_configs = [
        ("SGD_lr0.05_ep1", {"lr": 0.05, "epochs": 1, "opt_name": "sgd", "momentum": 0.0, "mu_prox": 0.0}),
        ("SGD_lr0.08_ep1", {"lr": 0.08, "epochs": 1, "opt_name": "sgd", "momentum": 0.0, "mu_prox": 0.0}),
        ("SGD_mom0.9_lr0.02_ep1", {"lr": 0.02, "epochs": 1, "opt_name": "sgd", "momentum": 0.9, "mu_prox": 0.0}),
        ("SGD_ep2_lr0.03", {"lr": 0.03, "epochs": 2, "opt_name": "sgd", "momentum": 0.0, "mu_prox": 0.0}),
        ("FedProx_mu0.01_lr0.05", {"lr": 0.05, "epochs": 1, "opt_name": "sgd", "momentum": 0.0, "mu_prox": 0.01}),
        ("FedProx_mu0.05_lr0.05", {"lr": 0.05, "epochs": 1, "opt_name": "sgd", "momentum": 0.0, "mu_prox": 0.05}),
        ("AdamW_lr0.005_ep1", {"lr": 0.005, "epochs": 1, "opt_name": "adamw", "momentum": 0.0, "mu_prox": 0.0}),
    ]
    
    opt_results = []
    for name, cfg in opt_configs:
        res = run_fl_experiment(
            lambda: M2_MediumMLP(), noniid_clients, X_test, y_test, X_val, y_val,
            rounds=8, epochs=cfg["epochs"], lr=cfg["lr"],
            opt_name=cfg["opt_name"], momentum=cfg["momentum"], mu_prox=cfg["mu_prox"],
            defense="robust", malicious_clients=[1], scale_factor=15.0
        )
        opt_results.append({
            "config": name,
            "train_time": res["train_time"],
            "f1_0.5": res["test_default"]["f1"],
            "acc_0.5": res["test_default"]["acc"],
            "prec_0.5": res["test_default"]["prec"],
            "rec_0.5": res["test_default"]["rec"],
            "opt_thresh": res["opt_test"]["opt_thresh"],
            "opt_f1": res["opt_test"]["opt_f1"]
        })
        print(f"Opt {name}: Default F1={res['test_default']['f1']}, Opt F1={res['opt_test']['opt_f1']} (th={res['opt_test']['opt_thresh']})")
    
    df_opt = pd.DataFrame(opt_results)
    print("\n--- Optimizer Comparison Table ---")
    print(df_opt.to_string())
    df_opt.to_csv("opt_search_results.csv", index=False)
    
    print("\nTesting Aggregation & Weighting Strategies...")
    agg_configs = [
        ("Sample_Size_Weighting", "robust", "sample_size"),
        ("Sqrt_Size_Weighting", "robust", "sqrt_size"),
        ("Uniform_Weighting", "robust", "uniform"),
        ("Cosine_Trust_Sample_Size", "cosine_trust", "sample_size"),
        ("Cosine_Trust_Sqrt_Size", "cosine_trust", "sqrt_size"),
    ]
    
    agg_results = []
    for name, def_m, w_m in agg_configs:
        res = run_fl_experiment(
            lambda: M2_MediumMLP(), noniid_clients, X_test, y_test, X_val, y_val,
            rounds=8, epochs=1, lr=0.05, opt_name="sgd",
            defense=def_m, weighting=w_m,
            malicious_clients=[1], scale_factor=15.0
        )
        agg_results.append({
            "strategy": name,
            "f1_0.5": res["test_default"]["f1"],
            "acc_0.5": res["test_default"]["acc"],
            "prec_0.5": res["test_default"]["prec"],
            "rec_0.5": res["test_default"]["rec"],
            "opt_thresh": res["opt_test"]["opt_thresh"],
            "opt_f1": res["opt_test"]["opt_f1"]
        })
        print(f"Agg {name}: Default F1={res['test_default']['f1']}, Opt F1={res['opt_test']['opt_f1']}")
        
    df_agg = pd.DataFrame(agg_results)
    print("\n--- Aggregation Comparison Table ---")
    print(df_agg.to_string())
    df_agg.to_csv("agg_search_results.csv", index=False)

if __name__ == '__main__':
    main()
