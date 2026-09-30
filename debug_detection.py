import sys, time, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import (
    load_data, make_partitions, df_to_tensors,
    get_flat_params, set_flat_params,
    seed_everything, SEED, FEATURE_COLS
)
from test_defense_and_tuning import ResNetMLP, local_train

def debug_detection():
    seed_everything(SEED)
    train_df, test_df, public_df, encoders, scaler = load_data(ROOT / 'data', ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    tensors = [df_to_tensors(df) for df in noniid_clients]
    
    print("--- Inspecting AdamW vs SGD Mom updates across rounds ---")
    for opt_type, lr in [("sgd_mom", 0.02), ("adamw", 0.005)]:
        print(f"\nOptimizer: {opt_type}, lr={lr}")
        seed_everything(SEED)
        model = ResNetMLP()
        for r in range(4):
            deltas = []
            global_flat = get_flat_params(model)
            global_params = [p.data.clone() for p in model.parameters()]
            for cid, (X, y0) in enumerate(tensors):
                y = 1 - y0 if cid == 1 else y0
                lm = local_train(model, X, y, epochs=1, lr=lr, opt_type=opt_type, mu_prox=0.001, global_params=global_params)
                delta = get_flat_params(lm) - global_flat
                if cid == 1:
                    delta = 15.0 * delta
                deltas.append(delta)
            
            delta_stack = torch.stack(deltas)
            raw_norms = delta_stack.norm(dim=1)
            # Clip
            clip_norm = 1.0
            clipped = delta_stack * torch.minimum(torch.ones_like(raw_norms), clip_norm / raw_norms)[:, None]
            # Pairwise distance
            pairwise = torch.cdist(clipped, clipped, p=2)
            peer_scores = pairwise.sum(dim=1)
            # Cosine similarity
            unit_clipped = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos_sim = torch.mm(unit_clipped, unit_clipped.t())
            mean_cos = (cos_sim.sum(dim=1) - 1.0) / 4.0
            
            print(f"Round {r+1}:")
            print(f"  Raw norms: {[round(float(n), 3) for n in raw_norms]}")
            print(f"  Peer dist scores: {[round(float(s), 3) for s in peer_scores]}")
            print(f"  Mean cosine to peers: {[round(float(c), 3) for c in mean_cos]}")
            
            # Trim top outlier
            outlier_idx = int(torch.argmax(peer_scores).item())
            print(f"  Outlier detected: Client {outlier_idx}")
            
            # Step forward
            keep = [i for i in range(5) if i != outlier_idx]
            avg_delta = clipped[keep].mean(0)
            set_flat_params(model, global_flat + avg_delta)

if __name__ == '__main__':
    debug_detection()
