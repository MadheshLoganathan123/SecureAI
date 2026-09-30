import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import load_data, make_partitions, SEED, FEATURE_COLS, CAT_COLS

def analyze_features():
    train_df, test_df, public_df, encoders, scaler = load_data(ROOT / 'data', ROOT / 'test_public.csv')
    train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df, seed=SEED)
    
    # Analyze train_pool features
    print(f"Total features: {len(FEATURE_COLS)}")
    variances = train_pool[FEATURE_COLS].var()
    low_var = variances[variances < 1e-4]
    print(f"Low variance features (< 1e-4): {low_var.to_dict()}")
    
    # Correlation with binary_label
    corrs = train_pool[FEATURE_COLS].apply(lambda c: c.corr(train_pool['binary_label'])).abs().sort_values(ascending=False)
    print("\nTop 15 most correlated features with attack label:")
    print(corrs.head(15))
    
    print("\nBottom 10 least correlated features with attack label:")
    print(corrs.tail(10))
    
    # Check categorical columns distributions
    for c in CAT_COLS:
        print(f"\nCategorical feature '{c}': {train_pool[c].nunique()} unique categories")

if __name__ == '__main__':
    analyze_features()
