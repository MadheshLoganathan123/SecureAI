from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
N_CLIENTS = 5
HOLDOUT_SIZE = 15000
ROUNDS = 8
LOCAL_EPOCHS = 1
LR = 0.05
BATCH_SIZE = 256
SCALE_FACTOR = 15.0

COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CAT_COLS = ["protocol_type", "service", "flag"]
FEATURE_COLS = [c for c in COLS if c not in ["label", "difficulty"]]
FAMILY_MAP = {
    "normal": "normal",
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "worm": "dos", "mailbomb": "dos",
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def download_if_needed(data_dir: Path) -> Tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = data_dir / "KDDTrain+.txt", data_dir / "KDDTest+.txt"
    import urllib.request
    urls = {
        train_path: "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
        test_path: "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
    }
    for path, url in urls.items():
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return train_path, test_path


def load_data(data_dir: Path, public_test_path: Optional[Path] = None):
    train_path, test_path = download_if_needed(data_dir)
    train_raw = pd.read_csv(train_path, names=COLS)
    test_raw = pd.read_csv(test_path, names=COLS)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["difficulty"]).copy()
        df["binary_label"] = (df["label"].str.rstrip(".") != "normal").astype(int)
        return df

    train_df, test_df = clean(train_raw), clean(test_raw)
    encoders = {}
    for c in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[c], test_df[c]], axis=0))
        train_df[c] = le.transform(train_df[c])
        test_df[c] = le.transform(test_df[c])
        encoders[c] = le
    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

    public_df = None
    if public_test_path and public_test_path.exists():
        public_df = pd.read_csv(public_test_path)
    return train_df, test_df, public_df, encoders, scaler


def make_partitions(train_df: pd.DataFrame, seed: int = SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_df))
    holdout_idx, pool_idx = perm[:HOLDOUT_SIZE], perm[HOLDOUT_SIZE:]
    train_pool = train_df.iloc[pool_idx].reset_index(drop=True)

    def iid(df):
        idx = rng.permutation(len(df))
        return [df.iloc[c].reset_index(drop=True) for c in np.array_split(idx, N_CLIENTS)]

    def noniid(df, alpha=0.3):
        work = df.copy()
        work["family"] = work["label"].map(lambda x: FAMILY_MAP.get(str(x).rstrip("."), "r2l_u2r_other"))
        client_indices = [[] for _ in range(N_CLIENTS)]
        for fam in work["family"].unique():
            fam_idx = work.index[work["family"] == fam].to_numpy().copy()
            rng.shuffle(fam_idx)
            proportions = rng.dirichlet(alpha=[alpha] * N_CLIENTS)
            split_points = (np.cumsum(proportions) * len(fam_idx)).astype(int)[:-1]
            for i, s in enumerate(np.split(fam_idx, split_points)):
                client_indices[i].extend(s.tolist())
        out = []
        for ci in client_indices:
            rng.shuffle(ci)
            out.append(work.loc[ci].drop(columns=["family"]).reset_index(drop=True))
        return out
    return train_pool, iid(train_pool), noniid(train_pool), holdout_idx


class WeakMLP(nn.Module):
    """Baseline 8-neuron model for benchmark traceability."""
    def __init__(self, n_features: int = len(FEATURE_COLS), hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResNetMLP(nn.Module):
    """
    Optimized intrusion-detection architecture:
    41 features -> LayerNorm + GELU(32) -> Residual Block(32) -> Head(1).
    Full TorchScript compatibility, accepts (N, 41), returns (N,).
    """
    def __init__(self, in_features: int = len(FEATURE_COLS), hidden_dim: int = 32):
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


def df_to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32),
            torch.tensor(df["binary_label"].values, dtype=torch.float32))


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i+n].view_as(p))
        i += n


def local_train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                epochs: int = LOCAL_EPOCHS, lr: float = LR, batch_size: int = BATCH_SIZE) -> nn.Module:
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return model


def local_train_adamw_prox(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                           epochs: int = 1, lr: float = 0.005, batch_size: int = 256,
                           mu_prox: float = 0.001, global_params: Optional[List[torch.Tensor]] = None) -> nn.Module:
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


def comprehensive_metrics(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, thresh: float = 0.5) -> Dict:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).cpu().numpy()
        pred = (probs >= thresh).astype(int)
    yt = y_test.cpu().numpy().astype(int)
    cm = confusion_matrix(yt, pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(yt, pred)
    prec = precision_score(yt, pred, zero_division=0)
    rec = recall_score(yt, pred, zero_division=0)
    f1 = f1_score(yt, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(yt, pred)
    mcc = matthews_corrcoef(yt, pred)
    roc_auc = roc_auc_score(yt, probs)
    p_curve, r_curve, _ = precision_recall_curve(yt, probs)
    pr_auc = auc(r_curve, p_curve)
    
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "specificity": round(float(spec), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "mcc": round(float(mcc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "fpr": round(float(fp / (fp + tn) if (fp + tn) > 0 else 0.0), 4),
        "fnr": round(float(fn / (fn + tp) if (fn + tp) > 0 else 0.0), 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def aggregate_fedavg(models: List[nn.Module], sizes: List[int], prev: nn.Module) -> nn.Module:
    weights = torch.tensor(np.asarray(sizes, dtype=float) / np.sum(sizes), dtype=torch.float32)
    avg = (torch.stack([get_flat_params(m) for m in models]) * weights[:, None]).sum(0)
    out = copy.deepcopy(prev); set_flat_params(out, avg); return out


def aggregate_trimmed_clip(models: List[nn.Module], sizes: List[int], prev: nn.Module, clip_norm: float = 1.0, trim: int = 1) -> nn.Module:
    ref = get_flat_params(prev)
    deltas = torch.stack([get_flat_params(m) - ref for m in models])
    norms = deltas.norm(dim=1).clamp_min(1e-12)
    clipped = deltas * torch.minimum(torch.ones_like(norms), clip_norm / norms)[:, None]
    pairwise = torch.cdist(clipped, clipped, p=2)
    scores = pairwise.sum(dim=1)
    keep = torch.argsort(scores)[:len(models)-trim] if len(models) > trim else torch.arange(len(models))
    w = torch.tensor(np.asarray(sizes, dtype=float)[keep.numpy()], dtype=torch.float32)
    w = w / w.sum()
    avg = (clipped[keep] * w[:, None]).sum(0)
    out = copy.deepcopy(prev); set_flat_params(out, ref + avg); return out


def aggregate_robust_multi(models: List[nn.Module], sizes: List[int], prev: nn.Module, clip_norm: float = 1.0, weighting: str = "sqrt_size"):
    """
    State-of-the-art Multi-Criterion Defense:
    1. L2 Norm Bounding (protects against scale attacks).
    2. Multi-criterion Anomaly Detection:
       Combines norm inflation ratio, pairwise peer Euclidean distance, and peer cosine alignment.
    3. Non-IID Skew Mitigation:
       Square-root sample weighting (sqrt(len(X))) dampens dominant clients (e.g. Client 3) to prevent bias.
    """
    ref = get_flat_params(prev)
    deltas = torch.stack([get_flat_params(m) - ref for m in models])
    raw_norms = deltas.norm(dim=1).clamp_min(1e-12)
    clipped = deltas * torch.minimum(torch.ones_like(raw_norms), clip_norm / raw_norms)[:, None]
    
    unit_clipped = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_sim = torch.mm(unit_clipped, unit_clipped.t())
    mean_cos = (cos_sim.sum(dim=1) - 1.0) / (len(models) - 1)
    
    pairwise = torch.cdist(clipped, clipped, p=2)
    peer_dist = pairwise.sum(dim=1)
    
    med_norm = torch.median(raw_norms)
    norm_ratio = raw_norms / (med_norm + 1e-6)
    
    anomaly_scores = peer_dist * norm_ratio - 2.0 * mean_cos
    malicious_idx = int(torch.argmax(anomaly_scores).item())
    keep = [i for i in range(len(models)) if i != malicious_idx]
    
    if weighting == "sqrt_size":
        kw = np.sqrt(np.asarray(sizes, dtype=float)[keep])
    elif weighting == "sample_size":
        kw = np.asarray(sizes, dtype=float)[keep]
    else:
        kw = np.ones(len(keep), dtype=float)
    kw = kw / kw.sum()
    
    avg = (clipped[keep] * torch.tensor(kw, dtype=torch.float32)[:, None]).sum(0)
    out = copy.deepcopy(prev)
    set_flat_params(out, ref + avg)
    return out, malicious_idx, [round(float(s), 3) for s in anomaly_scores]


def run_fl(client_dfs, X_test, y_test, model_type="weak", aggregation="fedavg",
           malicious_clients=None, scale_factor=SCALE_FACTOR, rounds=ROUNDS, seed=SEED,
           clip_norm=1.0, verbose=False):
    seed_everything(seed)
    malicious_clients = set(malicious_clients or [])
    
    if model_type == "weak":
        model = WeakMLP()
    elif model_type == "resnet":
        model = ResNetMLP()
    else:
        raise ValueError(model_type)
        
    tensors = [df_to_tensors(df) for df in client_dfs]
    history, diagnostics = [], []
    t0 = time.perf_counter()
    
    for r in range(rounds):
        local_models, sizes, norms = [], [], []
        global_flat = get_flat_params(model)
        global_params = [p.data.clone() for p in model.parameters()]
        
        for cid, (X, y0) in enumerate(tensors):
            y = 1 - y0 if cid in malicious_clients else y0
            if model_type == "resnet":
                lm = local_train_adamw_prox(model, X, y, epochs=1, lr=0.005, mu_prox=0.001, global_params=global_params)
            else:
                lm = local_train(model, X, y, epochs=LOCAL_EPOCHS, lr=LR)
                
            delta = get_flat_params(lm) - global_flat
            if cid in malicious_clients:
                delta = scale_factor * delta
                set_flat_params(lm, global_flat + delta)
                
            local_models.append(lm)
            sizes.append(len(X))
            norms.append(float(delta.norm()))
            
        if aggregation == "fedavg":
            model = aggregate_fedavg(local_models, sizes, model)
        elif aggregation == "robust_baseline":
            model = aggregate_trimmed_clip(local_models, sizes, model, clip_norm=clip_norm, trim=1)
        elif aggregation == "robust_multi":
            model, mal_idx, scores = aggregate_robust_multi(local_models, sizes, model, clip_norm=clip_norm, weighting="sqrt_size")
            diagnostics.append({"round": r+1, "detected_malicious": mal_idx, "anomaly_scores": scores})
        else:
            raise ValueError(aggregation)
            
        m = comprehensive_metrics(model, X_test, y_test)
        history.append(m)
        if verbose:
            print(f"round {r+1}: F1={m['f1']}, Acc={m['accuracy']}")
            
    return model, history, diagnostics, time.perf_counter() - t0


def client_analysis(clients):
    rows = []
    for i, df in enumerate(clients):
        counts = df["binary_label"].value_counts().to_dict()
        rows.append({
            "client": i,
            "samples": len(df),
            "normal": int(counts.get(0, 0)),
            "attack": int(counts.get(1, 0)),
            "attack_rate": round(float(df.binary_label.mean()), 4)
        })
    return pd.DataFrame(rows)


def save_figures(out: Path, analysis_df, experiment_rows, histories, test_y, final_model, X_test):
    sns.set_theme(style="whitegrid")
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Client Distribution
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(analysis_df)); w = 0.36
    plt.bar(x - w/2, analysis_df.normal, w, label="Normal", color="#4C72B0")
    plt.bar(x + w/2, analysis_df.attack, w, label="Attack", color="#DD8452")
    plt.xticks(x, [f"Client {i}" for i in analysis_df.client])
    plt.ylabel("Samples")
    plt.title("Client Label Distribution (Non-IID Traffic Skew)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "client_distribution.png", dpi=160)
    plt.close()

    # 2. F1 Comparison
    plot_df = pd.DataFrame(experiment_rows)
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=plot_df, x="method", y="f1", hue="scenario", palette="Blues_d")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.title("Federated Detection Performance Across Scenarios")
    plt.xticks(rotation=15)
    for p in plt.gca().patches:
        h = p.get_height()
        if h > 0.01:
            plt.gca().annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h / 2),
                               ha='center', va='center', color='white', weight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "f1_comparison.png", dpi=160)
    plt.close()

    # 3. F1 Over Rounds
    plt.figure(figsize=(8, 4.5))
    for name, hist in histories.items():
        plt.plot(range(1, len(hist) + 1), [h["f1"] for h in hist], marker="o", label=name, linewidth=2)
    plt.xlabel("Communication Round")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.title("Convergence & F1 Trajectory Over Federated Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "f1_over_rounds.png", dpi=160)
    plt.close()

    # 4. Final Confusion Matrix
    with torch.no_grad():
        pred = (torch.sigmoid(final_model(X_test)) >= 0.5).cpu().numpy().astype(int)
    cm = confusion_matrix(test_y.cpu().numpy().astype(int), pred)
    plt.figure(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Final Defended Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()


def write_notebook(path: Path):
    cells = [
        {
            "cell_type": "markdown",
            "id": "intro_cell",
            "metadata": {},
            "source": [
                "# CAIRLab Secure AI Hackathon 2026 — Final Submission\n",
                "This notebook runs the reproducible pipeline, preserves the challenge holdout procedure, compares naive FedAvg with robust aggregation, and exports the final artifacts.\n",
                "\n",
                "### Architectural & Algorithmic Highlights:\n",
                "- **ResNetMLP** (41 features -> LayerNorm + GELU(32) -> Residual Block(32) -> Head(1))\n",
                "- **AdamW Local Training** with FedProx proximal regularization (mu=0.001) to control client drift\n",
                "- **Multi-Criterion Robust Aggregation**: Norm clipping, peer distance trimming, and cosine alignment\n",
                "- **Square-Root Sample Weighting** to mitigate severe non-IID label skew across simulated banks"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "setup_cell",
            "metadata": {},
            "source": ["## 1. Data loading and validation"]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "id": "code_load",
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import sys, json, pandas as pd, torch\n",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'final_submission' else Path.cwd()\n",
                "sys.path.insert(0, str(ROOT))\n",
                "from src.pipeline import *\n",
                "OUT = ROOT / 'final_submission'\n",
                "DATA = ROOT / 'data'\n",
                "seed_everything(SEED)\n",
                "train_df, test_df, public_df, encoders, scaler = load_data(DATA, ROOT / 'test_public.csv')\n",
                "print('train/test:', train_df.shape, test_df.shape, 'public:', None if public_df is None else public_df.shape)\n",
                "assert len(FEATURE_COLS) == 41 and public_df is not None"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "client_cell",
            "metadata": {},
            "source": ["## 2. Five-client analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "id": "code_analysis",
            "metadata": {},
            "outputs": [],
            "source": [
                "train_pool, iid_clients, noniid_clients, holdout_idx = make_partitions(train_df)\n",
                "analysis = client_analysis(noniid_clients)\n",
                "analysis"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "train_cell",
            "metadata": {},
            "source": ["## 3. Baseline and proposed robust aggregation"]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "id": "code_train",
            "metadata": {},
            "outputs": [],
            "source": [
                "X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)\n",
                "y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)\n",
                "base, base_hist, _, base_time = run_fl(noniid_clients, X_test, y_test, model_type='weak', aggregation='fedavg')\n",
                "attacked, attacked_hist, _, attacked_time = run_fl(noniid_clients, X_test, y_test, model_type='weak', aggregation='fedavg', malicious_clients=[1])\n",
                "prev_def, prev_hist, _, prev_time = run_fl(noniid_clients, X_test, y_test, model_type='weak', aggregation='robust_baseline', malicious_clients=[1])\n",
                "defended, defended_hist, _, defended_time = run_fl(noniid_clients, X_test, y_test, model_type='resnet', aggregation='robust_multi', malicious_clients=[1])\n",
                "print('clean FedAvg F1:     ', base_hist[-1]['f1'])\n",
                "print('attacked FedAvg F1:  ', attacked_hist[-1]['f1'])\n",
                "print('previous defense F1: ', prev_hist[-1]['f1'])\n",
                "print('final robust F1:     ', defended_hist[-1]['f1'], f'(F1 Gain: +{round(defended_hist[-1][\"f1\"] - attacked_hist[-1][\"f1\"], 4)})')"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "export_cell",
            "metadata": {},
            "source": ["## 4. Export and visualizations"]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "id": "code_export",
            "metadata": {},
            "outputs": [],
            "source": [
                "OUT.mkdir(exist_ok=True)\n",
                "(OUT / 'figures').mkdir(exist_ok=True)\n",
                "rows = []\n",
                "for method, scenario, h in [\n",
                "    ('FedAvg', 'clean', base_hist[-1]),\n",
                "    ('FedAvg', 'attack', attacked_hist[-1]),\n",
                "    ('PreviousDefense', 'attack', prev_hist[-1]),\n",
                "    ('Robust', 'attack', defended_hist[-1])\n",
                "]:\n",
                "    rows.append({'method': method, 'scenario': scenario, **h})\n",
                "save_figures(OUT / 'figures', analysis, rows, {\n",
                "    'FedAvg clean': base_hist,\n",
                "    'FedAvg attack': attacked_hist,\n",
                "    'Previous defense': prev_hist,\n",
                "    'Robust defense attack': defended_hist\n",
                "}, y_test, defended, X_test)\n",
                "scripted = torch.jit.script(defended)\n",
                "scripted.save(OUT / 'model_scripted.pt')\n",
                "print('saved', OUT / 'model_scripted.pt')"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "metrics_cell",
            "metadata": {},
            "source": ["## 5. Final metrics"]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "id": "code_metrics",
            "metadata": {},
            "outputs": [],
            "source": [
                "print(json.dumps({\n",
                "    'baseline_clean': base_hist[-1],\n",
                "    'baseline_attack': attacked_hist[-1],\n",
                "    'previous_defended': prev_hist[-1],\n",
                "    'final_defended': defended_hist[-1],\n",
                "    'f1_recovered': round(defended_hist[-1]['f1'] - attacked_hist[-1]['f1'], 4)\n",
                "}, indent=2))"
            ]
        }
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path.write_text(json.dumps(nb, indent=2))


def run_pipeline(root: Path):
    seed_everything(SEED)
    out = root / 'final_submission'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (root / 'data').mkdir(exist_ok=True)
    
    train_df, test_df, public_df, encoders, scaler = load_data(root / 'data', root / 'test_public.csv')
    _, iid, noniid, _ = make_partitions(train_df, seed=SEED)
    
    X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['binary_label'].values, dtype=torch.float32)
    
    print("Running Experiment E001: Clean FedAvg (WeakMLP)...")
    clean, clean_hist, _, clean_time = run_fl(noniid, X_test, y_test, model_type="weak", aggregation="fedavg")
    
    print("Running Experiment E002: Attacked FedAvg (WeakMLP, 15x scale)...")
    attack, attack_hist, _, attack_time = run_fl(noniid, X_test, y_test, model_type="weak", aggregation="fedavg", malicious_clients=[1])
    
    print("Running Experiment E003: Previous Defended Model (WeakMLP, peer trim)...")
    prev_def, prev_hist, _, prev_time = run_fl(noniid, X_test, y_test, model_type="weak", aggregation="robust_baseline", malicious_clients=[1])
    
    print("Running Experiment E004: Final Optimized Defended Model (ResNetMLP, multi-criterion defense, sqrt weighting)...")
    final, final_hist, final_diag, final_time = run_fl(noniid, X_test, y_test, model_type="resnet", aggregation="robust_multi", malicious_clients=[1])
    
    rows = [
        {
            'experiment_id': 'E001', 'track': 'advanced', 'method': 'FedAvg', 'scenario': 'non_iid_clean',
            'seed': SEED, 'rounds': ROUNDS, 'local_epochs': LOCAL_EPOCHS, 'learning_rate': LR,
            'clip_norm': None, 'trim': None, 'aggregation': 'fedavg',
            'precision': clean_hist[-1]['precision'], 'recall': clean_hist[-1]['recall'],
            'f1': clean_hist[-1]['f1'], 'accuracy': clean_hist[-1]['accuracy'],
            'runtime_seconds': round(clean_time, 3),
            'notes': 'same held-out test protocol as challenge notebook'
        },
        {
            'experiment_id': 'E002', 'track': 'advanced', 'method': 'FedAvg', 'scenario': 'poisoned_scale',
            'seed': SEED, 'rounds': ROUNDS, 'local_epochs': LOCAL_EPOCHS, 'learning_rate': LR,
            'clip_norm': None, 'trim': None, 'aggregation': 'fedavg',
            'precision': attack_hist[-1]['precision'], 'recall': attack_hist[-1]['recall'],
            'f1': attack_hist[-1]['f1'], 'accuracy': attack_hist[-1]['accuracy'],
            'runtime_seconds': round(attack_time, 3),
            'notes': 'held-out NSL-KDD test labels used only for evaluation; malicious client is 1 with label-flip and scale attack'
        },
        {
            'experiment_id': 'E003', 'track': 'advanced', 'method': 'ClippedTrimmedPeerRobust', 'scenario': 'poisoned_scale',
            'seed': SEED, 'rounds': ROUNDS, 'local_epochs': LOCAL_EPOCHS, 'learning_rate': LR,
            'clip_norm': 1.0, 'trim': 1.0, 'aggregation': 'norm-clipping + peer-distance trim + size-weighted mean',
            'precision': prev_hist[-1]['precision'], 'recall': prev_hist[-1]['recall'],
            'f1': prev_hist[-1]['f1'], 'accuracy': prev_hist[-1]['accuracy'],
            'runtime_seconds': round(prev_time, 3),
            'notes': 'previous baseline defense'
        },
        {
            'experiment_id': 'E004', 'track': 'advanced', 'method': 'OptimizedResNetMLP_MultiRobust', 'scenario': 'poisoned_scale',
            'seed': SEED, 'rounds': ROUNDS, 'local_epochs': 1, 'learning_rate': 0.005,
            'clip_norm': 1.0, 'trim': 1.0, 'aggregation': 'norm-clipping + multi-criterion anomaly trim + sqrt-size weighted mean',
            'precision': final_hist[-1]['precision'], 'recall': final_hist[-1]['recall'],
            'f1': final_hist[-1]['f1'], 'accuracy': final_hist[-1]['accuracy'],
            'runtime_seconds': round(final_time, 3),
            'notes': 'final optimized defense with ResNetMLP, AdamW, FedProx, and sqrt weighting'
        },
    ]
    
    exp = pd.DataFrame(rows)
    exp.to_csv(out / 'experiment_results.csv', index=False)
    
    analysis = client_analysis(noniid)
    analysis.to_csv(out / 'client_analysis.csv', index=False)
    
    histories = {
        'FedAvg Clean': clean_hist,
        'FedAvg Attack': attack_hist,
        'Previous Defense': prev_hist,
        'Final Defended': final_hist
    }
    save_figures(out / 'figures', analysis, rows, histories, y_test, final, X_test)
    
    scripted = torch.jit.script(final)
    scripted.save(out / 'model_scripted.pt')
    
    with torch.no_grad():
        public_X = torch.tensor(public_df[FEATURE_COLS].values, dtype=torch.float32)
        public_pred = (torch.sigmoid(scripted(public_X)) >= 0.5).numpy().astype(int)
    submission = pd.DataFrame({'Id': public_df['Id'].astype(int), 'Expected': public_pred})
    submission.to_csv(out / 'submission.csv', index=False)
    
    recovered_f1 = round(final_hist[-1]['f1'] - attack_hist[-1]['f1'], 4)
    submission_json = {
        'team_name': 'CAIRLab Secure AI Hackathon Submission',
        'track': 'advanced',
        'self_reported_metrics': final_hist[-1],
        'baseline_f1_under_attack': attack_hist[-1]['f1'],
        'f1_recovered': recovered_f1,
        'model_file': 'model_scripted.pt',
        'n_input_features': len(FEATURE_COLS),
        'architecture': 'ResNetMLP: 41 -> LayerNorm/GELU(32) -> ResBlock(32) -> Linear(1)',
        'aggregation': 'per-client L2 norm clipping (1.0), multi-criterion anomaly trimming (norm ratio + peer distance + cosine alignment), square-root sample weighting',
        'seed': SEED,
        'rounds': ROUNDS
    }
    (out / 'submission.json').write_text(json.dumps(submission_json, indent=2))
    
    final_metrics = {
        'selected_track': 'Advanced',
        'architecture': 'ResNetMLP 41-32-32-1',
        'baseline_clean': clean_hist[-1],
        'baseline_attack': attack_hist[-1],
        'previous_defense': prev_hist[-1],
        'final_defended': final_hist[-1],
        'f1_recovered': recovered_f1,
        'runtimes_seconds': {'clean': clean_time, 'attack': attack_time, 'previous': prev_time, 'final': final_time},
        'parameter_count': sum(p.numel() for p in final.parameters())
    }
    (out / 'final_metrics.json').write_text(json.dumps(final_metrics, indent=2))
    
    # Copy comparison CSVs to final_submission
    for csv_name in ['intermediate_final_comparison.csv', 'advanced_final_comparison.csv', 'ablation_study.csv', 'final_client_generalization.csv', 'multi_seed_results.csv']:
        src_csv = root / csv_name
        if src_csv.exists():
            (out / csv_name).write_text(src_csv.read_text())
            
    write_notebook(out / 'final_notebook.ipynb')
    return final_metrics


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_pipeline(root), indent=2))
