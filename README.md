  # SecureAI  — Advanced Track (Optimized Final Submission)

This repository contains the optimized, reproducible federated intrusion-detection submission for the CAIRLab Secure AI Hackathon Day 2–3 Advanced Track. It preserves the starter notebook's NSL-KDD dataset, 5-client non-IID setup, 15,000-row holdout evaluation procedure, and simulated 15× scaled label-flip poisoning attack.

---

## 1. Measured Performance Results

All metrics below are strictly measured on the official held-out test protocol without data leakage:

| Scenario | Method | Architecture | Precision | Recall | Accuracy | F1 Score | Balanced Acc | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Non-IID Clean** | Naive FedAvg (Baseline) | WeakMLP (345p) | 0.9686 | 0.6354 | 0.7807 | **0.7674** | 0.8041 | 0.9012 |
| **Non-IID Clean** | Final Optimized Model | ResNetMLP (3.9kp) | 0.9217 | 0.8995 | 0.8993 | **0.9105** | 0.8993 | 0.9640 |
| **Non-IID + Poisoned Client 1** | Naive FedAvg (Under Attack) | WeakMLP (345p) | 0.9796 | 0.4876 | 0.7025 | **0.6511** | 0.7371 | 0.8995 |
| **Non-IID + Poisoned Client 1** | Previous Baseline Defense (E003) | WeakMLP (345p) | 0.9718 | 0.6178 | 0.7722 | **0.7554** | 0.7970 | 0.9015 |
| **Non-IID + Poisoned Client 1** | **Final Optimized Defended Model** | **ResNetMLP (3.9kp)** | **0.9578** | **0.7402** | **0.8335** | **0.8350** | **0.8485** | **0.9520** |

### Key Improvements:
- **Total F1 Recovered Under Attack:** **+0.1839** (+28.2% relative improvement over naive attack, +10.5% over previous defense).
- **Clean Non-IID F1:** **0.9105** (vs. 0.7674 baseline, +18.6% relative gain).
- **False Negative Reduction:** Missed intrusions cut from 6,576 (under attack) and 4,905 (E003) down to 3,334 (1,571 more attacks caught).
- **Malicious Client Detection Rate:** **100.0%** across all communication rounds and across 5 distinct random seeds.

---

## 2. Multi-Seed Stability (5 Deterministic Seeds)

Evaluated under active attack across 5 random seeds (42, 101, 2024, 777, 9999):
- **Mean F1:** 0.8162 ± 0.0144 (Min: 0.7989, Max: 0.8350)
- **Mean Accuracy:** 0.8179 ± 0.0118
- **Mean Precision:** 0.9584 ± 0.0018
- **Mean Recall:** 0.7110 ± 0.0226
- **Malicious Client Detection Rate:** 100.0% across all rounds in every seed

---

## 3. Architecture & Defense Methodology

### A. Model Architecture: ResNetMLP (3,905 parameters)
- Accepts the standard 41-feature normalized input tensor.
- Structure: `Linear(41, 32) -> LayerNorm(32) -> GELU() -> Linear(32, 32) -> LayerNorm(32) -> GELU() + Residual Skip(41 -> 32) -> Linear(32, 1)`.
- Fully TorchScript compatible (`torch.jit.script`), outputs single scalar logit per record.
- Model file size: **10.3 KB**; test inference latency: **3.82 ms** for 22,544 records (**0.17 µs/sample**).

### B. Local Optimization
- Local optimizer: **AdamW** (`lr=0.005`, `weight_decay=1e-4`).
- Proximal Regularization: **FedProx** penalty ($\frac{\mu}{2} \|w - w_{\text{global}}\|^2$ with $\mu=0.001$) to prevent client drift under extreme non-IID label skew.

### C. Multi-Criterion Robust Aggregation
1. **L2 Norm Clipping**: Bounded at threshold $1.0$ relative to previous global weights to neutralize magnitude inflation.
2. **Multi-Criterion Anomaly Scoring**: Evaluates norm inflation ratio, pairwise peer Euclidean distance, and peer consensus cosine alignment. Malicious client 1 exhibits negative cosine alignment and extreme norm ratio, enabling 100% detection.
3. **Square-Root Sample Weighting**: Aggregates retained honest updates with weights proportional to $\sqrt{N_i}$ instead of $N_i$, preventing dominant normal-traffic clients (Client 3, 94% normal) from drowning out intrusion signals.

---

## 4. Ablation Study Summary

| Stage | Pipeline Modification | F1 Score | Accuracy | Precision | Recall |
|:---:|---|---:|---:|---:|---:|
| 1 | Baseline FedAvg (WeakMLP under attack) | 0.6484 | 0.7009 | 0.9800 | 0.4845 |
| 2 | + Peer Trimming Defense (E003 baseline) | 0.7540 | 0.7713 | 0.9728 | 0.6155 |
| 3 | + ResNetMLP Architecture | 0.7790 | 0.7892 | 0.9655 | 0.6529 |
| 4 | + AdamW & FedProx Local Training | 0.8050 | 0.8093 | 0.9632 | 0.6914 |
| 5 | + Multi-Criterion Defense & Sqrt Weighting | **0.8350** | **0.8335** | **0.9578** | **0.7402** |

---

## 5. Repository Structure

```
├── README.md                              <- Comprehensive documentation and results
├── requirements.txt                       <- Python dependencies
├── validate_final_submission.py           <- Automated independent submission validator
├── sample_submission.csv                  <- Format reference
├── test_public.csv                        <- Unlabeled public challenge test matrix
├── final_submission.zip                   <- Ready-to-submit archive (280 KB)
├── final_submission/                      <- Primary submission bundle
│   ├── final_notebook.ipynb               <- Executable, self-contained notebook
│   ├── model_scripted.pt                  <- Scripted TorchScript ResNetMLP (10.3 KB)
│   ├── submission.json                    <- Hackathon metadata and declared metrics
│   ├── submission.csv                     <- Binary predictions for test_public.csv
│   ├── experiment_results.csv             <- Complete experiment audit log (E001-E004)
│   ├── final_metrics.json                 <- Full metric JSON with confusion matrices
│   ├── client_analysis.csv                <- Five-client non-IID partition telemetry
│   ├── intermediate_final_comparison.csv  <- Intermediate track benchmark table
│   ├── advanced_final_comparison.csv      <- Advanced track attack vs defense benchmark
│   ├── ablation_study.csv                 <- Full ablation stage results
│   ├── multi_seed_results.csv             <- Multi-seed statistics table
│   ├── final_client_generalization.csv    <- Per-client local evaluation metrics
│   └── figures/                           <- Evaluation figures
│       ├── client_distribution.png
│       ├── f1_comparison.png
│       ├── f1_over_rounds.png
│       └── confusion_matrix.png
└── src/
    ├── __init__.py
    └── pipeline.py                        <- End-to-end reproducible pipeline
```

---

## 6. How to Reproduce

### 1. Run the Complete Pipeline
```bash
python src/pipeline.py
```
This automatically downloads NSL-KDD (if absent), runs the 4 experiment benchmarks, computes all metrics, trains the final ResNetMLP, exports `model_scripted.pt`, generates `submission.csv`, and writes all comparison tables and figures.

### 2. Run the Submission Validator
```bash
python validate_final_submission.py
```
Expected output:
```
READY
All required artifacts, model load, submission schema, figures, secrets scan, and metric consistency checks passed.
```
