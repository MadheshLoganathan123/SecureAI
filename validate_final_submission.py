from pathlib import Path
import json, re, sys
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_submission'
errors=[]
required=['final_notebook.ipynb','model_scripted.pt','submission.json','submission.csv','experiment_results.csv','final_metrics.json','client_analysis.csv']
for name in required:
    if not (OUT/name).exists(): errors.append(f'missing final_submission/{name}')
for name in ['client_distribution.png','f1_comparison.png','f1_over_rounds.png','confusion_matrix.png']:
    p=OUT/'figures'/name
    if not p.exists() or p.stat().st_size < 1000: errors.append(f'missing or tiny figure: figures/{name}')

try:
    model=torch.jit.load(str(OUT/'model_scripted.pt')); model.eval()
    probe=torch.zeros((2,41),dtype=torch.float32)
    out=model(probe)
    if tuple(out.shape)!=(2,): errors.append(f'model output shape is {tuple(out.shape)}, expected (2,)')
except Exception as e: errors.append(f'model load/probe failed: {e}')

try:
    sub=json.loads((OUT/'submission.json').read_text())
    if sub.get('model_file')!='model_scripted.pt': errors.append('submission.json model_file mismatch')
    if sub.get('n_input_features')!=41: errors.append('submission.json input width mismatch')
    for k in ['precision','recall','f1','accuracy']:
        if k not in sub.get('self_reported_metrics',{}): errors.append(f'missing metric {k}')
except Exception as e: errors.append(f'submission.json invalid: {e}')

try:
    pred=pd.read_csv(OUT/'submission.csv')
    if list(pred.columns)!=['Id','Expected']: errors.append('submission.csv columns must be Id,Expected')
    if len(pred)==0 or not set(pred['Expected'].dropna().unique()).issubset({0,1}): errors.append('submission.csv Expected must be binary')
except Exception as e: errors.append(f'submission.csv invalid: {e}')

try:
    nb=json.loads((OUT/'final_notebook.ipynb').read_text())
    if not nb.get('cells'): errors.append('notebook has no cells')
    if not any('export' in ''.join(c.get('source',[])).lower() for c in nb['cells']): errors.append('notebook does not show export section')
except Exception as e: errors.append(f'notebook invalid: {e}')

try:
    text_source = (OUT/'writeup.md') if (OUT/'writeup.md').exists() else (ROOT/'README.md')
    text=text_source.read_text()
    if '0.6511' not in text or '0.7554' not in text: errors.append('missing measured baseline F1 comparison')
except Exception as e: errors.append(f'writeup/README check failed: {e}')

try:
    exp=pd.read_csv(OUT/'experiment_results.csv')
    fm=json.loads((OUT/'final_metrics.json').read_text())
    exp_def = exp.loc[exp.experiment_id == 'E004'] if 'E004' in exp.experiment_id.values else exp.loc[exp.experiment_id == 'E003']
    final=float(exp_def['f1'].iloc[0]); declared=float(fm['final_defended']['f1'])
    if abs(final-declared)>1e-9: errors.append('experiment log and final_metrics disagree')
    attack=float(exp.loc[exp.experiment_id=='E002','f1'].iloc[0])
    if abs(round(final-attack,4)-float(fm['f1_recovered']))>1e-9: errors.append('F1 recovery is inconsistent')
except Exception as e: errors.append(f'metrics consistency check failed: {e}')

secret_patterns=[r'AKIA[0-9A-Z]{16}',r'sk-[A-Za-z0-9]{20,}',r'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY',r'ghp_[A-Za-z0-9]{20,}']
for p in [ROOT/'README.md',ROOT/'requirements.txt',ROOT/'src'/'pipeline.py']:
    if p.exists():
        txt=p.read_text(errors='ignore')
        for pat in secret_patterns:
            if re.search(pat,txt): errors.append(f'possible secret pattern in {p.name}')

if errors:
    print('NOT READY')
    for e in errors: print('-',e)
    sys.exit(1)
print('READY')
print('All required artifacts, model load, submission schema, writeup, figures, secrets scan, and metric consistency checks passed.')
