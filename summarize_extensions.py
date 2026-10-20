"""Aggregate results_extensions.json over seeds (mean +/- sd) and write summary_extensions.json."""
import json

import numpy as np

E = json.load(open('results_extensions.json'))
R = json.load(open('results.json'))
MODELS = ['gravity', 'traffic_light', 'tabular_gbm', 'sage', 'gat']


def ms(v):
    return [float(np.mean(v)), float(np.std(v))]


S = {}
base_keys = [k for k in R if k.startswith('obs0.7')]
S['hiding'] = {'random': {m: ms([R[k][m]['ap'] for k in base_keys]) for m in MODELS}}
for mode in ('tier', 'region'):
    runs = list(E['hidden'][mode].values())
    S['hiding'][mode] = {m: ms([r[m]['ap'] for r in runs]) for m in MODELS}
S['hiding_indirect'] = {'random': {m: ms([R[k][m]['ap_indirect'] for k in base_keys]) for m in MODELS}}
for mode in ('tier', 'region'):
    runs = list(E['hidden'][mode].values())
    S['hiding_indirect'][mode] = {m: ms([r[m]['ap_indirect'] for r in runs]) for m in MODELS}

runs = list(E['features'].values())
S['features'] = {'default': {m: ms([R[k][m]['ap'] for k in base_keys]) for m in MODELS},
                 'rich': {m: ms([r[m]['ap'] for r in runs]) for m in MODELS}}

C = list(E['calibration_decision'].values())
S['calibration'] = {name: {k: ms([c[name][k] for c in C]) for k in ('ap', 'auroc', 'brier', 'ece')}
                    for name in ('tabular_gbm', 'sage_single', 'sage_ensemble', 'gat_single', 'gat_ensemble')}

D = [c['decision'] for c in C]
ks = list(D[0]['by_k'])
methods = list(D[0]['by_k'][ks[0]])
S['decision'] = {'baseline_tier0_hit': ms([d['baseline_tier0_hit'] for d in D]),
                 'by_k': {k: {m: ms([d['by_k'][k][m] for d in D]) for m in methods} for k in ks}}
json.dump(S, open('summary_extensions.json', 'w'), indent=1)


def row(label, d, f='{:.2f} ± {:.2f}'):
    return f'{label:34s} ' + '  '.join(f.format(*d[m]) for m in MODELS)


print('AP by link-hiding pattern (70% observed on average)        ' + '  '.join(f'{m:14s}' for m in MODELS))
for mode in S['hiding']:
    print(row(mode, S['hiding'][mode]))
print('\nAP, indirect (2+ hops)')
for mode in S['hiding_indirect']:
    print(row(mode, S['hiding_indirect'][mode]))
print('\nAP by feature set')
for k in S['features']:
    print(row(k, S['features'][k]))
print('\nCalibration (mean ± sd over seeds)')
for n, v in S['calibration'].items():
    print(f"{n:16s} " + '  '.join(f"{k} {v[k][0]:.4f}±{v[k][1]:.4f}" for k in v))
print('\nDecision layer: % reduction in end-use industries hit (baseline hit per scenario '
      f"{S['decision']['baseline_tier0_hit'][0]:.1f})")
for k in ks:
    print('k =', k)
    for m, v in sorted(S['decision']['by_k'][k].items(), key=lambda kv: -kv[1][0]):
        print(f'   {m:36s} {v[0]*100:5.1f} ± {v[1]*100:4.1f}')
