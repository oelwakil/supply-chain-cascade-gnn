"""Build the README figures from results.json and results_extensions.json.

    python make_figures.py

Figures are written to figures/. Parts whose results file is missing are skipped.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)

ORDER = ['gravity', 'traffic_light', 'tabular_gbm', 'sage', 'gat']
LABEL = {'gravity': 'Gravity', 'traffic_light': 'Traffic light', 'tabular_gbm': 'Gradient boosting',
         'sage': 'GraphSAGE', 'gat': 'GAT'}
# one hue family per model class: greys for no-graph baselines, blue/teal for graph models
COLOR = {'gravity': '#b8b8b8', 'traffic_light': '#8c8c8c', 'tabular_gbm': '#5e5e5e',
         'sage': '#2b7bba', 'gat': '#0b3d91'}

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 150, 'savefig.bbox': 'tight'})


def ms(vals):
    return float(np.mean(vals)), float(np.std(vals))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


R = load('results.json')
E = load('results_extensions.json')


# ---------------------------------------------------------------- 1. main comparison
def fig_main():
    keys = [k for k in R if k.startswith('obs0.7')]
    metrics = [('ap', 'All entities'), ('ap_tier0', 'End-use (tier 0)'), ('ap_indirect', 'Indirect (2+ hops)')]
    prev = {m: np.mean([R[k]['gravity'][{'ap': 'prev', 'ap_tier0': 'prev', 'ap_indirect': 'prev_indirect'}[m]] for k in keys])
            for m, _ in metrics}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.16
    for i, m in enumerate(ORDER):
        vals = [ms([R[k][m][met] for k in keys]) for met, _ in metrics]
        ax.bar(np.arange(3) + (i - 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
               color=COLOR[m], label=LABEL[m], capsize=2, error_kw={'lw': 0.8})
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylabel('Average precision'); ax.set_ylim(0, 0.75)
    ax.set_title('Graph models find the entities hit indirectly; direct-exposure scores do not', loc='left', fontsize=10)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.savefig('figures/main_comparison.png')
    plt.close(fig)


# ---------------------------------------------------------------- 2. hidden-link fraction
def fig_hidden_fraction():
    fracs = ['0.5', '0.7', '1.0']
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m in ORDER[1:]:
        ax.plot([50, 70, 100], [R[f'obs{f}_seed0'][m]['ap'] for f in fracs], marker='o', color=COLOR[m], label=LABEL[m])
    ax.set_xlabel('Share of true supplier links observed (%)'); ax.set_ylabel('Average precision')
    ax.set_xticks([50, 70, 100]); ax.set_ylim(0.2, 0.85)
    ax.set_title('Hiding links hurts everyone; graph models stay ahead', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig('figures/hidden_links.png')
    plt.close(fig)


# ---------------------------------------------------------------- 3. non-random hiding (extension)
def fig_hide_modes():
    if not E or 'hidden' not in E:
        return
    keys0 = [k for k in R if k.startswith('obs0.7')]
    modes = [('random', [R[k] for k in keys0]),
             ('tier', list(E['hidden']['tier'].values())),
             ('region', list(E['hidden']['region'].values()))]
    names = {'random': 'Random', 'tier': 'Deep links\nhidden more', 'region': 'Opaque regions\nhidden'}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.16
    for i, m in enumerate(ORDER):
        vals = [ms([r[m]['ap'] for r in runs]) for _, runs in modes]
        ax.bar(np.arange(3) + (i - 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
               color=COLOR[m], label=LABEL[m], capsize=2, error_kw={'lw': 0.8})
    ax.set_xticks(range(3)); ax.set_xticklabels([names[k] for k, _ in modes])
    ax.set_ylabel('Average precision'); ax.set_ylim(0, 0.75)
    ax.set_title('Same 70% visibility, different patterns of missing links', loc='left', fontsize=10)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    fig.savefig('figures/nonrandom_hiding.png')
    plt.close(fig)


# ---------------------------------------------------------------- 4. calibration (extension)
def fig_calibration():
    if not E or 'calibration_decision' not in E:
        return
    C = E['calibration_decision']
    seeds = list(C.values())
    panels = [('tabular_gbm', 'Gradient boosting'), ('gat_single', 'GAT, single model'), ('gat_ensemble', 'GAT, 5-model ensemble')]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color='#888', lw=1, ls='--')
        # pool bins across seeds by averaging per-bin values (weighted by count)
        bins = {}
        for s in seeds:
            for mp, fp, n in s[key]['reliability']:
                b = round(mp * 10 - 0.5)
                bins.setdefault(b, []).append((mp, fp, n))
        xs, ys, ns = [], [], []
        for b in sorted(bins):
            arr = np.array(bins[b]); wgt = arr[:, 2]
            xs.append(np.average(arr[:, 0], weights=wgt)); ys.append(np.average(arr[:, 1], weights=wgt)); ns.append(wgt.sum())
        ax.plot(xs, ys, marker='o', color=COLOR['gat'] if 'gat' in key else COLOR['tabular_gbm'], lw=1.5)
        eces = [s[key]['ece'] for s in seeds]
        ax.set_title(f"{title}\nECE {np.mean(eces):.3f}", fontsize=9, loc='left')
        ax.set_xlabel('Predicted probability')
    axes[0].set_ylabel('Observed frequency')
    fig.suptitle('Reliability: do predicted probabilities match how often entities are actually hit?',
                 x=0.01, y=1.10, ha='left', fontsize=10)
    fig.savefig('figures/calibration.png')
    plt.close(fig)


# ---------------------------------------------------------------- 5. decision layer (extension)
def fig_decision():
    if not E or 'calibration_decision' not in E:
        return
    seeds = [s['decision'] for s in E['calibration_decision'].values()]
    ks = sorted(seeds[0]['by_k'], key=int)
    methods = ['random', 'observed out-degree', 'observed reach (descendants)', 'tabular GBM',
               'GraphSAGE ensemble', 'GAT ensemble', 'oracle (true single-shock impact)']
    style = {'random': ('#b8b8b8', '-'), 'observed out-degree': ('#8c8c8c', '-'),
             'observed reach (descendants)': ('#5e5e5e', '-'), 'tabular GBM': ('#c98a1b', '-'),
             'GraphSAGE ensemble': ('#2b7bba', '-'), 'GAT ensemble': ('#0b3d91', '-'),
             'oracle (true single-shock impact)': ('#222', ':')}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in methods:
        ys = [np.mean([s['by_k'][k][m] for s in seeds]) * 100 for k in ks]
        c, ls = style[m]
        ax.plot([int(k) for k in ks], ys, marker='o', color=c, ls=ls, label=m)
    ax.set_xlabel('Suppliers hardened (budget)'); ax.set_ylabel('Reduction in end-use industries hit (%)')
    ax.set_title('Where to spend a hardening budget: model-guided vs heuristic picks', loc='left', fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.savefig('figures/decision_layer.png')
    plt.close(fig)


if __name__ == '__main__':
    fig_main(); fig_hidden_fraction(); fig_hide_modes(); fig_calibration(); fig_decision()
    print('figures:', sorted(os.listdir('figures')))
