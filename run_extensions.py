"""Extension experiments that address the limitations of the base study.

    python run_extensions.py hidden      # non-random link hiding (tier / region biased)
    python run_extensions.py features    # richer node features
    python run_extensions.py calib       # ensembles, calibration, uncertainty, decision layer
    python run_extensions.py all
    python run_extensions.py smoke       # tiny, fast sanity check of every part

Results are written to results_extensions.json (one key per part, updated as parts finish).
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import scsim
import models as M

OUT = 'results_extensions.json'
SEEDS = (0, 1, 2)
N_MEMBERS = 5


def save(key, value):
    cur = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cur[key] = value
    json.dump(cur, open(OUT, 'w'), indent=1)


# ----------------------------------------------------------------- calibration metrics
def brier(y, p):
    return float(np.mean((p - y) ** 2))


def reliability(y, p, bins=10):
    """Equal-width bins on predicted probability. Returns list of (mean_pred, frac_pos, n)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def ece(y, p, bins=10):
    rel = reliability(y, p, bins)
    n = sum(r[2] for r in rel)
    return float(sum(abs(a - b) * c for a, b, c in rel) / n)


def pooled(items, scores, net):
    """Flatten non-shocked nodes across scenarios: labels, scores, hop distance."""
    ys, ss, hs = [], [], []
    for (sev, loss), sc in zip(items, scores):
        m = sev == 0
        ys.append((loss >= M.THR)[m]); ss.append(sc[m]); hs.append(M.hop_dist(net, sev)[m])
    return np.concatenate(ys).astype(float), np.concatenate(ss), np.concatenate(hs)


# ----------------------------------------------------------------- part 1: non-random hiding
def part_hidden(n_train, n_test, epochs, seeds):
    out = {}
    for mode in ('tier', 'region'):
        out[mode] = {}
        for seed in seeds:
            res, *_ = M.run(0.7, seed, n_train=n_train, n_test=n_test, epochs=epochs, hide_mode=mode)
            out[mode][f'seed{seed}'] = res
            save('hidden', out)
    return out


# ----------------------------------------------------------------- part 2: richer features
def part_features(n_train, n_test, epochs, seeds):
    out = {}
    M.RICH = True
    try:
        for seed in seeds:
            res, *_ = M.run(0.7, seed, n_train=n_train, n_test=n_test, epochs=epochs)
            out[f'seed{seed}'] = res
            save('features', out)
    finally:
        M.RICH = False
    return out


# ----------------------------------------------------------------- part 3: calibration + decision layer
def decision_experiment(te_net, scorers, n_cand, k_list, n_scen, seed, mitigation=0.8):
    """Choose k upstream suppliers to harden, then replay fresh shock scenarios.

    Hardening a supplier removes `mitigation` of any shock it receives (think dual-sourcing or a
    safety-stock top-up). The metric is the expected number of end-use (tier 0) industries that
    lose >= 25% of capacity per scenario, relative to doing nothing.
    """
    rng = np.random.default_rng(5000 + seed)
    cand = rng.choice(np.where(te_net['tiers'] >= 2)[0], n_cand, replace=False)
    scen = M.make_set(te_net, n_scen, 7000 + seed)
    tier0 = te_net['tiers'] == 0

    def impact(hardened):
        h = np.zeros(te_net['N'], bool); h[list(hardened)] = True
        tot = 0.0
        for sev, _ in scen:
            s2 = sev.copy(); s2[h] *= (1 - mitigation)
            loss = scsim.cascade(te_net, s2)
            tot += ((loss >= M.THR) & tier0).sum()
        return tot / len(scen)

    base = impact([])
    # true single-shock criticality (an oracle that cannot be known in practice)
    single = []
    for j in cand:
        sev = np.zeros(te_net['N']); sev[j] = 1.0
        single.append(((scsim.cascade(te_net, sev) >= M.THR) & tier0).sum())
    single = np.array(single, float)

    scores = {'oracle (true single-shock impact)': single}
    for name, fn in scorers.items():
        scores[name] = fn(cand)
    res = {'baseline_tier0_hit': base, 'n_candidates': int(n_cand), 'n_scenarios': len(scen), 'by_k': {}}
    for k in k_list:
        row = {}
        for name, sc in scores.items():
            chosen = cand[np.argsort(-sc, kind='stable')[:k]]
            row[name] = 1 - impact(chosen) / base
        rr = []
        for r in range(5):  # random choice, averaged over draws
            rr.append(1 - impact(np.random.default_rng(r + 31 * seed).choice(cand, k, replace=False)) / base)
        row['random'] = float(np.mean(rr))
        res['by_k'][str(k)] = row
    return res


def part_calib(n_train, n_test, epochs, seeds, n_members, n_cand, k_list, n_scen):
    out = {}
    for seed in seeds:
        t0 = time.time()
        tr_net = scsim.make_network(100 + seed, 0.7); te_net = scsim.make_network(200 + seed, 0.7)
        tr = M.make_set(tr_net, n_train, 1000 + seed); te = M.make_set(te_net, n_test, 2000 + seed)
        Xtr = np.concatenate([M.tabular_feats(tr_net, s) for s, _ in tr]); ytr = np.concatenate([(l >= M.THR) for _, l in tr])
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(Xtr, ytr)
        gb_scores = [gb.predict_proba(M.tabular_feats(te_net, s))[:, 1] for s, _ in te]

        members = {'sage': [], 'gat': []}
        member_preds = {'sage': [], 'gat': []}
        for kind in members:
            for m_i in range(n_members):
                m = M.train_gnn(kind, tr_net, tr, epochs=epochs, seed=seed * 100 + m_i)
                members[kind].append(m); member_preds[kind].append(M.predict_gnn(m, te_net, te))
            print(f'  seed {seed} {kind} ensemble trained ({time.time()-t0:.0f}s)', flush=True)

        seed_res = {}
        y, _, _ = pooled(te, gb_scores, te_net)

        def summarize(name, score_list):
            _, s, h = pooled(te, score_list, te_net)
            return {'ap': float(average_precision_score(y, s)), 'auroc': float(roc_auc_score(y, s)),
                    'brier': brier(y, s), 'ece': ece(y, s), 'reliability': reliability(y, s)}

        seed_res['tabular_gbm'] = summarize('gbm', gb_scores)
        for kind in members:
            single = member_preds[kind][0]
            ens = [np.mean([mp[i] for mp in member_preds[kind]], 0) for i in range(len(te))]
            seed_res[f'{kind}_single'] = summarize(kind, single)
            seed_res[f'{kind}_ensemble'] = summarize(kind, ens)

        # ---- decision layer: criticality scorers built from the trained models
        def single_shock_items(cand):
            its = []
            for j in cand:
                sev = np.zeros(te_net['N']); sev[j] = 1.0
                its.append((sev, np.zeros(te_net['N'])))
            return its

        def gnn_scorer(kind):
            def fn(cand):
                its = single_shock_items(cand)
                P = [M.predict_gnn(m, te_net, its) for m in members[kind]]
                ens = [np.mean([p[i] for p in P], 0) for i in range(len(its))]
                tier0 = te_net['tiers'] == 0
                return np.array([e[(s == 0) & tier0].sum() for e, (s, _) in zip(ens, its)])
            return fn

        def gbm_scorer(cand):
            tier0 = te_net['tiers'] == 0
            return np.array([gb.predict_proba(M.tabular_feats(te_net, s))[:, 1][(s == 0) & tier0].sum()
                             for s, _ in single_shock_items(cand)])

        import run_experiments as RE
        outdeg = np.bincount(te_net['obs_src'], minlength=te_net['N']).astype(float)
        reach = RE.reach_counts(te_net)
        scorers = {
            'observed out-degree': lambda c: outdeg[c],
            'observed reach (descendants)': lambda c: reach[c],
            'size x out-degree': lambda c: (te_net['size'] * outdeg)[c],
            'tabular GBM': gbm_scorer,
            'GraphSAGE ensemble': gnn_scorer('sage'),
            'GAT ensemble': gnn_scorer('gat'),
        }
        seed_res['decision'] = decision_experiment(te_net, scorers, n_cand, k_list, n_scen, seed)
        out[f'seed{seed}'] = seed_res
        save('calibration_decision', out)
        print(f'seed {seed} calibration + decision done in {time.time()-t0:.0f}s', flush=True)
    return out


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    full = dict(n_train=360, n_test=120, epochs=30)
    if part == 'smoke':
        OUT = 'results_extensions_smoke.json'
        tiny = dict(n_train=40, n_test=16, epochs=2)
        part_hidden(seeds=(0,), **tiny)
        part_features(seeds=(0,), **tiny)
        part_calib(seeds=(0,), n_members=2, n_cand=20, k_list=(3,), n_scen=10, **tiny)
        print('smoke ok'); sys.exit()
    if part in ('hidden', 'all'):
        part_hidden(seeds=SEEDS, **full)
    if part in ('features', 'all'):
        part_features(seeds=SEEDS, **full)
    if part in ('calib', 'all'):
        part_calib(seeds=SEEDS, n_members=N_MEMBERS, n_cand=600, k_list=(10, 25, 50, 100), n_scen=100, **full)
