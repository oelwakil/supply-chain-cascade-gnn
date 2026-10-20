import json, sys, numpy as np, torch
from scipy.stats import spearmanr
import scsim, models as exp

def reach_counts(net):
    """observed-graph descendant count per node (naive 'how far downstream can this reach?' baseline)"""
    N = net['N']; adj = [[] for _ in range(N)]
    for s, d in zip(net['obs_src'], net['obs_dst']): adj[s].append(d)
    out = np.zeros(N)
    for j in range(N):
        seen = {j}; st = [j]
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen: seen.add(v); st.append(v)
        out[j] = len(seen) - 1
    return out

def criticality(net, models, gb, n_nodes=150, seed=0):
    rng = np.random.default_rng(seed)
    cand = np.where(net['tiers'] >= 2)[0]
    nodes = rng.choice(cand, n_nodes, replace=False)
    items = []
    for j in nodes:
        sev = np.zeros(net['N']); sev[j] = 1.0
        items.append((sev, scsim.cascade(net, sev)))
    true = np.array([((l >= exp.THR) & (s == 0)).sum() for s, l in items], float)
    res = {}
    outdeg = np.bincount(net['obs_src'], minlength=net['N']).astype(float)
    reach = reach_counts(net)
    base = {'observed out-degree': outdeg[nodes], 'observed reach (descendants)': reach[nodes],
            'size x out-degree (gravity-style)': (net['size'] * outdeg)[nodes]}
    base['tabular GBM (1-hop features)'] = np.array([gb.predict_proba(exp.tabular_feats(net, s))[:, 1][s == 0].sum() for s, _ in items])
    for k, m in models.items():
        P = exp.predict_gnn(m, net, items)
        base['GraphSAGE' if k == 'sage' else 'GAT'] = np.array([p[s == 0].sum() for p, (s, _) in zip(P, items)])
    top10 = set(np.argsort(-true)[:10])
    for name, sc in base.items():
        res[name] = dict(spearman=float(spearmanr(sc, true).statistic),
                         top10_overlap=len(top10 & set(np.argsort(-sc)[:10])) / 10)
    res['_n_nodes'] = n_nodes; res['_true_mean'] = float(true.mean()); res['_true_max'] = float(true.max())
    return res

if __name__ == '__main__':
    configs = [(0.7, 0), (0.7, 1), (0.7, 2), (1.0, 0), (0.5, 0)]
    allres = {}
    for obs, seed in configs:
        res, models, te_net, gb = exp.run(obs, seed)
        res['_criticality'] = criticality(te_net, models, gb, seed=seed)
        allres[f'obs{obs}_seed{seed}'] = res
        json.dump(allres, open('results.json', 'w'), indent=1)
        print('saved', obs, seed, flush=True)
