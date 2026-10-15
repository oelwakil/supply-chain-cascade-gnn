import numpy as np, json, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import spearmanr
from torch_geometric.nn import SAGEConv, GATConv
import scsim
torch.set_num_threads(2)
THR = 0.25

def make_set(net, n, seed):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        sev = scsim.sample_shock(net, rng)
        if (sev > 0).sum() == 0: continue
        loss = scsim.cascade(net, sev)
        out.append((sev, loss))
    return out

def hop_dist(net, sev):
    """true hop distance (downstream) from nearest shocked node"""
    N = net['N']; d = np.full(N, -1); q = deque()
    for j in np.where(sev > 0)[0]: d[j] = 0; q.append(j)
    adj = [[] for _ in range(N)]
    for j, i in zip(net['src'], net['dst']): adj[j].append(i)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] < 0: d[v] = d[u] + 1; q.append(v)
    return d

# ---------- baselines ----------
def gravity_score(net, sev):
    reg = net['region']; sh = np.where(sev > 0)[0]
    if len(sh) == 0: return np.zeros(net['N'])
    dist = np.abs(reg[:, None] - reg[sh][None, :]); dist = np.minimum(dist, scsim.N_REGIONS - dist)
    return (net['size'][:, None] * (sev[sh] * net['size'][sh])[None, :] / (1 + dist) ** 2).sum(1)

def traffic_light_score(net, sev):
    """direct (tier-1) exposure on OBSERVED network: severity-weighted share of observed suppliers disrupted"""
    N = net['N']; tot = np.zeros(N); hit = np.zeros(N)
    np.add.at(tot, net['obs_dst'], 1.0); np.add.at(hit, net['obs_dst'], sev[net['obs_src']])
    return np.where(tot > 0, hit / np.maximum(tot, 1), 0)

def tabular_feats(net, sev):
    X = scsim.node_features(net, sev); N = net['N']
    tot = np.zeros(N); hit = np.zeros(N); mx = np.zeros(N)
    np.add.at(tot, net['obs_dst'], 1.0); np.add.at(hit, net['obs_dst'], sev[net['obs_src']])
    np.maximum.at(mx, net['obs_dst'], sev[net['obs_src']])
    return np.concatenate([X, (hit / np.maximum(tot, 1))[:, None], mx[:, None], tot[:, None]], 1)

# ---------- GNNs ----------
class GNN(nn.Module):
    def __init__(s, kind, fin, hid=64, L=4):
        super().__init__(); s.kind = kind
        s.inp = nn.Linear(fin, hid)
        s.convs = nn.ModuleList()
        for _ in range(L):
            if kind == 'sage': s.convs.append(SAGEConv(hid, hid, aggr='mean'))
            else: s.convs.append(GATConv(hid, hid // 4, heads=4, add_self_loops=True))
        s.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(L)])
        s.out = nn.Linear(hid, 1)
    def forward(s, x, ei, return_att=False):
        h = F.relu(s.inp(x)); att = None
        for k, (c, n) in enumerate(zip(s.convs, s.norms)):
            if return_att and s.kind == 'gat' and k == len(s.convs) - 1:
                hn, att = c(h, ei, return_attention_weights=True)
            else: hn = c(h, ei)
            h = h + F.relu(n(hn))   # residual keeps node's own state
        o = s.out(h).squeeze(-1)
        return (o, att) if return_att else o

def batchify(net, items, idx):
    xs, eis, ys, off = [], [], [], 0
    ei0 = np.stack([net['obs_src'], net['obs_dst']])
    for k in idx:
        sev, loss = items[k]
        xs.append(scsim.node_features(net, sev)); eis.append(ei0 + off); ys.append((loss >= THR).astype(np.float32)); off += net['N']
    ms = np.concatenate([(items[k][0] > 0) for k in idx])
    return (torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(eis, 1), dtype=torch.long),
            torch.tensor(np.concatenate(ys)), torch.tensor(ms))

def train_gnn(kind, net, items, L=4, epochs=30, bs=16, seed=0, hid=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    fin = scsim.node_features(net, items[0][0]).shape[1]
    m = GNN(kind, fin, hid, L); opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    n = len(items)
    for ep in range(epochs):
        perm = rng.permutation(n); m.train()
        for b in range(0, n, bs):
            x, ei, y, ms = batchify(net, items, perm[b:b + bs])
            lo = m(x, ei); loss = F.binary_cross_entropy_with_logits(lo[~ms], y[~ms])
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def predict_gnn(m, net, items, bs=32):
    m.eval(); out = []
    for b in range(0, len(items), bs):
        idx = list(range(b, min(b + bs, len(items))))
        x, ei, y, ms = batchify(net, items, idx)
        p = torch.sigmoid(m(x, ei)).numpy().reshape(len(idx), net['N'])
        out.extend(list(p))
    return out

# ---------- evaluation ----------
def evaluate(net, items, scores):
    """metrics over non-shocked nodes; by-hop AUROC on true hop distance; tier-0 AP"""
    ys, ss, hs, ts = [], [], [], []
    for (sev, loss), sc in zip(items, scores):
        m = sev == 0
        ys.append((loss >= THR)[m]); ss.append(sc[m]); hs.append(hop_dist(net, sev)[m]); ts.append(net['tiers'][m])
    y = np.concatenate(ys); s = np.concatenate(ss); h = np.concatenate(hs); t = np.concatenate(ts)
    r = dict(auroc=roc_auc_score(y, s), ap=average_precision_score(y, s), prev=float(y.mean()))
    m0 = t == 0; r['ap_tier0'] = average_precision_score(y[m0], s[m0])
    mi = h != 1; r['ap_indirect'] = average_precision_score(y[mi], s[mi]); r['prev_indirect'] = float(y[mi].mean())
    for k in (1, 2, 3, 4):
        mk = (h == k) if k < 4 else (h >= 4)
        r[f'auroc_hop{k}'] = roc_auc_score(y[mk], s[mk]) if 0 < y[mk].sum() < mk.sum() else float('nan')
        r[f'n_hop{k}'] = int(mk.sum()); r[f'pos_hop{k}'] = int(y[mk].sum())
    return r

def run(obs_frac, seed, n_train=360, n_test=120, epochs=30, L=4):
    t0 = time.time()
    tr_net = scsim.make_network(100 + seed, obs_frac); te_net = scsim.make_network(200 + seed, obs_frac)  # test = unseen entities
    tr = make_set(tr_net, n_train, 1000 + seed); te = make_set(te_net, n_test, 2000 + seed)
    res = {}
    res['gravity'] = evaluate(te_net, te, [gravity_score(te_net, s) for s, _ in te])
    res['traffic_light'] = evaluate(te_net, te, [traffic_light_score(te_net, s) for s, _ in te])
    Xtr = np.concatenate([tabular_feats(tr_net, s) for s, _ in tr]); ytr = np.concatenate([(l >= THR) for _, l in tr])
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(Xtr, ytr)
    res['tabular_gbm'] = evaluate(te_net, te, [gb.predict_proba(tabular_feats(te_net, s))[:, 1] for s, _ in te])
    models = {}
    for kind in ('sage', 'gat'):
        m = train_gnn(kind, tr_net, tr, L=L, epochs=epochs, seed=seed); models[kind] = m
        res[kind] = evaluate(te_net, te, predict_gnn(m, te_net, te))
    print(f'obs={obs_frac} seed={seed} done {time.time()-t0:.0f}s', {k: round(v['ap'], 3) for k, v in res.items()}, flush=True)
    return res, models, te_net, gb

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'quick':
        res, *_ = run(0.7, 0, n_train=120, n_test=40, epochs=8)
        print(json.dumps({k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in res.items()}, indent=1))
    if mode == 'one':
        res, *_ = run(0.7, 0)
        json.dump(res, open('one.json', 'w'), indent=1)
