"""Synthetic multi-tier supply network + shock-cascade simulator.

Purely synthetic. Used to show *why* graph models help on nth-tier cascades, not to
claim performance on real trade data.

Tiers: 0 = end-use critical industries (what policymakers care about) ... 4 = raw/upstream.
Edges point supplier -> buyer (material flows downstream, tier k+1 -> tier k, with some tier skipping).
Dynamics: Leontief-style bottleneck by input category with within-category substitution
and inventory buffers. Shocks propagate downstream through the TRUE network.
The model only sees an OBSERVED network (a fraction of true edges are hidden = opaque topology).
"""
import numpy as np

TIER_SIZES = [120, 260, 380, 460, 280]  # tier 0..4  -> 1500 nodes
N_SECTORS = 6
N_REGIONS = 8


def observation_probs(tiers, region, src, dst, obs_frac, mode, seed):
    """Per-edge probability that a true link is observed.

    mode='random'  : every link is observed with probability obs_frac (original behaviour).
    mode='tier'    : links deeper in the chain (buyer in tier 2+) are less visible, as in real
                     supply networks where tier-1 relationships are known and sub-tier ones are not.
    mode='region'  : links whose supplier sits in a few 'opaque' regions are mostly hidden.
    Non-random modes are rescaled so the average observed fraction is still obs_frac, which keeps
    comparisons against the random mode fair.
    """
    if mode == 'random':
        return np.full(len(src), obs_frac)
    if mode == 'tier':
        rel = np.array([1.5, 1.3, 1.0, 0.6, 0.4])[tiers[dst]]
    elif mode == 'region':
        r = np.random.default_rng(seed + 999)
        opaque = r.choice(N_REGIONS, 3, replace=False)
        rel = np.where(np.isin(region[src], opaque), 0.25, 1.4)
    else:
        raise ValueError(f'unknown hide mode: {mode}')
    p = rel * obs_frac / rel.mean()
    return np.clip(p, 0.02, 1.0)


def make_network(seed, obs_frac=0.7, hide_mode='random'):
    rng = np.random.default_rng(seed)
    tiers = np.concatenate([np.full(n, t) for t, n in enumerate(TIER_SIZES)])
    N = len(tiers)
    sector = rng.integers(0, N_SECTORS, N)
    region = rng.integers(0, N_REGIONS, N)
    # upstream tiers are more regionally concentrated (e.g. few countries dominate raw inputs)
    for t in (3, 4):
        idx = np.where(tiers == t)[0]
        pr = rng.dirichlet(np.ones(N_REGIONS) * 0.4)
        region[idx] = rng.choice(N_REGIONS, len(idx), p=pr)
    size = rng.lognormal(0, 1.0, N)  # heavy-tailed => hubs / single points of failure
    buffer = rng.uniform(0.0, 0.3, N)

    src, dst, w = [], [], []
    for i in range(N):
        t = tiers[i]
        if t == 4:
            continue
        n_cat = rng.integers(1, 4)  # distinct input categories needed
        cats = rng.choice(N_SECTORS, n_cat, replace=False)
        for s in cats:
            n_sup = rng.choice([1, 1, 2, 3], p=[0.35, 0.25, 0.25, 0.15])  # many single-sourced inputs
            # candidate suppliers: next tier (80%) or skip a tier (20%)
            tt = min(4, t + (1 if rng.random() < 0.8 else 2))
            cand = np.where((tiers == tt) & (sector == s))[0]
            if len(cand) == 0:
                cand = np.where(tiers == tt)[0]
            p = size[cand] ** 1.2  # preferential attachment to big suppliers
            p /= p.sum()
            sup = rng.choice(cand, min(n_sup, len(cand)), replace=False, p=p)
            ww = rng.dirichlet(np.ones(len(sup)) * 2.0)
            for j, wj in zip(sup, ww):
                src.append(j); dst.append(i); w.append(wj)
    src, dst, w = map(np.array, (src, dst, w))
    keep = rng.random(len(src)) < observation_probs(tiers, region, src, dst, obs_frac, hide_mode, seed)
    return dict(N=N, tiers=tiers, sector=sector, region=region, size=size, buffer=buffer,
                src=src, dst=dst, w=w, obs_src=src[keep], obs_dst=dst[keep], obs_mask=keep)


def sample_shock(net, rng):
    """Return per-node severity vector (0 = untouched, up to 1 = fully disrupted)."""
    N = net['N']; sev = np.zeros(N)
    kind = rng.choice(['node', 'region', 'regsec', 'multi'], p=[0.25, 0.25, 0.3, 0.2])
    deep = np.where(net['tiers'] >= 2)[0]
    if kind == 'node':
        # a single large upstream node -> single point of failure
        j = rng.choice(deep, p=net['size'][deep] / net['size'][deep].sum())
        sev[j] = rng.uniform(0.7, 1.0)
    elif kind == 'region':
        r = rng.integers(0, N_REGIONS)
        idx = np.where((net['tiers'] >= 3) & (net['region'] == r))[0]
        sev[idx] = rng.uniform(0.3, 0.7)
    elif kind == 'regsec':
        r = rng.integers(0, N_REGIONS); s = rng.integers(0, N_SECTORS)
        idx = deep[(net['region'][deep] == r) & (net['sector'][deep] == s)]
        sev[idx] = rng.uniform(0.6, 1.0)
    else:
        for _ in range(rng.integers(2, 5)):
            j = rng.choice(deep)
            sev[j] = rng.uniform(0.6, 1.0)
    return sev


def cascade(net, sev):
    """Propagate downstream on the TRUE network. Returns capacity loss in [0,1] per node."""
    N = net['N']; cap = np.ones(N)
    order = np.argsort(-net['tiers'], kind='stable')  # upstream first
    # group incoming edges by dst
    by_dst = [[] for _ in range(N)]
    for e, (j, i) in enumerate(zip(net['src'], net['dst'])):
        by_dst[i].append(e)
    sector = net['sector']
    for i in order:
        a_min = 1.0
        if by_dst[i]:
            cats = {}
            for e in by_dst[i]:
                cats.setdefault(sector[net['src'][e]], []).append(e)
            for s, es in cats.items():
                ws = net['w'][es]
                avail = (ws * cap[net['src'][es]]).sum() / ws.sum()  # within-category substitution
                a_min = min(a_min, avail)  # Leontief bottleneck across categories
            # inventory buffer absorbs a share of the shortfall
            a_min = min(1.0, a_min / (1 - net['buffer'][i])) if a_min < 1 else 1.0
        cap[i] = (1 - sev[i]) * a_min
    return 1 - cap


def node_features(net, sev, rich=False):
    """Features visible to a model: attributes + shock severity (no cascade info, no hidden edges).

    rich=True adds what a real analyst could plausibly observe: observed in/out degree and a noisy
    reading of each firm's inventory cover (the true buffer plus measurement noise).
    """
    N = net['N']
    f = [np.log(net['size'])[:, None],
         np.eye(5)[net['tiers']], np.eye(N_SECTORS)[net['sector']], np.eye(N_REGIONS)[net['region']],
         sev[:, None]]
    if rich:
        indeg = np.bincount(net['obs_dst'], minlength=N).astype(float)
        outdeg = np.bincount(net['obs_src'], minlength=N).astype(float)
        noise = np.random.default_rng(int(net['size'].sum() * 1e6) % (2 ** 31)).normal(0, 0.1, N)
        f += [np.log1p(indeg)[:, None], np.log1p(outdeg)[:, None], (net['buffer'] + noise)[:, None]]
    return np.concatenate(f, 1).astype(np.float32)
