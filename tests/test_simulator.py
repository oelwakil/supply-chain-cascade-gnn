"""Invariants the cascade simulator must satisfy. These need only numpy (no torch)."""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scsim  # noqa: E402


@pytest.fixture(scope="module")
def net():
    return scsim.make_network(seed=3, obs_frac=0.7)


def _downstream(net, sev):
    adj = [[] for _ in range(net["N"])]
    for s, d in zip(net["src"], net["dst"]):
        adj[s].append(d)
    seen = np.zeros(net["N"], bool)
    q = deque(np.where(sev > 0)[0])
    seen[sev > 0] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def test_no_shock_no_loss(net):
    assert np.allclose(scsim.cascade(net, np.zeros(net["N"])), 0.0)


def test_loss_is_a_valid_fraction(net):
    rng = np.random.default_rng(0)
    for _ in range(15):
        loss = scsim.cascade(net, scsim.sample_shock(net, rng))
        assert loss.min() >= -1e-12 and loss.max() <= 1 + 1e-12


def test_losses_only_reach_true_downstream_nodes(net):
    """A node with no shocked ancestor must be untouched (capacity is conserved elsewhere)."""
    rng = np.random.default_rng(1)
    for _ in range(15):
        sev = scsim.sample_shock(net, rng)
        loss = scsim.cascade(net, sev)
        reachable = _downstream(net, sev)
        assert np.allclose(loss[~reachable], 0.0)


def test_shocked_node_loses_at_least_its_severity(net):
    rng = np.random.default_rng(2)
    sev = scsim.sample_shock(net, rng)
    loss = scsim.cascade(net, sev)
    hit = sev > 0
    assert np.all(loss[hit] >= sev[hit] - 1e-9)


def test_larger_shock_never_reduces_loss(net):
    """Monotonicity: scaling severities up cannot make any node better off."""
    rng = np.random.default_rng(4)
    sev = scsim.sample_shock(net, rng)
    small = scsim.cascade(net, sev * 0.5)
    large = scsim.cascade(net, sev)
    assert np.all(large >= small - 1e-9)


def test_network_is_reproducible():
    a = scsim.make_network(11, 0.7)
    b = scsim.make_network(11, 0.7)
    assert np.array_equal(a["src"], b["src"]) and np.array_equal(a["obs_mask"], b["obs_mask"])


def test_observed_links_are_a_subset_of_true_links(net):
    true_edges = set(zip(net["src"].tolist(), net["dst"].tolist()))
    obs_edges = set(zip(net["obs_src"].tolist(), net["obs_dst"].tolist()))
    assert obs_edges <= true_edges
    assert 0.6 < len(obs_edges) / len(true_edges) < 0.8


@pytest.mark.parametrize("mode", ["random", "tier", "region"])
def test_hiding_modes_keep_the_same_average_visibility(mode):
    fracs = [scsim.make_network(s, 0.7, hide_mode=mode)["obs_mask"].mean() for s in range(5)]
    assert abs(np.mean(fracs) - 0.7) < 0.05


def test_tier_mode_hides_deep_links_more_than_shallow_ones():
    n = scsim.make_network(5, 0.7, hide_mode="tier")
    shallow = n["obs_mask"][n["tiers"][n["dst"]] <= 1].mean()
    deep = n["obs_mask"][n["tiers"][n["dst"]] >= 3].mean()
    assert shallow > deep + 0.2


def test_default_mode_matches_original_random_behaviour():
    """hide_mode='random' must draw exactly the uniform mask the first version used."""
    a = scsim.make_network(7, 0.7)
    b = scsim.make_network(7, 0.7, hide_mode="random")
    assert np.array_equal(a["obs_mask"], b["obs_mask"])


def test_unknown_hide_mode_raises():
    with pytest.raises(ValueError):
        scsim.make_network(1, 0.7, hide_mode="nope")
