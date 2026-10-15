# Supply Chain Cascade Testbed: Graph Neural Networks vs. Direct-Exposure Baselines

A synthetic experiment that asks one question: **when a shock hits a multi-tier supply network, can a graph neural network predict which entities lose capacity, including entities several tiers downstream, better than scoring approaches that only look at direct exposure?**

Everything here is synthetic. There is no real trade, company, or government data. The results show that the approach can learn cascade structure and generalize to an unseen network. They do not show how well it would perform on real supply chains.

## Why this exists

Tracing how a disruption propagates through nth-tier supplier dependencies is hard because the effects are indirect, delayed, and hidden by incomplete knowledge of who supplies whom. Simple scoring methods look at direct suppliers or at aggregate size. This testbed gives a controlled setting, with known ground truth, to compare those methods against graph learning.

## Results

Trained on 360 shock scenarios from one network and tested on 120 scenarios from a **different network the models never saw**, with 70% of the true supply links observed. Mean ± standard deviation over 3 seeds. AP is average precision. Random guessing scores 0.07 on AP overall and 0.04 on indirect effects.

| Model | AP, all entities | AP, end-use industries (tier 0) | AP, 2+ hops from shock | AUROC |
| --- | --- | --- | --- | --- |
| Gravity (size-based) | 0.18 ± 0.02 | 0.28 ± 0.04 | 0.11 ± 0.02 | 0.75 ± 0.02 |
| Traffic light (direct suppliers) | 0.34 ± 0.01 | 0.14 ± 0.03 | 0.04 ± 0.01 | 0.68 ± 0.01 |
| Gradient boosting (1-hop features) | 0.45 ± 0.02 | 0.16 ± 0.03 | 0.11 ± 0.02 | 0.84 ± 0.00 |
| GraphSAGE | 0.56 ± 0.03 | 0.42 ± 0.07 | 0.39 ± 0.05 | 0.89 ± 0.01 |
| Graph attention (GAT) | 0.59 ± 0.03 | 0.46 ± 0.04 | 0.44 ± 0.03 | 0.90 ± 0.00 |

**Hidden links.** The models see only the observed part of the network. Average precision as the share of observed links changes (one seed):

| Model | 50% observed | 70% observed | 100% observed |
| --- | --- | --- | --- |
| Traffic light | 0.27 | 0.34 | 0.46 |
| Gradient boosting | 0.35 | 0.42 | 0.54 |
| GraphSAGE | 0.42 | 0.51 | 0.73 |
| Graph attention | 0.43 | 0.56 | 0.77 |

Every model degrades as links are hidden. The graph models stay ahead at every level, and the gap narrows at 50% but does not close.

**Single points of failure.** For 150 upstream suppliers per seed, the true downstream impact was computed by simulation, and each method ranked the suppliers. This is a negative result worth reporting:

| Ranking method | Spearman with true impact | Top-10 overlap (of 10) |
| --- | --- | --- |
| Observed reach (count of descendants) | 0.86 | 8.0 |
| Observed out-degree | 0.86 | 7.3 |
| GraphSAGE | 0.85 | 9.0 |
| Graph attention | 0.85 | 8.7 |
| Gradient boosting | 0.85 | 7.3 |
| Size x out-degree (gravity-style) | 0.84 | 6.0 |

All methods land within 0.02 on rank correlation, so network structure does most of the work for this question. A plain reachability count is a strong baseline. The graph models have a small edge on top-10 overlap, which is modest over three seeds. The learned models earn their keep on the other task, predicting which specific entities are hit under a given shock, especially indirect ones.

Also note that GAT's lead over GraphSAGE is small, within what tuning could change. Treat the two as comparable.

## How the simulator works

`scsim.py` builds a five-tier network of 1,500 entities (tier 0 is end-use industries, tier 4 is raw and upstream inputs) with 6 sectors and 8 regions.

- **Structure.** Each entity needs inputs from 1 to 3 categories. Suppliers are chosen by preferential attachment to large suppliers, which creates hubs and single points of failure. Many inputs are single-sourced. Upstream tiers are regionally concentrated.
- **Shocks.** Four types: one large upstream supplier, a regional event, a regional event within one sector, and several simultaneous supplier failures.
- **Cascade.** A Leontief-style bottleneck: an entity's output is limited by its scarcest input category. Within a category, suppliers can substitute for each other. An inventory buffer absorbs part of any shortfall. Losses propagate downstream through the true network.
- **Partial observation.** The models see only a random subset of the true links (default 70%), which stands in for opaque real-world topology.
- **Label.** An entity is positive if it loses at least 25% of capacity and was not directly shocked.

## Models compared

| Model | What it sees |
| --- | --- |
| Gravity | Entity size and shock severity, ignoring network structure |
| Traffic light | Direct (one-hop) supplier exposure |
| Gradient boosting | Tabular features, including one-hop neighborhood features (scikit-learn `HistGradientBoostingClassifier`) |
| GraphSAGE | Four layers, 64 hidden units, residual connections, layer normalization (PyTorch Geometric) |
| Graph attention (GAT) | Same architecture with attention-based aggregation |

All models are evaluated with average precision and AUROC. Average precision is the main metric because only about 7% of entities are positive.

## Reproduce

Python 3.10 or newer. A CPU is enough.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Quick smoke test (about 2 minutes): small data, few epochs
python exp.py quick

# 2. Full experiment (about 20 minutes on 2 CPU threads): 3 seeds at 70% observed,
#    plus 50% and 100% observed for seed 0. Writes results.json
python exp2.py

# 3. Summarize results.json into means and standard deviations. Writes summary.json
python summarize.py
```

The `results.json` in this repo is the output of the full run that produced the tables above.

Tested with Python 3.13, PyTorch 2.14, PyTorch Geometric 2.8, scikit-learn 1.9, NumPy 2.5, SciPy 1.18. If PyTorch Geometric fails to install, see its [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Files

| File | Purpose |
| --- | --- |
| `scsim.py` | Network generator, shock sampler, and cascade simulator |
| `exp.py` | Baseline scores, GNN models, training, and evaluation (`quick` and `one` modes) |
| `exp2.py` | Full experiment runner, including the single-point-of-failure ranking |
| `summarize.py` | Aggregates `results.json` across seeds |
| `results.json` | Raw results from the full run |

## Limitations

- **The simulator defines the cascade.** The models learn the simulator's rules. The results show feasibility and relative comparisons, not real-world accuracy.
- **Small scale.** 1,500 entities and 3 seeds. Standard deviations are reported, but the single-point-of-failure comparison and the GAT versus GraphSAGE gap are within noise.
- **Hidden links are random.** Real missing data is rarely random.
- **Features are simple.** Entity attributes are size, tier, sector, region, and shock severity.
- **No uncertainty estimates.** The models output point scores. Calibrated uncertainty would be the next step for decision use.

## Possible extensions

- Calibrated probabilities, for example ensembles or Bayesian last layers, with reliability diagrams.
- Link prediction for hidden edges, trained jointly with the cascade model.
- Combining the learned model with a structural or symbolic layer so predictions respect known constraints and can be explained along specific supply paths.
- A decision layer, such as a policy trained against the simulator to choose buffer or sourcing actions, with a heuristic baseline it must beat.
- Validation on a real or semi-real network.

## Development note

This code was developed with AI assistance (Claude). I specified the design, experiments, and evaluation, and reviewed the results.

## License

MIT. See `LICENSE`.
