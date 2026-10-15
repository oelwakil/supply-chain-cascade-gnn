import json, numpy as np
R = json.load(open('results.json'))
order = ['gravity', 'traffic_light', 'tabular_gbm', 'sage', 'gat']
labels = {'gravity': 'Gravity', 'traffic_light': 'Traffic light', 'tabular_gbm': 'Gradient boosting', 'sage': 'GraphSAGE', 'gat': 'Graph attention'}
main_keys = [k for k in R if k.startswith('obs0.7')]
def ms(vals): return [float(np.mean(vals)), float(np.std(vals))]
S = {'order': order, 'labels': labels, 'n_seeds': len(main_keys)}
S['main'] = {m: {k: ms([R[r][m][k] for r in main_keys]) for k in ('ap', 'auroc', 'ap_tier0', 'ap_indirect')} for m in order}
S['prev'] = float(np.mean([R[r]['gravity']['prev'] for r in main_keys]))
S['prev_indirect'] = float(np.mean([R[r]['gravity']['prev_indirect'] for r in main_keys]))
S['maxsd_ap'] = max(S['main'][m]['ap'][1] for m in order)
S['obs'] = {}
for m in order:
    S['obs'][m] = {'0.5': R['obs0.5_seed0'][m]['ap'], '0.7': R['obs0.7_seed0'][m]['ap'], '1.0': R['obs1.0_seed0'][m]['ap']}
S['obs_ind'] = {m: {'0.5': R['obs0.5_seed0'][m]['ap_indirect'], '0.7': R['obs0.7_seed0'][m]['ap_indirect'], '1.0': R['obs1.0_seed0'][m]['ap_indirect']} for m in order}
crit_names = [k for k in R[main_keys[0]]['_criticality'] if not k.startswith('_')]
S['crit'] = {n: {'spearman': ms([R[r]['_criticality'][n]['spearman'] for r in main_keys]), 'top10': ms([R[r]['_criticality'][n]['top10_overlap'] for r in main_keys])} for n in crit_names}
S['crit_order'] = sorted(crit_names, key=lambda n: -S['crit'][n]['spearman'][0])
S['cascade'] = {'impacted': 142, 'indirect': 102, 'total': 1500, 'tier0': 25, 'tier0total': 120}
ib = max(S['main'][m]['ap_indirect'][0] for m in ('gravity','traffic_light','tabular_gbm'))
S['takeaway12'] = (f"Models that see only direct links nearly miss indirect effects (AP {ib:.2f} at best, vs {S['main']['sage']['ap_indirect'][0]:.2f} to {S['main']['gat']['ap_indirect'][0]:.2f} for the graph models). "
  f"Hiding links hurts every model, and the gap narrows at 50% observed ({S['obs']['gat']['0.5']:.2f} vs {max(S['obs'][m]['0.5'] for m in ('traffic_light','tabular_gbm')):.2f}) but does not close.")
sp = [S['crit'][n]['spearman'][0] for n in crit_names]
t10 = lambda n: S['crit'][n]['top10'][0] * 10
S['takeaway13'] = (f"All six methods land within {max(sp)-min(sp):.2f} on rank correlation ({min(sp):.2f} to {max(sp):.2f}), so graph structure does most of the work. "
  f"The GNNs are slightly ahead on top-10 overlap ({t10('GraphSAGE'):.1f} and {t10('GAT'):.1f} of 10, vs {t10('observed reach (descendants)'):.1f} for plain reachability): a small edge over three seeds.")
json.dump(S, open('summary.json', 'w'), indent=1)
print(json.dumps({k: S[k] for k in ('main', 'obs', 'obs_ind', 'crit', 'crit_order', 'prev', 'prev_indirect')}, indent=1))
