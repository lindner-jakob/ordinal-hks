#!/usr/bin/env python3
"""Validate the fine grid and estimate means, doubling exponents and tail fits.

Bootstrap resampling is stratified by (n,rho). All rules on the same initial
profile use the same resampled trial indices. Intervals quantify Monte Carlo
sampling uncertainty and do not account for a change of scaling beyond the
observed sizes. Only the Python standard library is required.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics as st
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RULES = ['lower_median', 'max', '5-large', '10-large', '25-large']
METRICS = ['T_over_n', 'final_opinions', 'final_components',
           'largest_final_opinion_fraction', 'initial_components',
           'cut_crossings', 'metric_distance', 'moves_left', 'moves_right']


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (x - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def slope(ns, means):
    x = [math.log(n) for n in ns]
    y = [math.log(n * m) for n, m in zip(ns, means)]
    xb, yb = st.mean(x), st.mean(y)
    return sum((a-xb)*(b-yb) for a,b in zip(x,y)) / sum((a-xb)**2 for a in x)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT/'results'/'scaling')
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    directory = args.data_dir
    manifest = json.loads((directory/'manifest.json').read_text())
    ns, rhos = manifest['n_values'], manifest['rho_values']
    with (directory/'hks_density_scaling_raw.csv').open(newline='') as handle:
        raw = list(csv.DictReader(handle))
    profiles = defaultdict(dict)
    for row in raw:
        n, rho, trial = int(row['n']), float(row['rho']), int(row['trial'])
        rule = row['rule']
        key = (n,rho,trial)
        if rule in profiles[key]: raise ValueError(f'duplicate row {key}/{rule}')
        if int(row['truncated']): raise ValueError(f'truncated run {key}/{rule}')
        if int(row['steps']) != int(row['moves_left']) + int(row['moves_right']):
            raise ValueError('movement count mismatch')
        if int(row['initial_ell']) != n: raise ValueError('support mismatch')
        if abs(float(row['T_over_n'])-int(row['steps'])/n) > 1e-8:
            raise ValueError('runtime normalization mismatch')
        if float(row['metric_distance']) > int(row['steps']) + 1e-5:
            raise ValueError('confidence-radius travel bound violated')
        if rule == 'max' and int(row['moves_left']):
            raise ValueError('max monotonicity violated')
        profiles[key][rule] = row

    groups = {}
    for n in ns:
        for rho in rhos:
            count = manifest['repetitions_by_n'][str(n)]
            group = [profiles[(n,rho,t)] for t in range(1,count+1)]
            for trials in group:
                if set(trials) != set(RULES): raise ValueError('missing rules')
                if len({r['profile_seed'] for r in trials.values()}) != 1:
                    raise ValueError('unpaired initial profiles')
            groups[(n,rho)] = group
    expected = sum(manifest['repetitions_by_n'][str(n)] for n in ns)*len(rhos)*len(RULES)
    if len(raw) != expected: raise ValueError('grid row count mismatch')

    rng = random.Random(20261008)
    means, boot, summaries = {}, {}, []
    for (n,rho), group in sorted(groups.items()):
        count = len(group)
        values = {rule: [float(g[rule]['T_over_n']) for g in group] for rule in RULES}
        samples = {rule: [] for rule in RULES}
        for b in range(args.bootstrap):
            indices = [rng.randrange(count) for _ in range(count)]
            for rule in RULES:
                samples[rule].append(sum(values[rule][i] for i in indices)/count)
        for rule in RULES:
            key = (n,rho,rule)
            mean = st.mean(values[rule])
            means[key], boot[key] = mean, samples[rule]
            item = dict(n=n,rho=rho,rule=rule,repetitions=count,
                        mean_T_over_n=mean,sd_T_over_n=st.stdev(values[rule]),
                        ci95_low_T_over_n=quantile(samples[rule],.025),
                        ci95_high_T_over_n=quantile(samples[rule],.975),
                        median_T_over_n=st.median(values[rule]),
                        q05_T_over_n=quantile(values[rule],.05),
                        q95_T_over_n=quantile(values[rule],.95),
                        mean_T_over_n2=mean/n,
                        mean_distance_over_n2=st.mean(float(g[rule]['metric_distance'])/(n*n) for g in group),
                        mean_jump_length=st.mean(float(g[rule]['metric_distance'])/int(g[rule]['steps']) for g in group),
                        connected_fraction=st.mean(int(g[rule]['consensus_possible']) for g in group))
            for metric in METRICS[1:]:
                item['mean_'+metric] = st.mean(float(g[rule][metric]) for g in group)
            summaries.append(item)
    write_csv(directory/'hks_density_scaling_summary.csv',summaries)

    doubling = []
    fits = []
    for rho in rhos:
        for rule in RULES:
            for a,b in zip(ns,ns[1:]):
                if b != 2*a: raise ValueError('size sequence is not doubling')
                alpha = 1+math.log2(means[(b,rho,rule)]/means[(a,rho,rule)])
                samples = [1+math.log2(y/x) for x,y in zip(boot[(a,rho,rule)],boot[(b,rho,rule)])]
                doubling.append(dict(rho=rho,rule=rule,n_from=a,n_to=b,alpha=alpha,
                                     ci95_low=quantile(samples,.025),ci95_high=quantile(samples,.975)))
            for nmin in [250,2000,8000]:
                sizes = [n for n in ns if n >= nmin]
                alpha = slope(sizes,[means[(n,rho,rule)] for n in sizes])
                samples = [slope(sizes,[boot[(n,rho,rule)][i] for n in sizes])
                           for i in range(args.bootstrap)]
                fits.append(dict(rho=rho,rule=rule,n_min=nmin,n_max=max(sizes),
                                 points=len(sizes),alpha=alpha,ci95_low=quantile(samples,.025),
                                 ci95_high=quantile(samples,.975)))
    write_csv(directory/'hks_density_scaling_doubling_exponents.csv',doubling)
    write_csv(directory/'hks_density_scaling_fitted_exponents.csv',fits)
    # A descriptive location of the linear-to-quadratic crossover. This is
    # defined on each observed doubling interval, not as a critical density.
    crossovers = []
    for b in ns[1:]:
        for rule in ['max','5-large']:
            rows = sorted([r for r in doubling if r['n_to']==b and r['rule']==rule],
                          key=lambda r:r['rho'])
            crossing, status = None, 'above_grid'
            if rows[0]['alpha'] >= 1.5:
                status = 'below_grid'
            else:
                for a,c in zip(rows,rows[1:]):
                    if a['alpha'] < 1.5 <= c['alpha']:
                        crossing = a['rho']+(c['rho']-a['rho'])*(1.5-a['alpha'])/(c['alpha']-a['alpha'])
                        status = 'interpolated'
                        break
            crossovers.append(dict(n_from=b//2,n_to=b,rule=rule,
                                   target_alpha=1.5,rho_crossing=crossing,status=status))
    write_csv(directory/'hks_density_scaling_crossovers.csv',crossovers)
    manifest['analysis_bootstrap_repetitions'] = args.bootstrap
    manifest['analysis_seed'] = 20261008
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    print(f'validated {len(raw)} runs in {len(summaries)} groups; no truncations')
    print('Tail exponents: fit over n=8000,16000,32000; percentile bootstrap CI')
    for row in fits:
        if row['n_min']==8000 and row['rule'] in ['max','5-large','10-large']:
            print(f"rho={row['rho']:g} {row['rule']:8s} {row['alpha']:.3f} "
                  f"[{row['ci95_low']:.3f},{row['ci95_high']:.3f}]")


if __name__=='__main__':
    main()
