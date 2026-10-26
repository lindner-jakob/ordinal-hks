#!/usr/bin/env python3
"""Re-simulate seeded runs and require exact agreement of integer stopping times.

Default: a small deterministic cross-section. --all-sweeps replays all 5,400
rank/density runs. Scaling can be regenerated in full with its separate runner.
The archived outcomes are used only for comparison, never to drive dynamics.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay_sweeps(all_runs, output):
    children = np.random.SeedSequence(20260911).spawn(1000)
    rhos = [1,5,10,15,20,25,30,35,40,45]
    expected_seeds = {(rho,t+1): int(children[i*100+t].generate_state(1,dtype=np.uint64)[0])
                      for i,rho in enumerate(rhos) for t in range(100)}
    with subprocess.Popen([str(ROOT/'build'/'replay_sweeps')],stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,text=True) as simulator:
        for stem in ['hks_fig3_ksweep_n10000','hks_fig4_density_sweep_regular']:
            rows = read_csv(ROOT/'data'/'sweeps'/(stem+'_raw_100rep.csv'))
            density = 'rho' in rows[0]
            if density:
                for row in rows:
                    if int(row['profile_seed']) != expected_seeds[int(row['rho']),int(row['trial'])]:
                        raise ValueError('density profile seed differs from original SeedSequence')
            if not all_runs:
                rows = [r for r in rows if
                        (density and int(r['trial']) == 1 and int(r['rho']) in [1,10,20,35,45])
                        or (not density and int(r['trial']) in [1,100]
                            and int(r['k']) in [1,8,10,25,50])]
            rows.sort(key=lambda r:(int(r.get('rho',10)),int(r['trial']),int(r['k'])))
            graph_key, graph_text = None, None
            reproduced = []
            for index,row in enumerate(rows,1):
                n,rho,seed = int(row['n']),int(row.get('rho',10)),int(row['profile_seed'])
                key = n,rho,seed
                if key != graph_key:
                    positions = np.sort(np.random.default_rng(seed).uniform(0.0,n/rho,n))
                    left = np.searchsorted(positions,positions-1.0,side='left')
                    right = np.searchsorted(positions,positions+1.0,side='right')-1
                    graph_text = ''.join(f'{a} {b}\n' for a,b in zip(left,right))
                    graph_key = key
                code = 0 if row.get('rule') == 'lower_median' else 1
                simulator.stdin.write(f"{n} {code} {row['k']} {row['activation_seed']}\n"+graph_text)
                simulator.stdin.flush()
                answer = simulator.stdout.readline()
                if not answer:
                    raise RuntimeError('simulation bridge terminated unexpectedly')
                steps = int(answer)
                if steps != int(row['steps']):
                    raise ValueError(f"exact replay mismatch: {stem}, {row}; got {steps}")
                reproduced.append(dict(row,steps=str(steps),T_over_n=str(steps/n)))
                if index % 100 == 0:
                    print(f'{stem}: replayed {index}/{len(rows)}',flush=True)
            write_csv(output/(stem+'_replayed.csv'),reproduced)
            print(f'{stem}: {len(rows)} exact stopping-time matches',flush=True)
        simulator.stdin.close()
        if simulator.wait() != 0:
            raise RuntimeError('simulation bridge returned an error')


def replay_scaling(output):
    original = read_csv(ROOT/'data'/'scaling'/'hks_density_scaling_raw.csv')
    expected = {(int(r['n']),int(float(r['rho'])),int(r['trial']),r['rule']):r for r in original}
    jobs = [('250,1000','1,5,10,25','scaling_small'),('32000','25','scaling_large')]
    count = 0
    for ns,ks,name in jobs:
        destination = output/(name+'_replayed.csv')
        subprocess.run([str(ROOT/'build'/'hks_extended_experiments'),
                        '--design','density_scaling','--ns',ns,'--rhos','10,35',
                        '--ks',ks,'--reps','1','--output',str(destination)],check=True)
        for row in read_csv(destination):
            key = int(row['n']),int(float(row['rho'])),int(row['trial']),row['rule']
            archived = expected[key]
            for column in row:
                if row[column] != archived[column]:
                    raise ValueError(f'scaling replay mismatch: {key}, {column}')
            count += 1
    print(f'Scaling: {count} exact complete-row matches, including n=32,000',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-sweeps',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'replay')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    replay_sweeps(args.all_sweeps,args.output_dir)
    replay_scaling(args.output_dir)


if __name__ == '__main__':
    main()
