"""Validate and plot a completed temperature sweep without counting shared samples twice."""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

HISTORIES = ['none', 'W', 'L', 'WW', 'WL', 'LW', 'LL']
COLORS = ['#64748b', '#60a5fa', '#fb923c', '#1d4ed8', '#7c3aed', '#0d9488', '#c2410c']
GROUPS = ['weaker', 'neutral', 'stronger']
TEMPS = [0.0, 1.0, 2.0]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8-sig'))
    assert manifest['finished'], f'Incomplete run: {run}'
    trials = read_csv(run / 'unique_contests.csv')
    ids = {r['sample_id'] for r in trials}
    assert len(ids) == len(trials), 'Duplicate samples'
    observed = defaultdict(dict)
    with (run / 'rounds.csv').open(encoding='utf-8-sig', newline='') as stream:
        for r in csv.DictReader(stream):
            assert r['sample_id'] in ids
            k = int(r['current_round'])
            assert not r['validation_error'], f'Invalid response: {r["sample_id"]}'
            assert observed[r['sample_id']].setdefault(k, r['action']) == r['action']
    for r in trials:
        assert r['completed'] == 'True'
        assert r['stop_reason'] in ('retreat', 'strength_below_5')
        actions = observed[r['sample_id']]
        assert len(actions) == int(r['rounds_observed'])
        assert [actions[k] for k in sorted(actions)] == ['fight'] * int(r['fight_rounds']) + (['retreat'] if r['stop_reason'] == 'retreat' else [])
    print(f'Validated {run.name}: {len(trials):,} unique contests', flush=True)
    return trials, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path)
    args = parser.parse_args()
    sweep = args.sweep or sorted((ROOT / 'data/control/temperature_sweeps').glob('*/sweep_status.json'))[-1].parent
    out = ROOT / 'output/temperature_sweeps' / sweep.name
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    stats = []
    provenance = {'source_sweep': str(sweep.resolve()), 'runs': []}
    for exp in (1, 2):
        for temp in TEMPS:
            run = sweep / f'temperature_{temp}_experiment_{exp}'
            rows, manifest = load(run)
            assert manifest['config']['contest_sequence_experiments']['temperature'] == temp
            runs[exp, temp] = rows
            provenance['runs'].append({'experiment': exp, 'temperature': temp, 'unique_contests': len(rows),
                                      'undelivered_experiences': sum(r['experience_delivered'] == 'False' for r in rows),
                                      'manifest': str((run / 'manifest.json').resolve())})
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(out / 'temperature_sweep_plots.pdf') as pdf:
        for exp in (1, 2):
            for kind in ['fighting', 'endings'] + (['presented_endings'] if exp == 1 else []):
                fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharey=True)
                fig.subplots_adjust(left=.07, right=.985, top=.88, bottom=.115, hspace=.42, wspace=.17)
                for row_i, temp in enumerate(TEMPS):
                    for col_i, group in enumerate(GROUPS):
                        ax = axes[row_i, col_i]
                        cells = {h: [r for r in runs[exp, temp] if r['opponent_group'] == group and r['history'] == h] for h in HISTORIES}
                        assert all(cells.values())
                        counts = [len(cells[h]) for h in HISTORIES]
                        if kind == 'fighting':
                            max_round = max(int(r['rounds_observed']) for r in runs[exp, temp])
                            for history, color in zip(HISTORIES, COLORS):
                                fractions = [sum(int(r['fight_rounds']) >= k for r in cells[history]) / len(cells[history]) for k in range(1, max_round + 1)]
                                ax.plot(range(1, max_round + 1), fractions, 'o-', color=color, lw=2, ms=3.5, label='None' if history == 'none' else history)
                                for k, value in enumerate(fractions, 1):
                                    stats.append([exp, temp, group, history, 'original_fight_participation', k, len(cells[history]), value])
                            ax.set_xticks(range(1, max_round + 1))
                            ax.set_xlabel('Contest round')
                            n_label = str(counts[0]) if len(set(counts)) == 1 else f'{min(counts)}–{max(counts)}'
                            ax.text(.97, .94, f'n = {n_label} per history', transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#475569')
                        else:
                            if kind == 'endings':
                                field, categories, colors, labels = 'stop_reason', ['retreat', 'strength_below_5'], ['#0f766e', '#cbd5e1'], ['Focal retreat', 'Strength below 5']
                            else:
                                field, categories, colors, labels = 'presented_stop_reason', ['opponent_retreat', 'retreat', 'strength_below_5'], ['#2563eb', '#0f766e', '#cbd5e1'], ['Constructed opponent retreat (win)', 'Focal retreat (loss)', 'Strength below 5 (loss)']
                            bottom = [0.] * len(HISTORIES)
                            for category, color, label in zip(categories, colors, labels):
                                values = [sum(r[field] == category for r in cells[h]) / len(cells[h]) for h in HISTORIES]
                                ax.bar(range(7), values, bottom=bottom, color=color, label=label)
                                bottom = [a+b for a,b in zip(bottom,values)]
                                for h, value in zip(HISTORIES, values):
                                    stats.append([exp, temp, group, h, f'{kind}_{category}', '', len(cells[h]), value])
                            assert all(abs(v-1) < 1e-9 for v in bottom)
                            ax.set_xticks(range(7), [('None' if h == 'none' else h) + f'\n(n={n})' for h,n in zip(HISTORIES,counts)], fontsize=8)
                            ax.set_xlabel('Previous history (oldest to newest)')
                        strength_label = {'weaker': '70–79', 'neutral': '80', 'stronger': '81–90'}[group]
                        ax.set_title(f'T = {temp} · {group.capitalize()} opponent ({strength_label})', fontsize=11)
                        ax.set_ylim(0, 1.07)
                        ax.yaxis.set_major_formatter(PercentFormatter(1))
                        ax.grid(axis='y', alpha=.18)
                        ax.set_axisbelow(True)
                        if col_i == 0:
                            ax.set_ylabel('Starting contests choosing fight' if kind == 'fighting' else 'Contests')
                title = {'fighting': 'Fighting across rounds by previous history', 'endings': 'Original contest endings', 'presented_endings': 'Outcomes presented to the model'}[kind]
                fig.suptitle(f'Experiment {exp}: {title}', fontsize=19, weight='bold', y=.975)
                fig.text(.5,.938, 'Experienced history with prescribed-win rollback' if exp == 1 else 'Supplied outcome history; conversation retained within each contest', ha='center', fontsize=11, color='#475569')
                handles, labels = axes[0,0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.04), ncol=len(labels), frameon=False)
                note = 'Unique sample IDs counted once. Experiment 1 includes both prescribed current outcomes; later histories reflect sequence attrition.' if exp == 1 else '100 contests per opponent group/history. Earlier outcomes are supplied labels.'
                if kind == 'fighting':
                    note += '\nDenominator: all starting contests in each history; ended contests contribute zero in later rounds.'
                elif kind == 'endings':
                    note += '\nOriginal stopping reasons before rollback; these are not the win/loss experiences presented to the model.'
                else:
                    note += '\nIncludes undelivered prescribed wins as the losses actually presented; blue represents constructed win feedback.'
                fig.text(.5,.015,note, ha='center', fontsize=8, color='#475569')
                fig.savefig(out / f'experiment_{exp}_{kind}.png', dpi=160, facecolor='white')
                pdf.savefig(fig)
                plt.close(fig)
    with (out / 'plotted_statistics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment','temperature','opponent_group','history','measure','round','n','fraction'])
        writer.writerows(stats)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(out.resolve())


if __name__ == '__main__':
    main()
