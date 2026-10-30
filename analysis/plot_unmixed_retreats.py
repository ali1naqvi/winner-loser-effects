"""Plot original focal retreats by uninterrupted prior history."""
import csv
from plot_temperature_sweep import ROOT, TEMPS, GROUPS, load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    sweep = sorted((ROOT / 'data/control/temperature_sweeps').glob('*/sweep_status.json'))[-1].parent
    out = ROOT / 'output/temperature_sweeps' / sweep.name
    out.mkdir(parents=True, exist_ok=True)
    histories = ['none', 'L', 'LL', 'W', 'WW']
    labels = ['None', 'L', 'LL', 'W', 'WW']
    colors = ['#64748b', '#fb923c', '#c2410c', '#60a5fa', '#1d4ed8']
    plotted = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    with PdfPages(out / 'unmixed_retreats.pdf') as pdf:
        for exp in (1, 2):
            fig, axes = plt.subplots(3, 3, figsize=(11.5, 8), sharex=True, sharey=True)
            fig.subplots_adjust(left=.12, right=.985, top=.88, bottom=.13, hspace=.30, wspace=.18)
            limit = 100
            for row_i, temp in enumerate(TEMPS):
                trials, _ = load(sweep / f'temperature_{temp}_experiment_{exp}')
                if exp == 1:
                    # Analyze the full unmodified current-contest rollout once per repeat.
                    # W/WW and L/LL still refer to the PRIOR history.
                    trials = [r for r in trials if r['planned_outcome'] == 'loss']
                keys = [(r['opponent_group'], r['history'], r['repeat']) for r in trials]
                assert len(keys) == len(set(keys)), 'Repeated current contest within a history'

                for col_i, group in enumerate(GROUPS):
                    ax = axes[row_i, col_i]
                    counts = []
                    for history in histories:
                        rows = [r for r in trials if r['history'] == history and r['opponent_group'] == group]
                        assert rows
                        count = sum(r['stop_reason'] == 'retreat' for r in rows)
                        counts.append(count)
                        depleted = sum(r['stop_reason'] == 'strength_below_5' for r in rows)
                        assert count + depleted == len(rows)
                        plotted.append([exp, temp, group, history, len(rows), count, depleted])
                    bars = ax.bar(labels, counts, color=colors, width=.68)
                    ax.bar_label(bars, padding=2, fontsize=8)
                    ax.set_ylim(0, limit * 1.14)
                    ax.set_yticks([0, limit*.25, limit*.5, limit*.75, limit])
                    ax.tick_params(direction='out', length=3, width=.7)
                    ax.spines[['top','right']].set_visible(False)
                    for spine in ('left', 'bottom'):
                        ax.spines[spine].set_linewidth(.7)
                    ax.text(.98, .96, chr(65 + row_i*3 + col_i), transform=ax.transAxes,
                            ha='right', va='top', fontsize=10, fontweight='bold')
                    if row_i == 0:
                        ax.set_title({'weaker':'Weaker (70–79)', 'neutral':'Equal (80)', 'stronger':'Stronger (81–90)'}[group], fontsize=11, pad=15)
                    if col_i == 0:
                        ax.set_ylabel(f'Temperature = {temp}', fontsize=10, labelpad=13)
            fig.text(.12, .97, f"Experiment {exp}: {('Experienced history' if exp == 1 else 'Supplied history')}",
                     ha='left', va='top', fontsize=13, fontweight='bold')
            fig.supylabel('Contests ending in retreat (count)', x=.018, fontsize=11)
            fig.supxlabel('Previous contest outcomes', y=.045, fontsize=11)
            fig.savefig(out / f'experiment_{exp}_unmixed_retreats.png', dpi=300, facecolor='white')
            fig.savefig(out / f'experiment_{exp}_unmixed_retreats.pdf', facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)
    with (out / 'unmixed_retreat_counts.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment', 'temperature', 'opponent_group', 'history', 'contests_started', 'contests_ending_in_retreat', 'contests_ending_below_5'])
        writer.writerows(plotted)
    print(out / 'unmixed_retreats.pdf')


if __name__ == '__main__':
    main()
