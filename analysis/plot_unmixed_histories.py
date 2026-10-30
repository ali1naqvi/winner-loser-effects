"""Plot original fighting behavior after uninterrupted win or loss histories."""
import csv
from pathlib import Path
from plot_temperature_sweep import ROOT, TEMPS, GROUPS, load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


def main():
    sweep = sorted((ROOT / 'data/control/temperature_sweeps').glob('*/sweep_status.json'))[-1].parent
    out = ROOT / 'output/temperature_sweeps' / sweep.name
    out.mkdir(parents=True, exist_ok=True)
    histories = ['none', 'L', 'LL', 'W', 'WW']
    colors = {'none': '#64748b', 'L': '#fb923c', 'LL': '#c2410c', 'W': '#60a5fa', 'WW': '#1d4ed8'}
    stats = []
    with PdfPages(out / 'unmixed_histories.pdf') as pdf:
        for exp in (1, 2):
            plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
            fig, axes = plt.subplots(3, 3, figsize=(11.5, 8), sharex=True, sharey=True)
            fig.subplots_adjust(left=.12, right=.985, top=.88, bottom=.15, hspace=.30, wspace=.18)
            for row_i, temp in enumerate(TEMPS):
                trials, manifest = load(sweep / f'temperature_{temp}_experiment_{exp}')
                if exp == 1:
                    # Analyze the full unmodified current-contest rollout once per repeat.
                    # W/WW and L/LL still refer to the PRIOR history.
                    trials = [r for r in trials if r['planned_outcome'] == 'loss']
                keys = [(r['opponent_group'], r['history'], r['repeat']) for r in trials]
                assert len(keys) == len(set(keys)), 'Repeated current contest within a history'

                starting = manifest['config']['contest_sequence_experiments']['starting_strength']
                min_damage = manifest['config']['control']['strength_loss']['min']
                max_round = (starting - 5) // min_damage + 1
                for col_i, group in enumerate(GROUPS):
                    ax = axes[row_i, col_i]
                    for h in histories:
                        rows = [r for r in trials if r['history'] == h and r['opponent_group'] == group]
                        values = [sum(int(r['fight_rounds']) >= k for r in rows) / len(rows) for k in range(1, max_round+1)]
                        ax.plot(range(1,max_round+1), values, color=colors[h], linestyle='--' if h in ['L','LL'] else '-', marker='o', lw=2.2, ms=4, label='No history' if h == 'none' else h)
                        stats.append([exp,temp,group,h,len(rows),sum(int(r['fight_rounds']) for r in rows)/len(rows),*values,*(['']*(6-len(values)))])
                    ax.set(ylim=(0,1.03), xticks=range(1,max_round+1), yticks=[0,.25,.5,.75,1])
                    ax.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
                    ax.tick_params(direction='out', length=3, width=.7)
                    ax.spines[['top','right']].set_visible(False)
                    for spine in ('left', 'bottom'):
                        ax.spines[spine].set_linewidth(.7)
                    ax.text(.98, .96, chr(65 + row_i*3 + col_i), transform=ax.transAxes,
                            ha='right', va='top', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1.5))
                    if row_i == 0:
                        ax.set_title({'weaker':'Weaker (70–79)', 'neutral':'Equal (80)', 'stronger':'Stronger (81–90)'}[group], fontsize=11, pad=15)
                    if col_i == 0:
                        ax.set_ylabel(f'Temperature = {temp}', fontsize=10, labelpad=13)
            fig.text(.12, .97, f"Experiment {exp}: {('Experienced history' if exp == 1 else 'Supplied history')}",
                     ha='left', va='top', fontsize=13, fontweight='bold')
            fig.supylabel('Contests choosing fight (%)', x=.018, fontsize=11)
            fig.supxlabel('Contest round', y=.085, fontsize=11)
            handles, labels = axes[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55,.015),
                       ncol=5, frameon=False, title='Previous contest outcomes', title_fontsize=10, fontsize=10, handlelength=2.5, columnspacing=1.7)
            fig.savefig(out/f'experiment_{exp}_unmixed_histories.png',dpi=300,facecolor='white')
            fig.savefig(out/f'experiment_{exp}_unmixed_histories.pdf', facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)
    with (out/'unmixed_history_statistics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['experiment','temperature','opponent_group','history','n','mean_fight_rounds',*[f'fight_round_{i}_fraction' for i in range(1,7)]])
        writer.writerows(stats)
    print(out/'unmixed_histories.pdf')


if __name__=='__main__':
    main()
