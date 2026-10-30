"""Plot completed unique contests: endings and fight participation by history."""
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter, MaxNLocator

OUT = ROOT / 'output/experiment_endings_rounds'
RUNS = [(1, '20261026T224755687530Z', 'Rollback protocol: original rollouts', ['none','W','L','WW','WL','LW','LL']),
        (2, '20261026T233720102821Z', 'Supplied history', ['none','W','L','WW','WL','LW','LL'])]
COLORS = {'none':'#64748b','W':'#60a5fa','L':'#fb923c','WW':'#1d4ed8',
          'WL':'#7c3aed','LW':'#0d9488','LL':'#c2410c'}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    stats = []
    with PdfPages(OUT / 'experiment_endings_and_fighting.pdf') as pdf:
        for exp, stamp, mode, histories in RUNS:
            run = ROOT / f'data/control/experiment_{exp}' / stamp
            with (run/'unique_contests.csv').open(encoding='utf-8-sig', newline='') as f:
                trials = list(csv.DictReader(f))
            assert all(t['completed']=='True' for t in trials)
            assert len({t['sample_id'] for t in trials}) == len(trials)
            assert all(t['stop_reason'] in ('retreat','strength_below_5') for t in trials)
            # Each sequential contest consists of fights followed by optional retreat.
            # Verify this against the actual decisions, counting cached samples once.
            observed = defaultdict(dict)
            ids = {t['sample_id'] for t in trials}
            with (run/'rounds.csv').open(encoding='utf-8-sig', newline='') as f:
                for r in csv.DictReader(f):
                    if r['sample_id'] not in ids: continue
                    k = int(r['current_round'])
                    prior = observed[r['sample_id']].setdefault(k, r['action'])
                    assert prior == r['action'] and not r['validation_error']
            for t in trials:
                actions = observed[t['sample_id']]
                n = int(t['fight_rounds'])
                assert len(actions) == int(t['rounds_observed'])
                assert [actions[k] for k in sorted(actions)] == ['fight']*n + (['retreat'] if t['stop_reason']=='retreat' else [])
            max_round = max(int(t['rounds_observed']) for t in trials)
            for kind in ('endings','fighting'):
                fig, axes = plt.subplots(1,3,figsize=(18,6.4),sharey=True)
                fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.78,wspace=.12)
                for ax, group in zip(axes, ('weaker','neutral','stronger')):
                    cells = {h:[t for t in trials if t['opponent_group']==group and t['history']==h] for h in histories}
                    assert all(len(v)==(200 if exp==1 else 100) for v in cells.values())
                    if kind=='endings':
                        retreat = [sum(t['stop_reason']=='retreat' for t in cells[h])/len(cells[h]) for h in histories]
                        ax.bar(range(len(histories)),retreat,color='#0f766e',label='Retreat')
                        ax.bar(range(len(histories)),[1-v for v in retreat],bottom=retreat,color='#cbd5e1',label='Strength below 5')
                        for i,v in enumerate(retreat):
                            ax.text(i,v/2 if v>.08 else v+.025,f'{v:.1%}',ha='center',va='center',color='white' if v>.08 else '#0f766e',fontsize=9)
                            stats.append([exp,group,histories[i],'retreat', '',len(cells[histories[i]]),v])
                        ax.set(xticks=range(len(histories)),xticklabels=['None' if h=='none' else h for h in histories],xlabel='Previous history (oldest to newest)')
                    else:
                        for h in histories:
                            fractions = [sum(int(t['fight_rounds'])>=k for t in cells[h])/len(cells[h]) for k in range(1,max_round+1)]
                            ax.plot(range(1,max_round+1),fractions,'o-',color=COLORS[h],label='None' if h=='none' else h,linewidth=2,ms=4)
                            for k,v in enumerate(fractions,1): stats.append([exp,group,h,'fight',k,len(cells[h]),v])
                        ax.set(xlabel='Contest round',xticks=range(1,max_round+1))
                    ax.set(title=f'Perceived opponent: {group} ({dict(weaker=70,neutral=80,stronger=90)[group]})',ylim=(0,1.08))
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                    ax.grid(axis='y',alpha=.2)
                    ax.set_axisbelow(True)
                axes[0].set_ylabel('Trials (%)' if kind=='endings' else 'Starting trials choosing fight (%)')
                handles,labels=axes[0].get_legend_handles_labels()
                fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.085),ncol=len(labels),frameon=False)
                title='How trials end' if kind=='endings' else 'Fighting across rounds by previous history'
                fig.suptitle(f'Experiment {exp}: {title}',fontsize=17,weight='bold',y=.97)
                fig.text(.5,.865,f'October 26, 2026 · {mode} · strength 80 · {200 if exp==1 else 100} unique contests per group/history',ha='center')
                note='Labels show retreat percentages. Original stopping reasons; constructed win feedback is excluded.' if kind=='endings' else 'Denominator: all starting contests per line. Ended trials contribute zero in subsequent rounds.'
                fig.text(.5,.025,note,ha='center',fontsize=9,color='#475569')
                fig.savefig(OUT/f'experiment_{exp}_{kind}.png',dpi=180,facecolor='white')
                pdf.savefig(fig)
                plt.close(fig)
            print(f'Experiment {exp}: validated {len(trials):,} unique contests against round decisions')
    with (OUT/'plotted_statistics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['experiment','opponent_group','history','measure','round','n','fraction'])
        writer.writerows(stats)
    print(OUT/'experiment_endings_and_fighting.pdf')

if __name__=='__main__': main()
