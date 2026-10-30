"""Plot unsuccessful prescribed wins as a fraction of attempted wins by prior history."""
import csv
from plot_temperature_sweep import ROOT, TEMPS, GROUPS, load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


def main():
    sweep = sorted((ROOT / 'data/control/temperature_sweeps').glob('*/sweep_status.json'))[-1].parent
    out = ROOT / 'output/temperature_sweeps' / sweep.name
    histories = ['none', 'L', 'LL', 'W', 'WW', 'WL', 'LW']
    labels = ['None', 'L', 'LL', 'W', 'WW', 'WL', 'LW']
    colors = ['#64748b', '#fb923c', '#c2410c', '#60a5fa', '#1d4ed8', '#7c3aed', '#0d9488']
    stats = []
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig, axes = plt.subplots(3,3,figsize=(11.5,8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.985,top=.88,bottom=.13,hspace=.30,wspace=.18)
    total = 0
    expected_total = 0
    for row_i,temp in enumerate(TEMPS):
        trials,_ = load(sweep/f'temperature_{temp}_experiment_1')
        attempts = [r for r in trials if r['planned_outcome']=='win']
        expected_total += sum(r['experience_delivered']=='False' for r in attempts)
        keys = [(r['opponent_group'],r['history'],r['repeat']) for r in attempts]
        assert len(keys)==len(set(keys))
        for col_i,group in enumerate(GROUPS):
            ax=axes[row_i,col_i]
            fractions=[]
            annotations=[]
            for history in histories:
                rows=[r for r in attempts if r['opponent_group']==group and r['history']==history]
                failed=[r for r in rows if r['experience_delivered']=='False']
                assert rows
                assert all(r['first_action']=='retreat' and r['fight_rounds']=='0' and r['rounds_observed']=='1' for r in failed)
                count,n=len(failed),len(rows)
                total+=count
                fractions.append(count/n)
                annotations.append(f'{count}/{n}')
                stats.append([temp,group,history,n,count,count/n])
            bars=ax.bar(labels,fractions,color=colors,width=.70)
            ax.bar_label(bars,labels=annotations,padding=3,fontsize=7)
            ax.set(ylim=(0,.35),yticks=[0,.1,.2,.3])
            ax.yaxis.set_major_formatter(PercentFormatter(1,symbol='',decimals=0))
            ax.tick_params(direction='out',length=3,width=.7)
            ax.spines[['top','right']].set_visible(False)
            for spine in ['left','bottom']:ax.spines[spine].set_linewidth(.7)
            ax.text(.98,.96,chr(65+row_i*3+col_i),transform=ax.transAxes,ha='right',va='top',fontsize=10,fontweight='bold')
            if row_i==0:
                ax.set_title({'weaker':'Weaker (70–79)','neutral':'Equal (80)','stronger':'Stronger (81–90)'}[group],fontsize=11,pad=15)
            if col_i==0:ax.set_ylabel(f'Temperature = {temp}',fontsize=10,labelpad=13)
    assert total==expected_total, 'Some unsuccessful attempts were omitted'
    fig.text(.12,.97,'Experiment 1: Undelivered prescribed wins',ha='left',va='top',fontsize=13,fontweight='bold')
    fig.supylabel('Immediate retreat (%)',x=.018,fontsize=11)
    fig.supxlabel('Previous contest outcomes',y=.045,fontsize=11)
    fig.savefig(out/'experiment_1_undelivered_wins.png',dpi=300,facecolor='white')
    with PdfPages(out/'experiment_1_undelivered_wins.pdf') as pdf:pdf.savefig(fig)
    plt.close(fig)
    with (out/'undelivered_win_statistics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['temperature','opponent_group','history','prescribed_win_attempts','immediate_retreats','immediate_retreat_fraction'])
        writer.writerows(stats)
    (out/'undelivered_win_figure_notes.txt').write_text('Bars are the number of unique prescribed-win attempts with immediate focal retreat divided by the number of attempted prescribed wins in the same temperature/opponent/prior-history cell. Labels show failures/attempts. All undelivered wins in this sweep were immediate round-1 retreats. No surviving fight was available for rollback. Samples on shared sequence prefixes count once. Denominators can be smaller than 100 when prior failures prevent reaching a history. These rates describe feasibility of delivering the prescribed experience, not model errors.\n')
    print('Total undelivered wins:',total)
    print(out/'experiment_1_undelivered_wins.pdf')


if __name__=='__main__':main()
