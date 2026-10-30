"""Plot completed label-only Experiment 1 results in the history-extension style."""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".plot_deps"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RUN = ROOT / "data/control/experiment_1/20261011T150646453796Z"
OUT = ROOT / "output/experiment_1_sd"
HISTORIES = ("none", "W", "L", "WW", "WL", "LW", "LL")
COLORS = ("#377eb8", "#e6863b")


def main():
    with (RUN / "unique_contests.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len({r["sample_id"] for r in rows}) == 14000
    cells = defaultdict(list)
    for row in rows:
        assert row["completed"] == "True"
        cells[int(row["strength_difference"]), row["history"]].append(int(row["fight_rounds"]))
    assert len(cells) == 140 and all(len(v) == 100 for v in cells.values())
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = ROOT / "output/pdf/experiment_1_history_sd.pdf"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    stats = []
    with PdfPages(pdf_path) as pdf:
        for gap in [None, *range(1, 11)]:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
            fig.subplots_adjust(left=.055, right=.99, bottom=.18, top=.77, wspace=.27)
            panels = [("Repeated wins", ["none", "W", "WW"]),
                      ("Repeated losses", ["none", "L", "LL"]), ("Outcome order", ["WL", "LW"])]
            for ax, (title, histories) in zip(axes, panels):
                for i, (group, color) in enumerate(zip(("Weaker", "Stronger"), COLORS)):
                    differences = list(range(-10, 0) if i == 0 else range(1, 11)) if gap is None else [-gap if i == 0 else gap]
                    values = [sum((cells[d, h] for d in differences), []) for h in histories]
                    means = [statistics.mean(v) for v in values]
                    sds = [statistics.stdev(v) for v in values]
                    label = f"{group} (70-79)" if gap is None and i == 0 else f"{group} (81-90)" if gap is None else f"{group} ({80 + differences[0]})"
                    for h, v, m, sd in zip(histories, values, means, sds):
                        stats.append(dict(gap=gap, group=group, history=h, n=len(v), mean=m, sd=sd))
                    if title == "Outcome order":
                        x = [j + (i - .5) * .32 for j in range(2)]
                        ax.bar(x, means, .32, color=color, label=label)
                        ax.bar(x, [2*s for s in sds], .32, bottom=[m-s for m,s in zip(means,sds)], color="white", alpha=.8)
                        ax.bar(x, [2*s for s in sds], .32, bottom=[m-s for m,s in zip(means,sds)], color=color, alpha=.18)
                        ax.hlines(means, [v-.16 for v in x], [v+.16 for v in x], color=color, linewidth=2)
                    else:
                        ax.plot(range(len(histories)), means, "o-", color=color, label=label, linewidth=2)
                        ax.fill_between(range(len(histories)), [m-s for m,s in zip(means,sds)],
                                        [m+s for m,s in zip(means,sds)], color=color, alpha=.18, linewidth=0)
                ax.set(title=title, xlabel="Prior history (oldest to newest)", ylabel="Mean fighting rounds",
                       xticks=range(len(histories)), xticklabels=["None" if h=="none" else h for h in histories], ylim=(-.5,4.5))
                ax.set_yticks(range(5))
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
            axes[0].legend(title="Perceived opponent strength", frameon=False, fontsize=9, loc="lower left")
            detail = "All strength gaps pooled | n = 1,000 per group/history" if gap is None else f"Strength gap +/-{gap} | n = 100 per strength/history"
            fig.suptitle(f"Experiment 1 (completed label-only version) | starting strength 80\n{detail}\nMean +/- 1 SD across unique contests", fontsize=13, y=.98)
            fig.text(.055,.045,"Shading: response variation, not uncertainty of the mean. Reused samples counted once.\nNo fourth contest: responses to WWW/LLL were not measured. Full prior conversations were not retained.", fontsize=9)
            name = "overview" if gap is None else f"gap_{gap:02d}"
            fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
            pdf.savefig(fig)
            plt.close(fig)
    (OUT / "statistics.json").write_text(json.dumps(dict(source=str(RUN), cells=stats), indent=2), encoding="utf-8")
    print(pdf_path)


if __name__ == "__main__":
    main()
