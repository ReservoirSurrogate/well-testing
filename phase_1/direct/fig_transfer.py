"""Zero-shot transfer to other reservoir sizes: median test error vs time for fno and fftlog, trained on r_eD = 1000.

(a) relative L2 error of u, (b) relative well-rate error; one jump from u = 1, median over 100 test trajectories.
Color = model, line style = r_eD. Ticks at t_D = r_eD^2 / 4 mark when the outer boundary starts to be felt.
Reads runs/<run>/eval.json (r_eD = 1000) and eval_re299.json / eval_re2986.json (transfer_evaluate.py).

Run with:  /home/daniel_88/py314/bin/python phase_1/direct/fig_transfer.py [--png preview.png]
Writes:    tex/fig_transfer.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "fig_transfer.pdf"
MODELS = {"fno": ("stage2/fno", "#2a78d6"), "fftlog": ("stage4/fftlog", "#eda100")}
SIZES = [("eval_re299.json", ":"), ("eval.json", "-"), ("eval_re2986.json", "--")]
MUTED = "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG preview here")
    args = ap.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for model, (run, color) in MODELS.items():
        for fname, ls in SIZES:
            res = json.loads((HERE / "runs" / run / fname).read_text())
            r_e = res.get("r_e", 1000.0)
            label = f"{model}, $r_{{eD}}={r_e:.0f}$" + (" (trained)" if fname == "eval.json" else "")
            for a, key in zip(ax, ("grid", "q")):
                a.loglog(res["t"], res["per_time"][key]["median"], color=color, ls=ls, lw=1.8, label=label)

    for a, bar, name in zip(ax, (0.02, 0.05), (r"$u$", r"$q_D$")):
        a.axhline(bar, color=MUTED, lw=1, ls=(0, (1, 3)))
        a.text(1.05e3, bar * 1.08, f"bar {bar:.0%}", color=MUTED, fontsize=8)
        for r_e in (299, 1000):
            a.axvline(r_e**2 / 4, color=MUTED, lw=0.6, alpha=0.6)
            a.text(r_e**2 / 4 * 1.05, 0.98, rf"$r_{{eD}}^2/4$, {r_e}", transform=a.get_xaxis_transform(),
                   va="top", fontsize=7, color=MUTED)
        a.set_xlabel(r"$t_D$")
        a.set_ylabel(f"median relative error of {name}")
        a.grid(True, which="major", color="#e5e5e2", lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_title(r"(a) pressure $u$")
    ax[1].set_title(r"(b) well rate $q_D$")
    ax[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    if args.png:
        fig.savefig(args.png, dpi=110)


if __name__ == "__main__":
    main()
