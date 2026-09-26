"""Zero-shot transfer of trained direct operators to other reservoir sizes r_eD (Milestone 5b).

Models trained on r_eD = 1000 (N = 1024, ln-r spacing D = ln 1000 / 1023) are evaluated without retraining on test
sets with the same spacing D and the same k_D statistics (GRF in ln r) but a different outer radius:

    phase_1/data/re299/test.npz    N = 845,  r_eD = exp(844 D)  ~ 299   (boundary felt from t_D ~ r_eD^2/4 ~ 2e4)
    phase_1/data/re2986/test.npz   N = 1186, r_eD = exp(1185 D) ~ 2986  (infinite-acting over the whole t range)

The model is rebuilt on the new grid with the training grid's physical settings (direct_evaluate.load_model):
s = ln r / ln 1000, the fno pad length, and the FFTLog control points at the training grid's physical k. What
cannot transfer is by construction: fno's modes are tied to the length of the grid (mode m has frequency
m / (N + pad) in index space), while fftlog's R(k) is a function of physical k.

Usage: python transfer_evaluate.py [--runs stage2/fno stage4/fftlog] [--data re299 re2986] [--threads 6]
Writes: runs/<run>/eval_<data>.json
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from direct_evaluate import evaluate  # noqa: E402
from direct_train import DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=["stage2/fno", "stage4/fftlog"])
    ap.add_argument("--data", nargs="+", default=["re299", "re2986"])
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"{'run':<16} {'data':<8} {'u med':>9} {'u p90':>9} {'q med':>9} {'u max-t':>9} {'q max-t':>9} "
          f"{'k=1 max':>9} {'pass':>5}")
    for data in args.data:
        for run in args.runs:
            res = evaluate(HERE / "runs" / run, data_dir=DATA / data, out_name=f"eval_{data}.json")
            a, b = res["all_times"], res["bar"]
            print(f"{run:<16} {data:<8} {a['grid']['median']:9.2e} {a['grid']['p90']:9.2e} {a['q']['median']:9.2e} "
                  f"{b['u_max_median']:9.2e} {b['q_max_median']:9.2e} {res['homogeneous']['max']:9.2e} "
                  f"{str(b['passes_all_times']):>5}", flush=True)


if __name__ == "__main__":
    main()
