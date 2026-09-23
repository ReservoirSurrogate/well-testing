"""Milestone 1 grid study: transform accuracy vs grid size N, grid shape alpha, and method.

Run with:  /home/daniel_88/py314/bin/python phase_1/grid_study.py
"""
import numpy as np

from basis import HankelBasis, exact_coeffs_ones

R_E = 1000.0
K = 32


def main():
    print(f"r_e = {R_E:g}, K = {K}")
    print(f"{'method':<11}{'alpha':>6}{'N':>6}  {'|TB-I|':>8}  {'ones err':>8}  {'smooth rec':>10}")
    for method in ["quadrature", "lstsq"]:
        for alpha in [1.0, 0.5]:
            for n in [256, 512, 1024, 2048]:
                b = HankelBasis.build(K, n, R_E, alpha, method)
                tb = np.abs(b.T @ b.B - np.eye(K)).max()
                exact = exact_coeffs_ones(b.lam, R_E)
                ones = np.abs(b.forward(np.ones(n)) - exact).max() / np.abs(exact).max()
                f = np.log(b.r) - (b.r - 1) / R_E
                rec = np.abs(b.inverse(b.forward(f)) - f).max() / np.abs(f).max()
                print(f"{method:<11}{alpha:>6.1f}{n:>6}  {tb:8.1e}  {ones:8.1e}  {rec:10.1e}")


if __name__ == "__main__":
    main()
