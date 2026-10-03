"""Thống kê bổ sung từ kết quả đã có (không chạy lại thuật toán):
  E2 – cận dưới của ILP (RLT) khi chưa chứng minh tối ưu: bound = z_ilp·(1 − mip_gap) (HiGHS, cực tiểu)
       và khoảng cách của nghiệm GA tốt nhất tới cận dưới;
  E3 – so sánh từng cặp thuật toán (Mann–Whitney hai phía, các lần chạy độc lập), hiệu chỉnh Holm
       trong từng bộ dữ liệu, kích thước hiệu ứng Vargha–Delaney A12 = P(Z_a < Z_b) + ½P(Z_a = Z_b)
       (A12 > 0,5: thuật toán a cho Z nhỏ hơn – tốt hơn). Lưu ý: E3 so cùng ngân sách THỜI GIAN."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from experiments.common import RESULTS


def a12(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] < y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def e2_bound() -> pd.DataFrame | None:
    f_ilp, f_runs = RESULTS / "E2" / "ilp.csv", RESULTS / "E2" / "runs.csv"
    if not (f_ilp.exists() and f_runs.exists()):
        return None
    ilp = pd.read_csv(f_ilp)
    ilp = ilp[ilp.linearization == "rlt"]
    runs = pd.read_csv(f_runs)
    best = runs[runs.feasible].groupby(["n", "alpha"]).z.min().rename("ga_best").reset_index()
    out = ilp.merge(best, on=["n", "alpha"])
    out["bound"] = out.z_ilp * (1 - out.mip_gap)
    out["ga_gap_to_bound_pct"] = 100 * (out.ga_best - out.bound) / out.bound.abs()
    out["ga_vs_ilp_incumbent_pct"] = 100 * (out.ga_best - out.z_ilp) / out.z_ilp.abs()
    out = out[["n", "alpha", "optimal", "z_ilp", "mip_gap", "bound", "ga_best", "ga_vs_ilp_incumbent_pct",
               "ga_gap_to_bound_pct"]]
    out.to_csv(RESULTS / "E2" / "lower_bound.csv", index=False)
    return out


def e3_pairwise() -> pd.DataFrame | None:
    f = RESULTS / "E3" / "runs.csv"
    if not f.exists():
        return None
    runs = pd.read_csv(f)
    rows = []
    for nm, sub in runs.groupby("instance"):
        algos = sorted(sub.algo.unique())
        block = []
        for a, b in itertools.combinations(algos, 2):
            x, y = sub[sub.algo == a].z.values, sub[sub.algo == b].z.values
            p = 1.0 if np.array_equal(np.sort(x), np.sort(y)) else stats.mannwhitneyu(x, y).pvalue
            block.append({"instance": nm, "a": a, "b": b, "mean_a": x.mean(), "mean_b": y.mean(),
                          "p": p, "A12": a12(x, y)})
        ph = holm(np.array([r["p"] for r in block]))
        for r, q in zip(block, ph):
            r["p_holm"] = q
            r["significant"] = q < 0.05
            r["effect"] = ("không đáng kể" if abs(r["A12"] - .5) < .06 else "nhỏ" if abs(r["A12"] - .5) < .14
                           else "trung bình" if abs(r["A12"] - .5) < .21 else "lớn")
        rows += block
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "E3" / "pairwise_holm_a12.csv", index=False)
    return out


def main():
    pd.set_option("display.width", 200)
    b = e2_bound()
    if b is not None:
        print("== E2: cận dưới ILP (RLT) và khoảng cách GA")
        print(b.round(4).to_string(index=False))
    p = e3_pairwise()
    if p is not None:
        print("\n== E3: so sánh từng cặp (Holm) + A12")
        print(p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
