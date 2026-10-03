"""Chạy lại toàn bộ thí nghiệm E1–E7 (mặc định hồ sơ "full").

    python -m experiments.run_all                 # đầy đủ theo đề cương (vài giờ)
    python -m experiments.run_all --profile quick # chạy thử nhanh
    python -m experiments.run_all --only e3 e5    # chỉ chạy một số thí nghiệm
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["e1_qaplib", "e3_heuristics", "e4_pareto", "e5_simulation", "e6_sensitivity", "e7_ablation",
         "e2_ga_vs_ilp", "extra_stats"]   # E2 (ILP, lâu nhất) chạy gần cuối; extra_stats đọc kết quả E2, E3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "quick"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", nargs="*", default=None, help="vd. e1 e3")
    args = ap.parse_args()
    todo = [e for e in ORDER if not args.only or e.split("_")[0] in args.only]
    log = ROOT / "results" / "run_all.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for e in todo:
        t0 = time.time()
        print(f"=== {e} ({args.profile})", flush=True)
        with open(ROOT / "results" / f"{e}.out.txt", "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, "-m", f"experiments.{e}", "--profile", args.profile,
                                  "--workers", str(args.workers)], cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT, env=env)
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {e}  rc={rc}  {time.time() - t0:.0f}s"
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


if __name__ == "__main__":
    main()
