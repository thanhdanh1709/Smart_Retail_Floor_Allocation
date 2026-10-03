"""Hạ tầng chung cho các thí nghiệm E1–E7 (mục B7).

- Cấu hình đọc từ experiments/config.yaml (hồ sơ "full" theo đề cương, "quick" để chạy thử).
- Bảng payoff của mỗi instance được tính MỘT lần và lưu ở results/payoff_cache.json để mọi
  lần chạy/tiến trình dùng chung một phép chuẩn hóa (11).
- Mọi lần chạy ghi lại cấu hình, hạt giống, phiên bản thư viện (B7.3 – "chạy lại có ra như vậy không").
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import baselines, solvers  # noqa: E402
from src.instance import make_instance  # noqa: E402

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
PAYOFF_CACHE = RESULTS / "payoff_cache.json"
CONFIG = ROOT / "experiments" / "config.yaml"


def load_config(profile: str) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base = cfg["full"]
    if profile != "full":
        base = {**base, **cfg[profile]}
    return base


def cli(desc: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--profile", default="full", choices=["full", "quick"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return ap.parse_args()


def out_dir(exp: str) -> Path:
    d = RESULTS / exp
    d.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    return d


def save_run_config(exp: str, cfg: dict, extra: dict | None = None) -> None:
    import numba
    import pandas
    import pulp
    import pymoo
    import scipy
    info = {
        "experiment": exp, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
                     "numba": numba.__version__, "pymoo": pymoo.__version__, "pulp": pulp.__version__},
        "config": cfg, **(extra or {}),
    }
    (out_dir(exp) / "run_config.yaml").write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
                                                  encoding="utf-8")


# ------------------------------------------------------------- instances
def spec_name(spec: dict) -> str:
    s = f"{spec['kind']}_{spec['scale']}"
    if spec.get("n"):
        s += f"_n{spec['n']}"
    if spec.get("restrict"):
        s += "_r"
    if spec.get("R", -1) >= 0:
        s += f"_R{spec['R']}"
    return s


CASE_SPEC = {"kind": "case", "scale": "minimart", "n": 30,
             "grid_file": "data/floorplans/case_minimart.txt"}


def build(spec: dict):
    grid = None
    if spec.get("grid_file"):
        grid = (ROOT / spec["grid_file"]).read_text(encoding="utf-8").split()
    return make_instance(spec["kind"], spec["scale"], n=spec.get("n"), R=spec.get("R", -1),
                         restrict_slots=spec.get("restrict", False), grid=grid,
                         level=spec.get("level", "group" if grid is not None else None))


def _read_cache() -> dict:
    if PAYOFF_CACHE.exists():
        return json.loads(PAYOFF_CACHE.read_text(encoding="utf-8"))
    return {}


def get_instance(spec: dict, payoff_method: str = "auto", **payoff_kw):
    """Dựng instance và gắn bảng payoff (tính và lưu cache nếu chưa có)."""
    inst = build(spec)
    key = spec_name(spec)
    cache = _read_cache()
    if key not in cache:
        P = solvers.compute_payoff(inst, method=payoff_method, **payoff_kw)
        cache = _read_cache()
        cache[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
        RESULTS.mkdir(parents=True, exist_ok=True)
        PAYOFF_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    P = cache[key]
    inst.payoff = {k: (np.array(v, dtype=np.int64) if k.startswith("perm") else v) for k, v in P.items()}
    return inst


def ensure_payoffs(specs: list[dict], method: str = "auto") -> None:
    for s in specs:
        t = time.time()
        get_instance(s, method)
        print(f"  payoff {spec_name(s)} ({time.time() - t:.1f}s)")


# ------------------------------------------------------------- parallel
def _init_worker():
    baselines.warmup()


def pmap(fn, tasks: list, workers: int):
    if workers <= 1:
        _init_worker()
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def instance_specs(scales=("small", "medium", "large"), kinds=("grid", "racetrack", "free")) -> list[dict]:
    return [{"kind": k, "scale": s} for s in scales for k in kinds]


# ------------------------------------------------------------- Pareto plans
PLAN_NAMES = ["Tiện lợi (min Z1)", "Cân bằng (điểm gối)", "Giá trị (max Z2)"]


def pareto_plans(inst, time_limit: float = 20.0, seed: int = 0, pop: int = 100,
                 ls_prob: float = 0.1) -> tuple[dict, dict]:
    """Tập Pareto bằng NSGA-II lai + GA ở hai đầu α ∈ {0, 1}; trả về (3 phương án, tập Pareto):
    cực biên tiện lợi, điểm gối (đề xuất mặc định cho nhà quản lý) và cực biên giá trị."""
    from src import ga, moo
    r = moo.nsga2(inst, pop_size=pop, time_limit=time_limit, seed=seed, ls_prob=ls_prob)
    extra = [ga.run(inst, a, ga.GAConfig(seed=seed, stall_gens=60)).perm for a in (0.0, 1.0)]
    front = moo.front_from_perms(inst, list(r["perms"]) + extra)
    F = front["F"]
    k = moo.knee_point(F)
    plans = {PLAN_NAMES[0]: front["perms"][int(np.argmin(F[:, 0]))],
             PLAN_NAMES[1]: front["perms"][k],
             PLAN_NAMES[2]: front["perms"][int(np.argmin(F[:, 1]))]}
    return plans, front
