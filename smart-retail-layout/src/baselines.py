"""Các phương pháp so sánh (mục B5.6): hiện trạng, ngẫu nhiên, tham lam,
mô phỏng luyện kim (SA) và tìm kiếm tabu (Taillard, 1991) – cùng lân cận hoán đổi
và cùng ngân sách thời gian với GA."""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from . import fastops
from .instance import Instance, random_perm


def random_baseline(inst: Instance, alpha: float, n: int = 1000, seed: int = 0) -> dict:
    """Trung bình của n hoán vị ngẫu nhiên (thỏa tương thích) – mức 'không tối ưu'."""
    rng = np.random.default_rng(seed)
    perms = [random_perm(inst, rng) for _ in range(n)]
    vals = np.array([inst.z(p, alpha) for p in perms])           # Z thuần, không phạt
    feas = np.mean([inst.feasible(p) for p in perms])
    return {"mean": float(vals.mean()), "std": float(vals.std()), "best": float(vals.min()),
            "feasible_rate": float(feas)}


def warmup() -> None:
    """Biên dịch trước các nhân numba để không tính thời gian biên dịch vào ngân sách."""
    m = 6
    W = np.random.rand(m, m)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    D = np.abs(np.subtract.outer(np.arange(m), np.arange(m))).astype(float)
    L = np.zeros((m, m))
    A = np.ones((m, m), dtype=np.bool_)
    si = np.array([0], dtype=np.int64)
    sj = np.array([1], dtype=np.int64)
    sd = np.array([1.0])
    k0 = np.arange(m, dtype=np.int64)
    perm = np.arange(m, dtype=np.int64)
    loc = fastops.inverse(perm)
    fastops.fitness(perm, W, D, L, 1.0, A, si, sj, sd, k0, 2, 1.0)
    fastops.local_search(perm.copy(), W, D, L, 1.0, A, si, sj, sd, k0, 2, 1.0, 5, True)
    _sa_chunk(perm.copy(), loc.copy(), 0.0, perm.copy(), 0.0, 0, 10, 1.0, 0.1,
              W, D, L, 1.0, A, si, sj, sd, k0, 2, 1.0, 0)
    _tabu_chunk(perm.copy(), loc.copy(), 0.0, perm.copy(), 0.0, 0, np.zeros((m, m), np.int64), 0, 2,
                1, 2, W, D, L, 1.0, A, si, sj, sd, k0, 2, 1.0, 0)
    from .ga import ox, pmx
    ox(perm, perm[::-1].copy(), 1, 3)
    pmx(perm, perm[::-1].copy(), 1, 3)


def greedy_popularity(inst: Instance) -> np.ndarray:
    """Tham lam (B5.6): nhóm phổ biến nhất vào slot có mức tiếp xúc cao nhất (trong từng lớp
    tương thích lạnh/thường)."""
    n, m = inst.n, inst.m
    perm = -np.ones(m, dtype=np.int64)
    cold_cat = inst.meta.needs_cold.values.astype(bool)
    for want in (True, False):
        cats = [i for i in np.argsort(-inst.f[:n]) if cold_cat[i] == want]
        slots = [k for k in np.argsort(-inst.e) if inst.is_cold[k] == want]
        for i, k in zip(cats, slots):
            perm[k] = i
    free = iter(range(n, m))
    for k in range(m):
        if perm[k] < 0:
            perm[k] = next(free)
    return perm


# --------------------------------------------------------------------- SA
@njit(cache=True)
def _sa_chunk(perm, loc, cur, best_perm, best_val, moved, n_iter, T_hi, T_lo,
              W, D, L, cq, allowed, si, sj, sd, k0, R, rho, seed):
    np.random.seed(seed)
    m = perm.shape[0]
    ratio = (T_lo / T_hi) ** (1.0 / max(n_iter, 1))
    T = T_hi
    for _ in range(n_iter):
        r = np.random.randint(m)
        s = np.random.randint(m - 1)
        if s >= r:
            s += 1
        a = perm[r]
        b = perm[s]
        if allowed[a, s] and allowed[b, r]:
            d, dm = fastops.swap_delta_full(perm, loc, W, D, L, cq, si, sj, sd, k0, R, rho,
                                            moved, r, s)
            if d <= 0.0 or np.random.random() < np.exp(-d / T):
                perm[r] = b
                perm[s] = a
                loc[a] = s
                loc[b] = r
                moved += dm
                cur += d
                if cur < best_val - 1e-12:
                    best_val = cur
                    best_perm[:] = perm
        T *= ratio
    return cur, best_val, moved


def simulated_annealing(inst: Instance, alpha: float, time_limit: float = 10.0, seed: int = 0,
                        init: np.ndarray | None = None, T0: float | None = None,
                        T_end_ratio: float = 1e-4, chunk: int = 20000, rho: float = 10.0) -> dict:
    """SA với nhiệt độ giảm theo hàm mũ theo thời gian đã trôi qua trong ngân sách time_limit."""
    rng = np.random.default_rng(seed)
    args, c0 = inst.kernel_args(alpha, rho)
    W, D, L, cq, allowed, si, sj, sd, k0, R, rho = args
    perm = (init.copy() if init is not None else random_perm(inst, rng)).astype(np.int64)
    loc = fastops.inverse(perm)
    moved = inst.moved(perm) if R >= 0 else 0
    cur = fastops.fitness(perm, *args)
    if T0 is None:   # T0: chấp nhận ~50% nước đi xấu trung bình
        ds = []
        for _ in range(500):
            r, s = rng.choice(inst.m, 2, replace=False)
            if allowed[perm[r], s] and allowed[perm[s], r]:
                d, _ = fastops.swap_delta_full(perm, loc, W, D, L, cq, si, sj, sd, k0, R, rho, moved, r, s)
                if d > 0:
                    ds.append(d)
        T0 = (np.mean(ds) / np.log(2)) if ds else 1e-3
    T_end = T0 * T_end_ratio
    best_perm, best_val = perm.copy(), cur
    t0 = time.time()
    history = [(0.0, best_val + c0)]
    iters = 0
    while True:
        el = time.time() - t0
        if el >= time_limit:
            break
        f0, f1 = el / time_limit, min(1.0, (el + 0.02) / time_limit)
        T_hi = T0 * (T_end / T0) ** f0
        T_lo = T0 * (T_end / T0) ** f1
        cur, best_val, moved = _sa_chunk(perm, loc, cur, best_perm, best_val, moved, chunk,
                                         T_hi, T_lo, W, D, L, cq, allowed, si, sj, sd, k0, R, rho,
                                         int(rng.integers(2**31 - 1)))
        iters += chunk
        history.append((time.time() - t0, best_val + c0))
    # đánh bóng cuối bằng 2-swap
    fastops.local_search(best_perm, W, D, L, cq, allowed, si, sj, sd, k0, R, rho, 10 * inst.m, False)
    best_val = fastops.fitness(best_perm, *args)
    return {"perm": best_perm, "z": best_val + c0, "time": time.time() - t0, "iters": iters,
            "history": history}


# ------------------------------------------------------------------- tabu
@njit(cache=True)
def _tabu_chunk(perm, loc, cur, best_perm, best_val, moved, tabu, it0, n_iter, tmin, tmax,
                W, D, L, cq, allowed, si, sj, sd, k0, R, rho, seed):
    """Robust tabu search (Taillard, 1991): duyệt toàn bộ lân cận hoán đổi; nước đi bị cấm nếu
    cả hai nhóm quay về slot vừa rời gần đây, trừ khi cho nghiệm tốt nhất mới (aspiration)."""
    np.random.seed(seed)
    m = perm.shape[0]
    it = it0
    for _ in range(n_iter):
        it += 1
        bd = 1e300
        br = -1
        bs = -1
        bdm = 0
        for r in range(m - 1):
            a = perm[r]
            for s in range(r + 1, m):
                b = perm[s]
                if a == b or not (allowed[a, s] and allowed[b, r]):
                    continue
                d, dm = fastops.swap_delta_full(perm, loc, W, D, L, cq, si, sj, sd, k0, R, rho,
                                                moved, r, s)
                is_tabu = tabu[a, s] >= it and tabu[b, r] >= it
                aspir = cur + d < best_val - 1e-12
                if (not is_tabu or aspir) and d < bd:
                    bd = d
                    br = r
                    bs = s
                    bdm = dm
        if br < 0:
            continue
        a = perm[br]
        b = perm[bs]
        perm[br] = b
        perm[bs] = a
        loc[a] = bs
        loc[b] = br
        moved += bdm
        cur += bd
        tabu[a, br] = it + tmin + np.random.randint(tmax - tmin + 1)
        tabu[b, bs] = it + tmin + np.random.randint(tmax - tmin + 1)
        if cur < best_val - 1e-12:
            best_val = cur
            best_perm[:] = perm
    return cur, best_val, moved, it


def tabu_search(inst: Instance, alpha: float, time_limit: float = 10.0, seed: int = 0,
                init: np.ndarray | None = None, chunk: int | None = None, rho: float = 10.0) -> dict:
    rng = np.random.default_rng(seed)
    args, c0 = inst.kernel_args(alpha, rho)
    W, D, L, cq, allowed, si, sj, sd, k0, R, rho = args
    m = inst.m
    perm = (init.copy() if init is not None else random_perm(inst, rng)).astype(np.int64)
    loc = fastops.inverse(perm)
    moved = inst.moved(perm) if R >= 0 else 0
    cur = fastops.fitness(perm, *args)
    best_perm, best_val = perm.copy(), cur
    tabu = np.zeros((m, m), dtype=np.int64)
    tmin, tmax = max(1, int(0.9 * m)), max(2, int(1.1 * m))
    chunk = chunk or max(1, 200_000 // (m * m))
    t0 = time.time()
    it = 0
    history = [(0.0, best_val + c0)]
    while time.time() - t0 < time_limit:
        cur, best_val, moved, it = _tabu_chunk(perm, loc, cur, best_perm, best_val, moved, tabu, it,
                                               chunk, tmin, tmax, W, D, L, cq, allowed, si, sj, sd,
                                               k0, R, rho, int(rng.integers(2**31 - 1)))
        cur = fastops.fitness(perm, *args)      # tránh trôi số học
        history.append((time.time() - t0, best_val + c0))
    best_val = fastops.fitness(best_perm, *args)
    return {"perm": best_perm, "z": best_val + c0, "time": time.time() - t0, "iters": it,
            "history": history}
