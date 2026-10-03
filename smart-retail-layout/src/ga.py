"""Thuật toán di truyền có tìm kiếm cục bộ cho bài toán gán nhóm hàng (mục B5).

Mã hóa hoán vị (perm[k] = nhóm ở slot k), khởi tạo 90% ngẫu nhiên + 10% tham lam,
chọn lọc tournament (k = 3), lai ghép OX hoặc PMX, đột biến hoán đổi/đảo đoạn,
sửa lỗi tương thích, tinh hoa, tìm kiếm cục bộ 2-swap trên 10% cá thể tốt nhất dùng Δ(r, s).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import fastops
from .instance import Instance, random_perm, repair


@dataclass
class GAConfig:
    pop_size: int = 100
    tournament_k: int = 3
    crossover: str = "ox"            # "ox" | "pmx"
    pc: float = 0.9
    mutation: str = "swap"           # "swap" | "inversion" | "mixed"
    pm: float = 0.2
    elite: int = 2
    ls_frac: float = 0.10            # tỷ lệ cá thể được tìm kiếm cục bộ mỗi thế hệ
    ls_max_moves: int = 0            # 0 = tới cực tiểu địa phương
    greedy_frac: float = 0.10
    immigrant_every: int = 20        # trì trệ bao nhiêu thế hệ thì chèn cá thể mới (0 = tắt)
    immigrant_frac: float = 0.5      # tỷ lệ quần thể được thay
    max_gens: int = 1000
    stall_gens: int = 100
    time_limit: float = 0.0          # giây; 0 = không giới hạn
    rho: float = 10.0                # hệ số phạt
    seed: int = 0


@dataclass
class GAResult:
    perm: np.ndarray
    fitness: float
    z: float
    history: list = field(default_factory=list)   # (thế hệ, thời gian, fitness tốt nhất)
    gens: int = 0
    evals: int = 0
    time: float = 0.0


# ------------------------------------------------------------- toán tử numba
@njit(cache=True)
def ox(p1, p2, a, b):
    """Order Crossover: giữ đoạn [a, b] của p1, phần còn lại theo thứ tự xuất hiện trong p2."""
    m = p1.shape[0]
    child = -np.ones(m, dtype=np.int64)
    used = np.zeros(m, dtype=np.bool_)
    for k in range(a, b + 1):
        child[k] = p1[k]
        used[p1[k]] = True
    pos = (b + 1) % m
    for t in range(m):
        g = p2[(b + 1 + t) % m]
        if not used[g]:
            child[pos] = g
            used[g] = True
            pos = (pos + 1) % m
    return child


@njit(cache=True)
def pmx(p1, p2, a, b):
    """Partially Mapped Crossover."""
    m = p1.shape[0]
    child = p2.copy()
    pos2 = np.empty(m, dtype=np.int64)
    for k in range(m):
        pos2[p2[k]] = k
    for k in range(a, b + 1):
        g = p1[k]
        j = pos2[g]
        # đổi chỗ trong child để đặt g vào k
        child[j] = child[k]
        pos2[child[k]] = j
        child[k] = g
        pos2[g] = k
    return child


def _mutate(c: np.ndarray, kind: str, rng: np.random.Generator) -> None:
    m = len(c)
    r, s = rng.choice(m, 2, replace=False)
    if kind == "mixed":
        kind = "swap" if rng.random() < 0.5 else "inversion"
    if kind == "swap":
        c[r], c[s] = c[s], c[r]
    else:
        lo, hi = min(r, s), max(r, s)
        c[lo:hi + 1] = c[lo:hi + 1][::-1].copy()


def greedy_perm(inst: Instance, alpha: float, rng: np.random.Generator | None = None,
                noise: float = 0.0) -> np.ndarray:
    """Khởi tạo tham lam (B5.1): đặt lần lượt các nhóm theo độ phổ biến giảm dần vào slot hợp lệ
    làm tăng hàm mục tiêu ít nhất (gồm phần tuyến tính – v·p·e và luồng với các nhóm đã đặt)."""
    L, cq, _ = inst.weighted(alpha)
    n, m = inst.n, inst.m
    order = np.argsort(-(inst.f[:n] + (rng.random(n) * noise if rng is not None else 0)))
    perm = -np.ones(m, dtype=np.int64)
    loc = -np.ones(m, dtype=np.int64)
    placed: list[int] = []
    free = np.ones(m, dtype=bool)
    for i in order:
        cand = np.where(free & inst.allowed[i])[0]
        cost = L[i, cand].copy()
        if placed:
            pl = np.array(placed)
            cost += cq * (inst.W[i, pl][None, :] * inst.D[np.ix_(cand, loc[pl])]).sum(axis=1)
        if rng is not None and noise > 0:
            cost += rng.random(len(cand)) * noise * (abs(cost).mean() + 1e-12)
        k = cand[np.argmin(cost)]
        perm[k] = i
        loc[i] = k
        free[k] = False
        placed.append(i)
    rest = iter(range(n, m))
    for k in range(m):
        if perm[k] < 0:
            perm[k] = next(rest)
    return repair(perm, inst.allowed)


# ----------------------------------------------------------------- vòng lặp
def run(inst: Instance, alpha: float = 0.5, cfg: GAConfig | None = None,
        init: list[np.ndarray] | None = None, verbose: bool = False) -> GAResult:
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(cfg.seed)
    args, c0 = inst.kernel_args(alpha, cfg.rho)
    W, D, L, cq, allowed, si, sj, sd, k0, R, rho = args
    t0 = time.time()

    def fit(p):
        return fastops.fitness(p, *args)

    N = cfg.pop_size
    pop: list[np.ndarray] = [np.asarray(p, dtype=np.int64).copy() for p in (init or [])][:N]
    n_greedy = max(1, int(round(cfg.greedy_frac * N))) if cfg.greedy_frac > 0 else 0
    for g in range(n_greedy):
        if len(pop) >= N:
            break
        pop.append(greedy_perm(inst, alpha, rng, noise=0.0 if g == 0 else 0.3))
    while len(pop) < N:
        pop.append(random_perm(inst, rng))
    fits = np.array([fit(p) for p in pop])
    evals = N
    best_i = int(np.argmin(fits))
    best, best_f = pop[best_i].copy(), float(fits[best_i])
    history = [(0, 0.0, best_f + c0)]
    stall = 0
    n_ls = max(1, int(round(cfg.ls_frac * N))) if cfg.ls_frac > 0 else 0
    max_moves = cfg.ls_max_moves or 10 * inst.m
    m = inst.m
    gen = 0
    for gen in range(1, cfg.max_gens + 1):
        order = np.argsort(fits)
        elite = [pop[i].copy() for i in order[:cfg.elite]]
        elite_f = [float(fits[i]) for i in order[:cfg.elite]]
        children: list[np.ndarray] = []
        while len(children) < N - cfg.elite:
            p1 = pop[_tournament(fits, cfg.tournament_k, rng)]
            p2 = pop[_tournament(fits, cfg.tournament_k, rng)]
            if rng.random() < cfg.pc:
                a, b = sorted(rng.choice(m, 2, replace=False))
                c = (ox if cfg.crossover == "ox" else pmx)(p1, p2, int(a), int(b))
            else:
                c = p1.copy()
            if rng.random() < cfg.pm:
                _mutate(c, cfg.mutation, rng)
            c = repair(c, allowed, rng)
            children.append(c)
        cf = np.array([fit(c) for c in children])
        evals += len(children)
        if n_ls:
            for idx in np.argsort(cf)[:n_ls]:
                fastops.local_search(children[idx], W, D, L, cq, allowed, si, sj, sd, k0, R, rho,
                                     max_moves, True)
                cf[idx] = fit(children[idx])
        pop = elite + children
        fits = np.concatenate([elite_f, cf])
        gi = int(np.argmin(fits))
        if fits[gi] < best_f - 1e-12:
            best, best_f = pop[gi].copy(), float(fits[gi])
            stall = 0
        else:
            stall += 1
        if cfg.immigrant_every and stall and stall % cfg.immigrant_every == 0:
            # chống hội tụ sớm (B10): thay phần kém nhất bằng cá thể ngẫu nhiên
            # và bản sao của cá thể tốt nhất bị xáo trộn mạnh (m/5 lần hoán đổi)
            worst = np.argsort(fits)[::-1][: int(cfg.immigrant_frac * N)]
            for t, idx in enumerate(worst):
                if t % 2 == 0:
                    c = random_perm(inst, rng)
                else:
                    c = best.copy()
                    for _ in range(max(2, m // 5)):
                        _mutate(c, "swap", rng)
                    c = repair(c, allowed, rng)
                if n_ls:
                    fastops.local_search(c, W, D, L, cq, allowed, si, sj, sd, k0, R, rho,
                                         max_moves, True)
                pop[idx] = c
                fits[idx] = fit(c)
                if fits[idx] < best_f - 1e-12:
                    best, best_f = c.copy(), float(fits[idx])
                    stall = 0
            evals += len(worst)
        el = time.time() - t0
        history.append((gen, el, best_f + c0))
        if verbose and gen % 50 == 0:
            print(f"  thế hệ {gen}: Z = {best_f + c0:.6f}  ({el:.1f}s)")
        if stall >= cfg.stall_gens or (cfg.time_limit and el >= cfg.time_limit):
            break
    return GAResult(perm=best, fitness=best_f, z=best_f + c0, history=history, gens=gen,
                    evals=evals, time=time.time() - t0)


def _tournament(fits: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idx = rng.choice(len(fits), k, replace=False)
    return int(idx[np.argmin(fits[idx])])
