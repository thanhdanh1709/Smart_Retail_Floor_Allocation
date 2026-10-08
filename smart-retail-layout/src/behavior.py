"""Tầng 4a – mô hình hành vi khách (kế hoạch v4 mục 5.1–5.3, GĐ3). Luồng khách chỉ DỰ ĐOÁN được.

Mỗi mô hình = (luật thứ tự điểm dừng, luật chọn đường mỗi chặng):
    SP    gần nhất + 2-opt (≈ TSP)     đường ngắn nhất
    NN    gần nhất kế tiếp             đường ngắn nhất
    SNK   theo đường rắn               đường ngắn nhất
    RL    gần nhất kế tiếp             recursive logit (Fosgerau, Frejinger & Karlström 2013)
    RL-A  gần nhất kế tiếp             RL + sức hút η của ô lối đi trước mặt kệ
    PER   theo vòng chu vi (góc)       RL + sức hút ζ của ô gần tường bao
    SUE   gần nhất kế tiếp             RL, chi phí ô tăng theo mật độ khách – cân bằng ngẫu nhiên bằng MSA

Recursive logit giải tích trên đồ thị lối đi (mỗi bước 1 m), cho đích D (một ô hoặc tập quầy thu ngân):
    M[n, n'] = exp(−μ·c(n') + bonus(n'))                      (đi sang ô kề n')
    z_T = (I − M_TT)^{-1} M_TD 1                              (tồn tại ⟺ bán kính phổ ρ(M_TT) < 1 ⟺ z_T > 0)
    Q[n, n'] = M[n, n'] z(n') / z(n)                          (xác suất chuyển; chuỗi Markov hút tại D)
    G = (I − Q)^{-1}                                          (số lần ghé kỳ vọng)
    quãng đường kỳ vọng a → D = Σ_n G(a, n)
    P(ghé ô c) = G(a, c) / G(c, c)                            (chính xác cho một ô)
    P(đi qua slot s) = max_{c ∈ vùng tiếp xúc s} G(a, c)/G(c, c)   (CẬN DƯỚI khi vùng có nhiều ô)
Chỉ phụ thuộc đồ thị (và μ, sức hút, chi phí) → tính MỘT LẦN cho mỗi mặt bằng, cache, rồi dùng trong GA.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .floorplan import FloorPlan
from .instance import Instance
from .routing import LegTable, RouteModel, shortest_legs

__all__ = ["BehaviorParams", "MODELS", "LegTable", "shortest_legs", "rl_legs", "simulate_leg",
           "calibrate_mu", "sue", "build_route_model", "stylized_facts"]

# mã -> (luật thứ tự, kiểu chặng)
MODELS = {"SP": ("tsp", "sp"), "NN": ("nn", "sp"), "SNK": ("snake", "sp"), "RL": ("nn", "rl"),
          "RL-A": ("nn", "rl-a"), "PER": ("perimeter", "rl-per"), "SUE": ("nn", "sue")}


@dataclass(frozen=True)
class BehaviorParams:
    mu: float | None = None          # độ "lý trí" của RL (1/m); None = hiệu chỉnh theo target_detour
    eta: float = 0.5                 # RL-A: sức hút của ô lối đi trước mặt kệ
    zeta: float = 1.0                # PER: sức hút của ô sát tường bao
    perimeter_band: int = 3          # PER: ô cách tường bao ≤ band là "chu vi"
    kappa: float = 1.0               # SUE: chi phí ô = 1 + κ·mật độ tương đối
    target_detour: float = 1.28      # quãng đường RL / quãng đường ngắn nhất (Lee et al., AAMAS 2026)
    sue_iters: int = 10
    thresh: float = 1e-4             # bỏ xác suất đi qua nhỏ hơn ngưỡng


# ------------------------------------------------------------- đồ thị
def _main_component(fp: FloorPlan) -> np.ndarray:
    _, lab = connected_components(fp.adj, directed=False)
    return lab == lab[fp.node_of[fp.entrance]]


def _zones(fp: FloorPlan) -> list[np.ndarray]:
    return [np.array([fp.node_of[c] for c in s.zone], dtype=np.int64) for s in fp.slots]


def shelf_bonus(fp: FloorPlan) -> np.ndarray:
    """1 cho ô lối đi nằm trong vùng tiếp xúc của ít nhất một slot (đi dọc mặt kệ)."""
    b = np.zeros(fp.adj.shape[0])
    for z in _zones(fp):
        b[z] = 1.0
    return b


def perimeter_bonus(fp: FloorPlan, band: int) -> np.ndarray:
    b = np.zeros(fp.adj.shape[0])
    for (r, c), n in fp.node_of.items():
        if min(r, fp.R - 1 - r, c, fp.C - 1 - c) <= band:
            b[n] = 1.0
    return b


def snake_rank(fp: FloorPlan) -> np.ndarray:
    rank = np.zeros(len(fp.points), dtype=np.int64)
    rank[fp.snake_order()] = np.arange(fp.m)
    return rank


def perimeter_rank(fp: FloorPlan) -> np.ndarray:
    """Hạng theo góc quanh tâm cửa hàng, bắt đầu từ hướng cửa vào, ngược chiều kim đồng hồ (đi vòng)."""
    cells = np.array(list(fp.node_of))
    rc, cc = cells[:, 0].mean(), cells[:, 1].mean()
    ang = lambda r, c: np.arctan2(-(r - rc), c - cc)         # noqa: E731 – hàng tăng xuống dưới
    a0 = ang(*fp.entrance)
    th = np.array([(ang(*fp.points[k]) - a0) % (2 * np.pi) for k in range(fp.m)])
    rank = np.zeros(len(fp.points), dtype=np.int64)
    rank[np.argsort(th, kind="stable")] = np.arange(fp.m)
    return rank


# ------------------------------------------------------------- recursive logit
def _utility_matrix(fp: FloorPlan, mu: float, bonus, node_cost) -> sp.csr_matrix:
    N = fp.adj.shape[0]
    cost = np.ones(N) if node_cost is None else np.asarray(node_cost, float)
    bon = np.zeros(N) if bonus is None else np.asarray(bonus, float)
    return sp.csr_matrix(fp.adj.multiply(np.exp(-mu * cost + bon)[None, :]))


def _absorbing_chain(M: sp.csr_matrix, keep: np.ndarray, dest: np.ndarray, mu: float):
    """(T, pos, z_T, Q, isD) cho chuỗi hút tại `dest`; lỗi nếu ρ(M_TT) ≥ 1."""
    N = M.shape[0]
    isD = np.zeros(N, dtype=bool)
    isD[dest] = True
    T = np.where(keep & ~isD)[0]
    pos = -np.ones(N, dtype=np.int64)
    pos[T] = np.arange(len(T))
    MT = M[T]
    M_TT = MT[:, T]
    rhs = np.asarray(MT[:, dest].sum(axis=1)).ravel()
    I = sp.identity(len(T), format="csc")
    try:
        z = splu((I - M_TT).tocsc()).solve(rhs)
    except RuntimeError as e:                       # ma trận suy biến
        raise ValueError(f"μ = {mu}: I − M suy biến (bán kính phổ ≥ 1)") from e
    if not np.all(np.isfinite(z)) or z.min() <= 0:
        raise ValueError(f"μ = {mu}: bán kính phổ của M ≥ 1 – tổng tiện ích các đường vòng phân kỳ; tăng μ")
    Q = sp.diags(1.0 / z) @ M_TT @ sp.diags(z)
    return T, pos, z, Q, isD


def _dest_nodes(fp: FloorPlan, b: int) -> np.ndarray:
    npts = len(fp.points)
    if b == npts:
        return np.array([fp.node_of[c] for c in fp.checkouts], dtype=np.int64)
    return np.array([fp.node_of[fp.points[b]]], dtype=np.int64)


def rl_legs(fp: FloorPlan, mu: float, bonus=None, node_cost=None, thresh: float = 1e-4,
            method: str = "schur") -> LegTable:
    """Bảng chặng recursive logit cho mọi cặp điểm (giải tích, không nhiễu).
    method="schur": nghịch đảo G = (I − M)^{-1} trên toàn đồ thị MỘT lần; với mỗi đích D dùng phần bù Schur
        (I − M_TT)^{-1} = G_TT − G_TD G_DD^{-1} G_DT  → mỗi đích chỉ vài phép nhân ma trận–vectơ O(N²).
    method="sparse": giải trực tiếp cho từng đích (đối chứng trong test)."""
    if method == "sparse":
        return _rl_legs_sparse(fp, mu, bonus, node_cost, thresh)
    if method != "schur":
        raise ValueError(method)
    M = _utility_matrix(fp, mu, bonus, node_cost)
    keep = _main_component(fp)
    K = np.where(keep)[0]
    nk = len(K)
    kpos = -np.ones(M.shape[0], dtype=np.int64)
    kpos[K] = np.arange(nk)
    Mk = M[K][:, K].toarray()
    try:
        G = np.linalg.inv(np.eye(nk) - Mk)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"μ = {mu}: I − M suy biến (bán kính phổ ≥ 1)") from e
    npts = len(fp.points)
    M1 = npts + 1
    pk = kpos[np.array([fp.node_of[p] for p in fp.points], dtype=np.int64)]
    zones = [kpos[z] for z in _zones(fp)]
    zcells = np.unique(np.concatenate(zones))
    zpad = np.full((fp.m, max(len(z) for z in zones)), nk, dtype=np.int64)   # chỉ số nk = hàng 0 đệm
    for s, z in enumerate(zones):
        zpad[s, :len(z)] = z
    PL = np.zeros((npts, M1))
    legs: dict = {}
    for b in range(M1):
        Dn = kpos[_dest_nodes(fp, b)]
        isD = np.zeros(nk, dtype=bool)
        isD[Dn] = True
        GDDi = np.linalg.inv(G[np.ix_(Dn, Dn)])
        G_D = G[Dn]                                   # |D| × nk

        def gt(v):                                    # (I − M_TT)^{-1} v trên T (v = 0 tại D)
            return G @ v - G[:, Dn] @ (GDDi @ (G_D @ v))

        r = Mk[:, Dn].sum(axis=1)
        r[isD] = 0.0
        z = gt(r)
        zT = z[~isD]
        if not np.all(np.isfinite(zT)) or zT.min() <= 0:
            raise ValueError(f"μ = {mu}: bán kính phổ của M ≥ 1 – tổng tiện ích các đường vòng phân kỳ; tăng μ")
        z[isD] = 0.0
        y = gt(z)                                     # PL(a) = Σ_n G_Q(a, n) = (G_T z)(a) / z(a)
        orig = [a for a in range(npts) if a != b and not isD[pk[a]]]
        pa = pk[orig]
        PL[orig, b] = y[pa] / z[pa]
        zc = zcells[~isD[zcells]]
        GTac = G[np.ix_(pa, zc)] - G[np.ix_(pa, Dn)] @ GDDi @ G_D[:, zc]
        GTcc = G[zc, zc] - np.einsum("cd,dc->c", G[np.ix_(zc, Dn)] @ GDDi, G_D[:, zc])
        H = np.zeros((nk + 1, len(orig)))            # H[c, j] = P(ghé c | a_j) = G_T(a,c) z(c) / (z(a) G_T(c,c))
        H[zc] = (GTac * (z[zc] / GTcc)[None, :] / z[pa][:, None]).T
        H[Dn] = 1.0
        PS = np.minimum(1.0, H[zpad].max(axis=1))
        for j, a in enumerate(orig):
            k = np.where(PS[:, j] > thresh)[0]
            legs[a, b] = (k, PS[k, j])
        for a in range(npts):                         # điểm trùng ô đích (≠ chính nó): chỉ ô đó
            if a != b and isD[pk[a]]:
                ps = np.array([1.0 if isD[z].any() else 0.0 for z in zones])
                k = np.where(ps > 0)[0]
                legs[a, b] = (k, ps[k])
    return _assemble(legs, PL, npts, M1, mu)


def _assemble(legs: dict, PL: np.ndarray, npts: int, M1: int, mu: float) -> LegTable:
    ptr, idx, w = [0], [], []
    for a in range(npts):
        for b in range(M1):
            k, p = legs.get((a, b), ((), ()))
            idx.extend(int(x) for x in k)
            w.extend(float(x) for x in p)
            ptr.append(len(idx))
    return LegTable(np.array(ptr, dtype=np.int64), np.array(idx, dtype=np.int64), np.array(w), PL,
                    {"kind": "rl", "mu": mu})


def _rl_legs_sparse(fp: FloorPlan, mu: float, bonus=None, node_cost=None, thresh: float = 1e-4) -> LegTable:
    M = _utility_matrix(fp, mu, bonus, node_cost)
    keep = _main_component(fp)
    npts = len(fp.points)
    M1 = npts + 1
    pnode = np.array([fp.node_of[p] for p in fp.points], dtype=np.int64)
    zones = _zones(fp)
    zcells = np.unique(np.concatenate(zones))
    N = M.shape[0]
    zpad = np.full((fp.m, max(len(z) for z in zones)), N, dtype=np.int64)   # chỉ số N = hàng 0 đệm
    for s, z in enumerate(zones):
        zpad[s, :len(z)] = z
    PL = np.zeros((npts, M1))
    legs: dict = {}
    for b in range(M1):
        dest = _dest_nodes(fp, b)
        T, pos, _, Q, isD = _absorbing_chain(M, keep, dest, mu)
        lu = splu((sp.identity(len(T), format="csc") - Q).tocsc())
        orig = [a for a in range(npts) if a != b and not isD[pnode[a]]]
        E = np.zeros((len(T), len(orig)))
        E[pos[pnode[orig]], np.arange(len(orig))] = 1.0
        X = lu.solve(E, trans="T")                   # cột j: G(a_j, ·)
        zc = zcells[(pos[zcells] >= 0)]
        Ez = np.zeros((len(T), len(zc)))
        Ez[pos[zc], np.arange(len(zc))] = 1.0
        Gcc = lu.solve(Ez)[pos[zc], np.arange(len(zc))]
        PL[orig, b] = X.sum(axis=0)
        H = np.zeros((N + 1, len(orig)))             # H[c, j] = P(ghé ô c | xuất phát a_j)
        H[zc] = X[pos[zc]] / Gcc[:, None]
        H[dest] = 1.0
        PS = np.minimum(1.0, H[zpad].max(axis=1))    # (m, |orig|)
        for j, a in enumerate(orig):
            k = np.where(PS[:, j] > thresh)[0]
            legs[a, b] = (k, PS[k, j])
        for a in range(npts):                        # điểm trùng ô đích (≠ chính nó): chỉ ô đó
            if a != b and isD[pnode[a]]:
                ps = np.array([1.0 if isD[z].any() else 0.0 for z in zones])
                k = np.where(ps > 0)[0]
                legs[a, b] = (k, ps[k])
    return _assemble(legs, PL, npts, M1, mu)


def simulate_leg(fp: FloorPlan, mu: float, a: int, b: int, n: int, rng: np.random.Generator,
                 bonus=None, node_cost=None) -> tuple[np.ndarray, np.ndarray]:
    """Kiểm chứng Monte Carlo: n đường đi RL a → b; trả (quãng đường, tần suất đi qua từng slot)."""
    M = _utility_matrix(fp, mu, bonus, node_cost)
    dest = _dest_nodes(fp, b)
    T, pos, z, Q, isD = _absorbing_chain(M, _main_component(fp), dest, mu)
    zfull = np.zeros(M.shape[0])
    zfull[T] = z
    zfull[dest] = 1.0
    nbrs = [M.indices[M.indptr[i]:M.indptr[i + 1]] for i in range(M.shape[0])]
    wts = [M.data[M.indptr[i]:M.indptr[i + 1]] * zfull[nbrs[i]] for i in range(M.shape[0])]
    zones = _zones(fp)
    slot_of_node: dict = {}
    for s, zn in enumerate(zones):
        for c in zn:
            slot_of_node.setdefault(int(c), []).append(s)
    start = fp.node_of[fp.points[a]]
    lengths = np.zeros(n)
    hits = np.zeros(fp.m)
    for t in range(n):
        cur, steps, seen = start, 0, set(slot_of_node.get(start, ()))
        while not isD[cur]:
            p = wts[cur] / wts[cur].sum()
            cur = int(nbrs[cur][rng.choice(len(p), p=p)])
            steps += 1
            seen.update(slot_of_node.get(cur, ()))
        lengths[t] = steps
        hits[list(seen)] += 1
    return lengths, hits / n


# ------------------------------------------------------------- mô hình đầy đủ
def _legs_for(inst: Instance, kind: str, par: BehaviorParams, node_cost=None) -> LegTable:
    fp = inst.fp
    if kind == "sp":
        return shortest_legs(fp)
    if par.mu is None:
        raise ValueError("Cần μ (hiệu chỉnh bằng calibrate_mu trước)")
    cache = fp.__dict__.setdefault("_rl_cache", {})
    key = (kind, round(par.mu, 9), par.eta, par.zeta, par.perimeter_band,
           None if node_cost is None else np.asarray(node_cost).tobytes())
    if key not in cache:
        cache[key] = rl_legs(fp, par.mu, bonus=_bonus(fp, kind, par), node_cost=node_cost, thresh=par.thresh)
    return cache[key]


def _order(fp: FloorPlan, rule: str):
    if rule in ("tsp", "nn"):
        return rule, None
    return "rank", snake_rank(fp) if rule == "snake" else perimeter_rank(fp)


def build_route_model(inst: Instance, model: str, par: BehaviorParams, lam: float, n_baskets: int = 1500,
                      seed: int = 0, sigma: float = 0.6, node_cost=None) -> RouteModel:
    if model not in MODELS:
        raise ValueError(f"Mô hình {model!r} không có; chọn trong {list(MODELS)}")
    rule, kind = MODELS[model]
    order, rank = _order(inst.fp, rule)
    if kind == "sue" and node_cost is None:
        return sue(inst, par, lam, n_baskets, seed=seed, sigma=sigma)["model"]
    legs = _legs_for(inst, kind, par, node_cost)
    return RouteModel(inst, lam, sigma, n_baskets, seed=seed, order=order, rank=rank, legs=legs)


def _base_of(model: str) -> str:
    """Mô hình đường ngắn nhất cùng luật thứ tự – mẫu số của độ lệch quãng đường."""
    return {"nn": "NN", "tsp": "SP", "snake": "SNK"}.get(MODELS[model][0], "PER-SP")


def _trip_length(inst: Instance, model: str, par: BehaviorParams, n_baskets: int, seed: int,
                 node_cost=None) -> float:
    if model == "PER-SP":
        order, rank = _order(inst.fp, "perimeter")
        rm = RouteModel(inst, 0.05, 0.6, n_baskets, seed=seed, order=order, rank=rank)
    else:
        rm = build_route_model(inst, model, par, 0.05, n_baskets, seed, node_cost=node_cost)
    return rm.evaluate(np.asarray(inst.current))[0]


def detour_ratio(inst: Instance, model: str, par: BehaviorParams, n_baskets: int = 300, seed: int = 0,
                 node_cost=None) -> float:
    return (_trip_length(inst, "RL" if model == "SUE" else model, par, n_baskets, seed, node_cost)
            / _trip_length(inst, _base_of(model), par, n_baskets, seed))


def _bonus(fp: FloorPlan, kind: str, par: BehaviorParams):
    return {"rl": None, "sue": None, "rl-a": par.eta * shelf_bonus(fp),
            "rl-per": par.zeta * perimeter_bonus(fp, par.perimeter_band)}[kind]


def rl_exists(fp: FloorPlan, mu: float, bonus=None, node_cost=None) -> bool:
    """ρ(M_TT) < 1 với mọi đích (chỉ giải z, không tính G)."""
    M = _utility_matrix(fp, mu, bonus, node_cost)
    keep = _main_component(fp)
    try:
        for b in range(len(fp.points) + 1):
            _absorbing_chain(M, keep, _dest_nodes(fp, b), mu)
    except ValueError:
        return False
    return True


def critical_mu(inst: Instance, model: str, par: BehaviorParams, hi: float = 10.0, tol: float = 1e-3) -> float:
    """μ nhỏ nhất để RL tồn tại (ρ < 1), tìm bằng chia đôi."""
    kind = MODELS["RL" if model == "SUE" else model][1]
    bonus = _bonus(inst.fp, kind, par)
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rl_exists(inst.fp, mid, bonus):
            hi = mid
        else:
            lo = mid
    return hi


def calibrate_mu(inst: Instance, model: str, par: BehaviorParams, target: float | None = None,
                 n_baskets: int = 300, seed: int = 0, tol: float = 2e-3, max_iter: int = 30) -> float:
    """Khớp mô-men (mục 5.2): chọn μ để quãng đường chuyến theo mô hình / theo đường ngắn nhất (cùng thứ tự
    dừng) = target (mặc định 1,28 – Lee et al.). Độ lệch giảm đơn điệu theo μ → chia đôi."""
    target = target or par.target_detour
    lo = critical_mu(inst, model, par) * 1.0005
    f = lambda mu: detour_ratio(inst, model, replace(par, mu=mu), n_baskets, seed) - target   # noqa: E731
    if f(lo) < 0:
        return lo                                  # ngay sát ngưỡng vẫn chưa đủ lệch → trả về biên
    hi = lo + 1.0
    while f(hi) > 0:
        hi += 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if abs(v) < tol:
            return mid
        lo, hi = (mid, hi) if v > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def sue(inst: Instance, par: BehaviorParams, lam: float, n_baskets: int = 1500, seed: int = 0,
        sigma: float = 0.6, iters: int | None = None) -> dict:
    """Cân bằng ngẫu nhiên (SUE) bằng MSA trên chi phí ô: c ← c + (c* − c)/t, c* = 1 + κ·ρ/ρ̄ với ρ = mật độ
    khách đi qua vùng tiếp xúc (từ định tuyến RL trên sơ đồ hiện trạng). gaps = ‖c* − c‖/‖c‖ (độ lệch cân bằng)."""
    fp = inst.fp
    N = fp.adj.shape[0]
    zones = _zones(fp)
    cost = np.ones(N)
    gaps, steps = [], []
    cur = np.asarray(inst.current, dtype=np.int64)
    for t in range(1, (iters or par.sue_iters) + 1):
        legs = _legs_for(inst, "sue", par, cost)
        rm = RouteModel(inst, lam, sigma, n_baskets, seed=seed, order="nn", legs=legs)
        rm._run(cur)
        dens = np.zeros(N)
        for s, z in enumerate(zones):
            dens[z] += rm._expo[s]
        ref = dens[dens > 0].mean() if (dens > 0).any() else 1.0
        target = 1.0 + par.kappa * dens / ref
        gaps.append(float(np.linalg.norm(target - cost) / np.linalg.norm(cost)))
        new = cost + (target - cost) / t
        steps.append(float(np.linalg.norm(new - cost) / np.linalg.norm(cost)))
        cost = new
    legs = _legs_for(inst, "sue", par, cost)
    model = RouteModel(inst, lam, sigma, n_baskets, seed=seed, order="nn", legs=legs)
    return {"model": model, "node_cost": cost, "gaps": gaps, "steps": steps}


def stylized_facts(rm: RouteModel, base: RouteModel, perm) -> dict:
    """Chỉ tiêu so với sự thật cách điệu: độ lệch quãng đường, tỷ lệ slot chu vi được đi qua."""
    perm = np.asarray(perm, dtype=np.int64)
    fp = rm.fp
    e = rm.exposure(perm)
    band = perimeter_bonus(fp, 3)
    per = np.array([band[z].max() > 0 for z in _zones(fp)])[rm.inst.slot_idx]
    return {"detour": rm.evaluate(perm)[0] / base.evaluate(perm)[0],
            "perimeter_share": float(e[per].sum() / max(e.sum(), 1e-12)),
            "mean_exposure": float(e.mean())}
