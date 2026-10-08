"""Hàm mục tiêu theo định tuyến (độ tiếp xúc phụ thuộc sơ đồ) – "Cách A".

Với mỗi sơ đồ π, một mẫu giỏ hàng cố định (gom giỏ trùng, có trọng số) đi theo đúng quy tắc của
mô phỏng: cửa vào → các slot trong giỏ (láng giềng gần nhất + 2-opt) → quầy thu ngân gần nhất.
Từ cùng một lần định tuyến:
    Z1(π) = quãng đường trung bình mỗi khách (m)
    Z2(π) = doanh thu ngẫu hứng kỳ vọng mỗi khách
          = E_giỏ[ Σ_{slot k nằm trên lộ trình, nhóm j = π(k) ∉ giỏ} v_j · p_j · E(1 − e^{−λ T_j}) ],
            T_j ~ LogNormal(ln t_j, σ) (thời gian dừng như trong mô phỏng).
Độ tiếp xúc e_k(π) vì thế tự thay đổi theo sơ đồ (đúng với mô phỏng, khác e_k cố định của mô hình QAP).
Nhân numba; mỗi lần đánh giá ~1 ms với ~1.500 giỏ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .instance import Instance

ORDER_RULES = {"tsp": 0, "nn": 1, "rank": 2}    # gần nhất + 2-opt | gần nhất | theo hạng cố định (rắn, chu vi)


@dataclass
class LegTable:
    """Chặng a → b giữa các điểm (m slot, cửa vào m, khu tập kết m+1; đích npts = thu ngân), dạng CSR:
    slot idx[ptr[l]:ptr[l+1]] được đi qua với xác suất w; PL[a, b] = quãng đường kỳ vọng (m).
    Đường ngắn nhất: w = 1, PL = khoảng cách; recursive logit (behavior.py): w ∈ (0, 1]."""
    ptr: np.ndarray
    idx: np.ndarray
    w: np.ndarray
    PL: np.ndarray
    info: dict = field(default_factory=dict)


def shortest_legs(fp) -> LegTable:
    """Chặng theo đường ngắn nhất (có cache trên mặt bằng)."""
    hit = getattr(fp, "_sp_legs", None)
    if hit is not None:
        return hit
    npts = len(fp.points)
    M1 = npts + 1
    ptr, idx = [0], []
    PL = np.zeros((npts, M1))
    for a in range(npts):
        for b in range(M1):
            if b == npts:
                exps = fp.leg(a, -1)[1]
                PL[a, b] = fp.leg_length(a, -1)
            elif b == a:
                exps = ()
            else:
                exps = fp.leg(a, b)[1]
                PL[a, b] = fp.leg_length(a, b)
            idx.extend(int(x) for x in exps)
            ptr.append(len(idx))
    out = LegTable(np.array(ptr, dtype=np.int64), np.array(idx, dtype=np.int64), np.ones(len(idx)), PL,
                   {"kind": "sp"})
    fp._sp_legs = out
    return out


@njit(cache=True)
def _route_one(s0, L, b_idx, slot_of, cat_at, mark, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef, E, M1, FIN,
               order_rule, rank, seen, stamp, miss, touched, tps, stops, seq, used):
    """Định tuyến MỘT giỏ (ngành b_idx[s0:s0+L]; mark[j] = True với j trong giỏ do nơi gọi đặt).
    Trả (quãng đường, doanh thu ngẫu hứng, nt); touched[:nt] = slot đi qua, tps[:nt] = P(đi qua ≥ 1 lần)."""
    for t in range(L):
        stops[t] = slot_of[b_idx[s0 + t]]
        used[t] = False
    seq[0] = E
    cur = E
    for t in range(L):
        best = -1
        bd = 1e18
        for u in range(L):
            if not used[u]:
                # láng giềng gần nhất từ cửa vào (cùng thứ tự duyệt với simulate._route) hoặc hạng cố định
                d = PD[cur, stops[u]] if order_rule != 2 else float(rank[stops[u]])
                if d < bd:
                    bd = d
                    best = u
        used[best] = True
        cur = stops[best]
        seq[t + 1] = cur
    # 2-opt đường mở, đầu E, cuối là quầy thu ngân
    n_ = L + 1
    improved = order_rule == 0 and L >= 2
    while improved:
        improved = False
        for a in range(n_ - 2):
            for bb in range(a + 2, n_):
                i0 = seq[a]
                i1 = seq[a + 1]
                j0 = seq[bb]
                if bb + 1 < n_:
                    old = PD[i0, i1] + PD[j0, seq[bb + 1]]
                    new = PD[i0, j0] + PD[i1, seq[bb + 1]]
                else:
                    old = PD[i0, i1] + PC[j0]
                    new = PD[i0, j0] + PC[i1]
                if new < old - 1e-9:
                    lo = a + 1
                    hi = bb
                    while lo < hi:
                        tmp = seq[lo]
                        seq[lo] = seq[hi]
                        seq[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
    # quãng đường kỳ vọng + xác suất đi qua mỗi slot ít nhất một lần trong chuyến:
    # 1 − Π_chặng (1 − w) (giả định các chặng độc lập; đường ngắn nhất: w = 1 → đúng như cũ)
    d = 0.0
    nt = 0
    for t in range(L + 1):
        a = seq[t]
        nxt = seq[t + 1] if t < L else FIN    # đích cuối: thu ngân hoặc khu tập kết
        d += PL[a, nxt]
        lg = a * M1 + nxt
        for q in range(leg_ptr[lg], leg_ptr[lg + 1]):
            k = leg_idx[q]
            if seen[k] != stamp:
                seen[k] = stamp
                miss[k] = 1.0
                touched[nt] = k
                nt += 1
            miss[k] *= 1.0 - leg_w[q]
    rev = 0.0
    for u in range(nt):
        k = touched[u]
        ps = 1.0 - miss[k]
        tps[u] = ps
        j = cat_at[k]
        if j >= 0 and not mark[j]:
            rev += coef[j] * ps
    return d, rev, nt


@njit(cache=True)
def _buffers(b_ptr, m):
    maxL = 0
    for b in range(b_ptr.shape[0] - 1):
        if b_ptr[b + 1] - b_ptr[b] > maxL:
            maxL = b_ptr[b + 1] - b_ptr[b]
    return (np.empty(maxL + 1, np.int64), np.empty(maxL + 1, np.int64), np.zeros(maxL + 1, np.bool_),
            np.empty(m, np.float64))


@njit(cache=True)
def _evaluate(cat_at, slot_of, b_ptr, b_idx, b_w, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef, E, M1, FIN,
              order_rule, rank, seen, mark, expo, miss, touched):
    nb = b_ptr.shape[0] - 1
    tot_w = 0.0
    dist_sum = 0.0
    rev_sum = 0.0
    stops, seq, used, tps = _buffers(b_ptr, expo.shape[0])
    for k in range(expo.shape[0]):
        expo[k] = 0.0
    for b in range(nb):
        s0 = b_ptr[b]
        L = b_ptr[b + 1] - s0
        w = b_w[b]
        for t in range(L):
            mark[b_idx[s0 + t]] = True
        d, rev, nt = _route_one(s0, L, b_idx, slot_of, cat_at, mark, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef,
                                E, M1, FIN, order_rule, rank, seen, b + 1, miss, touched, tps, stops, seq, used)
        for u in range(nt):
            expo[touched[u]] += w * tps[u]
        for t in range(L):
            mark[b_idx[s0 + t]] = False
        tot_w += w
        dist_sum += w * d
        rev_sum += w * rev
    for k in range(expo.shape[0]):
        expo[k] /= tot_w
    for k in range(seen.shape[0]):
        seen[k] = 0
    return dist_sum / tot_w, rev_sum / tot_w


@njit(cache=True)
def _base_stats(cat_at, slot_of, b_ptr, b_idx, b_w, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef, E, M1, FIN,
                order_rule, rank, seen, mark, miss, touched, bd_out, brev_out, T, S):
    """Thống kê của sơ đồ gốc cho đánh giá chênh lệch: quãng đường, doanh thu từng giỏ;
    T[k] = Σ_b w P_b(k); S[k, i] = Σ_{b ∋ i} w P_b(k)."""
    stops, seq, used, tps = _buffers(b_ptr, T.shape[0])
    T[:] = 0.0
    S[:, :] = 0.0
    for b in range(b_ptr.shape[0] - 1):
        s0 = b_ptr[b]
        L = b_ptr[b + 1] - s0
        w = b_w[b]
        for t in range(L):
            mark[b_idx[s0 + t]] = True
        d, rev, nt = _route_one(s0, L, b_idx, slot_of, cat_at, mark, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef,
                                E, M1, FIN, order_rule, rank, seen, b + 1, miss, touched, tps, stops, seq, used)
        bd_out[b] = d
        brev_out[b] = rev
        for u in range(nt):
            k = touched[u]
            v = w * tps[u]
            T[k] += v
            for t in range(L):
                S[k, b_idx[s0 + t]] += v
        for t in range(L):
            mark[b_idx[s0 + t]] = False
    for k in range(seen.shape[0]):
        seen[k] = 0


@njit(cache=True)
def _delta(cat_old, slot_old, cat_new, slot_new, kr, ks, ci, cj, affected, both, b_ptr, b_idx, b_w, PD, PC, PL,
           leg_ptr, leg_idx, leg_w, coef, E, M1, FIN, order_rule, rank, seen, mark, miss, touched, bd, brev, T, S,
           D_tot, R_tot, tot_w, expo_out):
    """Đánh giá sơ đồ sau khi đổi chỗ hai slot kr ↔ ks (ngành ci, cj; −1 = rỗng) mà KHÔNG định tuyến lại
    các giỏ không chứa ci, cj: lộ trình của chúng không đổi, doanh thu chỉ đổi (coef_cj − coef_ci)(U_kr − U_ks)."""
    m = T.shape[0]
    stops, seq, used, tps = _buffers(b_ptr, m)
    baseA = np.zeros(m)
    if ci >= 0:
        baseA += S[:, ci]
    if cj >= 0:
        baseA += S[:, cj]
    stamp = 1
    for q in range(both.shape[0]):                       # giỏ chứa cả hai: đã bị cộng hai lần
        b = both[q]
        s0 = b_ptr[b]
        L = b_ptr[b + 1] - s0
        for t in range(L):
            mark[b_idx[s0 + t]] = True
        d, rev, nt = _route_one(s0, L, b_idx, slot_old, cat_old, mark, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef,
                                E, M1, FIN, order_rule, rank, seen, stamp, miss, touched, tps, stops, seq, used)
        stamp += 1
        for u in range(nt):
            baseA[touched[u]] -= b_w[b] * tps[u]
        for t in range(L):
            mark[b_idx[s0 + t]] = False
    newA = np.zeros(m)
    dA_old = 0.0
    rA_old = 0.0
    dA_new = 0.0
    rA_new = 0.0
    for q in range(affected.shape[0]):
        b = affected[q]
        s0 = b_ptr[b]
        L = b_ptr[b + 1] - s0
        w = b_w[b]
        dA_old += w * bd[b]
        rA_old += w * brev[b]
        for t in range(L):
            mark[b_idx[s0 + t]] = True
        d, rev, nt = _route_one(s0, L, b_idx, slot_new, cat_new, mark, PD, PC, PL, leg_ptr, leg_idx, leg_w, coef,
                                E, M1, FIN, order_rule, rank, seen, stamp, miss, touched, tps, stops, seq, used)
        stamp += 1
        dA_new += w * d
        rA_new += w * rev
        for u in range(nt):
            newA[touched[u]] += w * tps[u]
        for t in range(L):
            mark[b_idx[s0 + t]] = False
    for k in range(seen.shape[0]):
        seen[k] = 0
    c_i = coef[ci] if ci >= 0 else 0.0
    c_j = coef[cj] if cj >= 0 else 0.0
    u_r = T[kr] - baseA[kr]
    u_s = T[ks] - baseA[ks]
    rev_tot = (R_tot - rA_old) + (c_j - c_i) * (u_r - u_s) + rA_new
    d_tot = (D_tot - dA_old) + dA_new
    for k in range(m):
        expo_out[k] = (T[k] - baseA[k] + newA[k]) / tot_w
    return d_tot / tot_w, rev_tot / tot_w


def expected_purchase_prob(dwell_median_s: np.ndarray, lam: float, sigma: float, n_draws: int = 20000,
                           seed: int = 0) -> np.ndarray:
    """E[1 − exp(−λ T)], T ~ LogNormal(ln t_med, σ) – ước lượng Monte Carlo với hạt giống cố định."""
    z = np.random.default_rng(seed).standard_normal(n_draws)
    t = np.exp(np.log(np.asarray(dwell_median_s, float))[:, None] + sigma * z[None, :])
    return (1.0 - np.exp(-lam * t)).mean(axis=1)


class RouteModel:
    """Bộ đánh giá Z1, Z2 theo định tuyến cho một instance (dùng chung mặt bằng với Simulator)."""

    def __init__(self, inst: Instance, lam: float, sigma: float = 0.6, n_baskets: int = 1500,
                 p_scale: float = 1.0, seed: int = 0, origin: str = "entrance", dest: str = "checkout",
                 two_opt: bool = True, order: str | None = None, rank: np.ndarray | None = None,
                 legs: LegTable | None = None):
        """origin ∈ {entrance, staging}, dest ∈ {checkout, staging}: khách tại chỗ đi cửa vào → thu ngân;
        người nhặt đơn online (v4) đi khu tập kết → khu tập kết.
        order ∈ ORDER_RULES (mặc định "tsp", hoặc "nn" khi two_opt=False); rank: hạng của mỗi điểm khi
        order="rank"; legs: bảng chặng (mặc định đường ngắn nhất; recursive logit lấy từ behavior.py).
        Thứ tự dừng luôn lập trên khoảng cách ngắn nhất PD; quãng đường tính theo legs.PL."""
        self.inst = inst
        fp = inst.fp
        self.fp = fp
        m = fp.m
        npts = len(fp.points)                        # m slot + cửa vào (m) + khu tập kết (m + 1)
        self.E = {"entrance": m, "staging": m + 1}[origin]
        self.M1 = npts + 1                           # đích npts = quầy thu ngân
        self.FIN = {"checkout": npts, "staging": m + 1}[dest]
        order = order or ("tsp" if two_opt else "nn")
        self.order, self.order_rule = order, ORDER_RULES[order]
        if order == "rank" and rank is None:
            raise ValueError("order='rank' cần mảng rank")
        self.rank = (np.zeros(npts, dtype=np.int64) if rank is None else np.asarray(rank, dtype=np.int64))
        src_nodes = [fp.node_of[p] for p in fp.points]
        self.PD = np.ascontiguousarray(fp._dist[:, src_nodes], dtype=np.float64)
        if dest == "checkout":
            self.PC = np.array([fp.leg_length(a, -1) for a in range(npts)], dtype=np.float64)
        else:
            self.PC = np.ascontiguousarray(self.PD[:, self.FIN])
        self.legs = legs or shortest_legs(fp)
        self.leg_ptr, self.leg_idx = self.legs.ptr, self.legs.idx
        self.leg_w = np.ascontiguousarray(self.legs.w, dtype=np.float64)
        self.PL = np.ascontiguousarray(self.legs.PL, dtype=np.float64)
        # mẫu giỏ cố định, gom giỏ trùng
        B = inst.baskets
        rng = np.random.default_rng(seed)
        rows = rng.choice(B.shape[0], size=min(n_baskets, B.shape[0]), replace=False)
        cnt: dict = {}
        for r in rows:
            key = tuple(int(x) for x in B.indices[B.indptr[r]:B.indptr[r + 1]])
            cnt[key] = cnt.get(key, 0) + 1
        keys = list(cnt)
        self.b_ptr = np.cumsum([0] + [len(k) for k in keys]).astype(np.int64)
        self.b_idx = np.array([j for k in keys for j in k], dtype=np.int64)
        self.b_w = np.array([cnt[k] for k in keys], dtype=np.float64)
        n = inst.n
        g = expected_purchase_prob(inst.meta.dwell_median_s.values, lam, sigma)
        self.coef = np.clip(inst.p[:n] * p_scale, 0, 1) * g * inst.v[:n]
        self.lam, self.sigma = lam, sigma
        self._seen = np.zeros(m, dtype=np.int64)
        self._mark = np.zeros(n, dtype=np.bool_)
        self._expo = np.zeros(m, dtype=np.float64)
        self._miss = np.ones(m, dtype=np.float64)
        self._touched = np.zeros(m, dtype=np.int64)
        self._cache: dict = {}

    def _maps(self, perm: np.ndarray):
        inst, fp = self.inst, self.fp
        cat_at = -np.ones(fp.m, dtype=np.int64)
        real = perm < inst.n
        cat_at[inst.slot_idx[real]] = perm[real]
        slot_of = np.empty(inst.n, dtype=np.int64)
        slot_of[cat_at[cat_at >= 0]] = np.where(cat_at >= 0)[0]
        return cat_at, slot_of

    def _run(self, perm: np.ndarray):
        cat_at, slot_of = self._maps(perm)
        return _evaluate(cat_at, slot_of, self.b_ptr, self.b_idx, self.b_w, self.PD, self.PC, self.PL,
                         self.leg_ptr, self.leg_idx, self.leg_w, self.coef, self.E, self.M1, self.FIN,
                         self.order_rule, self.rank, self._seen, self._mark, self._expo, self._miss,
                         self._touched)

    # ------------------------------------------------ đánh giá chênh lệch (GĐ4)
    def _cat_index(self):
        """Chỉ số ngược ngành → các giỏ chứa ngành đó (CSR)."""
        if getattr(self, "_c_ptr", None) is None:
            n = self.inst.n
            nb = len(self.b_w)
            owner = np.repeat(np.arange(nb), np.diff(self.b_ptr))
            order = np.argsort(self.b_idx, kind="stable")
            self._c_idx = owner[order].astype(np.int64)
            self._c_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.b_idx, minlength=n))]).astype(np.int64)
        return self._c_ptr, self._c_idx

    def set_base(self, perm) -> None:
        """Ghi nhớ sơ đồ gốc và thống kê để `delta_swap` đánh giá các bước đổi chỗ từ nó."""
        perm = np.asarray(perm, dtype=np.int64).copy()
        cat_at, slot_of = self._maps(perm)
        nb, m, n = len(self.b_w), self.fp.m, self.inst.n
        self._bd = np.empty(nb)
        self._brev = np.empty(nb)
        self._T = np.empty(m)
        self._S = np.empty((m, n))
        _base_stats(cat_at, slot_of, self.b_ptr, self.b_idx, self.b_w, self.PD, self.PC, self.PL, self.leg_ptr,
                    self.leg_idx, self.leg_w, self.coef, self.E, self.M1, self.FIN, self.order_rule, self.rank,
                    self._seen, self._mark, self._miss, self._touched, self._bd, self._brev, self._T, self._S)
        self._base = perm
        self._base_maps = (cat_at, slot_of)
        self._D_tot = float(np.dot(self.b_w, self._bd))
        self._R_tot = float(np.dot(self.b_w, self._brev))
        self._W_tot = float(self.b_w.sum())
        self._cat_index()

    def delta_swap(self, r: int, s: int):
        """(quãng đường, doanh thu, độ tiếp xúc theo slot của instance) của sơ đồ gốc sau khi đổi r ↔ s
        (chỉ số slot của instance); chỉ định tuyến lại các giỏ chứa hai ngành bị đổi."""
        base, inst = self._base, self.inst
        n = inst.n
        cat_old, slot_old = self._base_maps
        ci = int(base[r]) if base[r] < n else -1
        cj = int(base[s]) if base[s] < n else -1
        kr, ks = int(inst.slot_idx[r]), int(inst.slot_idx[s])
        cat_new = cat_old.copy()
        cat_new[kr], cat_new[ks] = cj, ci
        slot_new = slot_old.copy()
        if ci >= 0:
            slot_new[ci] = ks
        if cj >= 0:
            slot_new[cj] = kr
        c_ptr, c_idx = self._c_ptr, self._c_idx
        bi = c_idx[c_ptr[ci]:c_ptr[ci + 1]] if ci >= 0 else np.zeros(0, np.int64)
        bj = c_idx[c_ptr[cj]:c_ptr[cj + 1]] if cj >= 0 else np.zeros(0, np.int64)
        affected = np.union1d(bi, bj).astype(np.int64)
        both = np.intersect1d(bi, bj).astype(np.int64)
        expo = np.empty(self.fp.m)
        d, rev = _delta(cat_old, slot_old, cat_new, slot_new, kr, ks, ci, cj, affected, both, self.b_ptr, self.b_idx,
                        self.b_w, self.PD, self.PC, self.PL, self.leg_ptr, self.leg_idx, self.leg_w, self.coef,
                        self.E, self.M1, self.FIN, self.order_rule, self.rank, self._seen, self._mark, self._miss,
                        self._touched, self._bd, self._brev, self._T, self._S, self._D_tot, self._R_tot,
                        self._W_tot, expo)
        return d, rev, expo[inst.slot_idx]

    def evaluate(self, perm) -> tuple[float, float]:
        """(Z1 = quãng đường TB, Z2 = doanh thu ngẫu hứng kỳ vọng TB)."""
        perm = np.asarray(perm, dtype=np.int64)
        key = perm.tobytes()
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) > 20000:
                self._cache.clear()
            hit = self._cache[key] = self._run(perm)
        return hit

    def exposure(self, perm) -> np.ndarray:
        """e_k(π): tỷ lệ khách (trong mẫu) đi qua vùng tiếp xúc của slot k – theo slot của instance."""
        self._run(np.asarray(perm, dtype=np.int64))
        return self._expo[self.inst.slot_idx].copy()


class RouteObjective:
    """Instance có Z1, Z2 thay bằng giá trị theo định tuyến; mọi thuộc tính khác (ràng buộc, toán tử
    tìm kiếm cục bộ của NSGA-II lai) lấy từ instance mô hình QAP bên dưới."""

    def __init__(self, base: Instance, rm: RouteModel, seeds: list[np.ndarray]):
        self._base = base
        self.rm = rm
        Z = np.array([rm.evaluate(p) for p in seeds])
        self.payoff = {"z1min": float(Z[:, 0].min()), "z1max": float(Z[:, 0].max()),
                       "z2min": float(Z[:, 1].min()), "z2max": float(Z[:, 1].max())}
        if self.payoff["z1max"] - self.payoff["z1min"] < 1e-9:
            self.payoff["z1max"] += 1e-6
        if self.payoff["z2max"] - self.payoff["z2min"] < 1e-9:
            self.payoff["z2min"] -= 1e-6

    def __getattr__(self, k):
        return getattr(self._base, k)

    def z1(self, perm) -> float:
        return self.rm.evaluate(perm)[0]

    def z2(self, perm) -> float:
        return self.rm.evaluate(perm)[1]
