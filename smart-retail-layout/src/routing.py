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

import numpy as np
from numba import njit

from .instance import Instance


@njit(cache=True)
def _evaluate(cat_at, slot_of, b_ptr, b_idx, b_w, PD, PC, leg_ptr, leg_idx, coef, E, M1, seen, mark,
              expo):
    nb = b_ptr.shape[0] - 1
    tot_w = 0.0
    dist_sum = 0.0
    rev_sum = 0.0
    maxL = 0
    for b in range(nb):
        if b_ptr[b + 1] - b_ptr[b] > maxL:
            maxL = b_ptr[b + 1] - b_ptr[b]
    stops = np.empty(maxL + 1, np.int64)
    seq = np.empty(maxL + 1, np.int64)
    used = np.zeros(maxL + 1, np.bool_)
    for k in range(expo.shape[0]):
        expo[k] = 0.0
    for b in range(nb):
        s0 = b_ptr[b]
        L = b_ptr[b + 1] - s0
        w = b_w[b]
        for t in range(L):
            j = b_idx[s0 + t]
            stops[t] = slot_of[j]
            mark[j] = True
            used[t] = False
        # láng giềng gần nhất từ cửa vào (cùng thứ tự duyệt với simulate._route)
        seq[0] = E
        cur = E
        for t in range(L):
            best = -1
            bd = 1e18
            for u in range(L):
                if not used[u]:
                    d = PD[cur, stops[u]]
                    if d < bd:
                        bd = d
                        best = u
            used[best] = True
            cur = stops[best]
            seq[t + 1] = cur
        # 2-opt đường mở, đầu E, cuối là quầy thu ngân
        n_ = L + 1
        improved = L >= 2
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
        # quãng đường + tiếp xúc (mỗi slot tính một lần mỗi khách)
        d = 0.0
        rev = 0.0
        stamp = b + 1
        for t in range(L + 1):
            a = seq[t]
            if t < L:
                nxt = seq[t + 1]
                d += PD[a, nxt]
            else:
                nxt = M1 - 1                      # đích "thu ngân"
                d += PC[a]
            lg = a * M1 + nxt
            for q in range(leg_ptr[lg], leg_ptr[lg + 1]):
                k = leg_idx[q]
                if seen[k] != stamp:
                    seen[k] = stamp
                    expo[k] += w
                    j = cat_at[k]
                    if j >= 0 and not mark[j]:
                        rev += coef[j]
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


def expected_purchase_prob(dwell_median_s: np.ndarray, lam: float, sigma: float, n_draws: int = 20000,
                           seed: int = 0) -> np.ndarray:
    """E[1 − exp(−λ T)], T ~ LogNormal(ln t_med, σ) – ước lượng Monte Carlo với hạt giống cố định."""
    z = np.random.default_rng(seed).standard_normal(n_draws)
    t = np.exp(np.log(np.asarray(dwell_median_s, float))[:, None] + sigma * z[None, :])
    return (1.0 - np.exp(-lam * t)).mean(axis=1)


class RouteModel:
    """Bộ đánh giá Z1, Z2 theo định tuyến cho một instance (dùng chung mặt bằng với Simulator)."""

    def __init__(self, inst: Instance, lam: float, sigma: float = 0.6, n_baskets: int = 1500,
                 p_scale: float = 1.0, seed: int = 0):
        self.inst = inst
        fp = inst.fp
        self.fp = fp
        m = fp.m
        self.E = m                                   # chỉ số điểm cửa vào (như Simulator)
        self.M1 = m + 1                              # đích m = quầy thu ngân
        src_nodes = [fp.node_of[p] for p in fp.points]
        self.PD = np.ascontiguousarray(fp._dist[:, src_nodes], dtype=np.float64)
        self.PC = np.array([fp.leg_length(a, -1) for a in range(m + 1)], dtype=np.float64)
        # danh sách slot tiếp xúc của mọi chặng a -> b (b = m: thu ngân), dạng CSR
        ptr, idx = [0], []
        for a in range(m + 1):
            for b in range(m + 1):
                if b == m:
                    exps = fp.leg(a, -1)[1]
                elif b == a:
                    exps = ()
                else:
                    exps = fp.leg(a, b)[1]
                idx.extend(int(x) for x in exps)
                ptr.append(len(idx))
        self.leg_ptr = np.array(ptr, dtype=np.int64)
        self.leg_idx = np.array(idx, dtype=np.int64)
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
        return _evaluate(cat_at, slot_of, self.b_ptr, self.b_idx, self.b_w, self.PD, self.PC, self.leg_ptr,
                         self.leg_idx, self.coef, self.E, self.M1, self._seen, self._mark, self._expo)

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
