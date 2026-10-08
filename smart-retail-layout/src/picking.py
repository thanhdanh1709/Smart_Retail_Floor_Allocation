"""Tầng 4b – nhặt đơn online: luồng ĐIỀU KHIỂN ĐƯỢC (kế hoạch v4 mục 5.4, GĐ5).

    Định tuyến từng đơn (khu tập kết → các slot trong đơn → khu tập kết), trên khoảng cách đồ thị lối đi:
        exact        Held-Karp (quy hoạch động, chính xác) khi ≤ 15 điểm dừng (98% đơn mức nhóm, 94% mức aisle);
                     đơn lớn hơn: gần nhất + 2-opt + Or-opt (đo khoảng cách với lời giải chính xác trên đơn nhỏ)
        nn2opt       gần nhất + 2-opt (chính là mô hình người nhặt trong FlowBank khi tối ưu sơ đồ)
        sshape       S-shape: đi rắn qua các "dải lối" chứa món, mỗi dải đi hết một chiều
        largest_gap  largest gap: mỗi dải tách tại khoảng trống lớn nhất, phần gần lấy lúc đi, phần xa lấy lúc về
      (S-shape/largest gap sinh THỨ TỰ dừng; quãng đường đo bằng đường ngắn nhất giữa các điểm dừng liên tiếp.)
    Gộp đơn (batching): xe chở tối đa B đơn; tiết kiệm Clarke–Wright trên các cặp đơn gần nhau.
    Tránh khách: chi phí vào ô = 1 + κ·mật độ khách tương đối → đường cong (quãng đường, chạm mặt) theo κ.
    Lịch nhặt theo giờ: bài vận tải (LP, nghiệm nguyên do ma trận unimodular hoàn toàn) – đơn đặt giờ h nhặt trong
        [h, h + Δ] (đơn ban đêm: sáng hôm sau), sức chứa K đơn/giờ, cực tiểu Σ (số đơn nhặt giờ t)·λ_W(t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit
from scipy import stats
from scipy.optimize import linprog
from scipy.sparse.csgraph import dijkstra

from .instance import Instance

MAX_EXACT = 15


# ------------------------------------------------------------- TSP một đơn
@njit(cache=True)
def _held_karp(C):
    n = C.shape[0] - 1
    order = np.empty(n, np.int64)
    if n == 0:
        return 0.0, order
    FULL = 1 << n
    dp = np.full((FULL, n), np.inf)
    par = -np.ones((FULL, n), np.int64)
    for j in range(n):
        dp[1 << j, j] = C[0, j + 1]
    for S in range(1, FULL):
        for j in range(n):
            if not (S >> j) & 1:
                continue
            v = dp[S, j]
            if v == np.inf:
                continue
            for k in range(n):
                if (S >> k) & 1:
                    continue
                S2 = S | (1 << k)
                nv = v + C[j + 1, k + 1]
                if nv < dp[S2, k]:
                    dp[S2, k] = nv
                    par[S2, k] = j
    best = np.inf
    last = -1
    for j in range(n):
        v = dp[FULL - 1, j] + C[j + 1, 0]
        if v < best:
            best = v
            last = j
    S = FULL - 1
    j = last
    for t in range(n - 1, -1, -1):
        order[t] = j + 1
        pj = par[S, j]
        S ^= 1 << j
        j = pj
    return best, order


@njit(cache=True)
def _tour_len(C, seq):
    s = 0.0
    for t in range(seq.shape[0] - 1):
        s += C[seq[t], seq[t + 1]]
    return s


@njit(cache=True)
def _heuristic(C, or_opt):
    """Gần nhất + 2-opt (+ Or-opt đoạn 1–3) cho chu trình qua điểm 0. Trả (độ dài, thứ tự 1..n)."""
    n = C.shape[0] - 1
    seq = np.empty(n + 2, np.int64)
    used = np.zeros(n + 1, np.bool_)
    seq[0] = 0
    used[0] = True
    cur = 0
    for t in range(1, n + 1):
        best, bd = -1, np.inf
        for u in range(1, n + 1):
            if not used[u] and C[cur, u] < bd:
                bd, best = C[cur, u], u
        used[best] = True
        seq[t] = best
        cur = best
    seq[n + 1] = 0
    improved = True
    while improved:
        improved = False
        for a in range(0, n):
            for b in range(a + 2, n + 1):
                d0 = C[seq[a], seq[a + 1]] + C[seq[b], seq[b + 1]]
                d1 = C[seq[a], seq[b]] + C[seq[a + 1], seq[b + 1]]
                if d1 < d0 - 1e-9:
                    lo, hi = a + 1, b
                    while lo < hi:
                        tmp = seq[lo]
                        seq[lo] = seq[hi]
                        seq[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
        if or_opt and not improved:
            for L in range(1, 4):
                for i in range(1, n + 2 - L):
                    j = i + L - 1
                    p, q = seq[i - 1], seq[j + 1]
                    rem = C[p, seq[i]] + C[seq[j], q] - C[p, q]
                    for k in range(0, n + 1):
                        if k >= i - 1 and k <= j:
                            continue
                        a, b = seq[k], seq[k + 1]
                        add = C[a, seq[i]] + C[seq[j], b] - C[a, b]
                        if add < rem - 1e-9:
                            seg = seq[i:j + 1].copy()
                            rest = np.concatenate((seq[:i], seq[j + 1:]))
                            pos = k + 1 if k < i else k + 1 - L
                            seq[:] = np.concatenate((rest[:pos], seg, rest[pos:]))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return _tour_len(C, seq), seq[1:n + 1].copy()


def held_karp(D: np.ndarray, nodes) -> tuple[float, list]:
    """Chu trình ngắn nhất qua nodes (nodes[0] = kho) – chính xác. Trả (độ dài, thứ tự các node ≠ kho)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    L, o = _held_karp(np.ascontiguousarray(D[np.ix_(nodes, nodes)], dtype=np.float64))
    return float(L), [int(nodes[i]) for i in o]


def tour_heuristic(D: np.ndarray, nodes, or_opt: bool = True) -> tuple[float, list]:
    """Gần nhất + 2-opt (+ Or-opt). Chỉ cho khoảng cách ĐỐI XỨNG (2-opt đảo chiều đoạn đường)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    C = np.ascontiguousarray(D[np.ix_(nodes, nodes)], dtype=np.float64)
    if not np.allclose(C, C.T):
        raise ValueError("tour_heuristic cần ma trận khoảng cách đối xứng")
    L, o = _heuristic(C, or_opt)
    return float(L), [int(nodes[i]) for i in o]


# ------------------------------------------------------------- mô hình nhặt đơn
class PickingModel:
    """Đơn online (mẫu từ cùng dữ liệu giỏ) trên mặt bằng của instance; điểm = fp.points, kho = khu tập kết."""

    def __init__(self, inst: Instance, n_orders: int = 300, seed: int = 0, max_exact: int = MAX_EXACT):
        self.inst, self.fp = inst, inst.fp
        fp = inst.fp
        src = [fp.node_of[p] for p in fp.points]
        self.D = np.ascontiguousarray(fp._dist[:, src], dtype=np.float64)       # điểm × điểm
        self.depot = fp.m + 1
        self.max_exact = max_exact
        B = inst.baskets
        rows = np.random.default_rng(seed + 7).choice(B.shape[0], size=min(n_orders, B.shape[0]), replace=False)
        self.orders = [B.indices[B.indptr[r]:B.indptr[r + 1]].astype(np.int64) for r in rows]
        ns = sum(s.face in "NS" for s in fp.slots)
        self.horizontal = ns >= fp.m - ns                    # dải lối chạy ngang (lưới) hay dọc (vòng)
        acc = np.array([s.access for s in fp.slots])
        self.band = acc[:, 0] if self.horizontal else acc[:, 1]
        self.pos = acc[:, 1] if self.horizontal else acc[:, 0]
        st = fp.staging
        self.band_dep, self.pos_dep = (st[0], st[1]) if self.horizontal else (st[1], st[0])
        self.band_lo = {b: self.pos[self.band == b].min() for b in np.unique(self.band)}
        self.band_hi = {b: self.pos[self.band == b].max() for b in np.unique(self.band)}

    def order_stops(self, perm) -> list[np.ndarray]:
        perm = np.asarray(perm, dtype=np.int64)
        slot_of = np.empty(self.inst.n, dtype=np.int64)
        real = perm < self.inst.n
        slot_of[perm[real]] = self.inst.slot_idx[np.where(real)[0]]
        return [np.unique(slot_of[o]) for o in self.orders]

    def _length(self, seq) -> float:
        s = [self.depot, *seq, self.depot]
        return float(sum(self.D[a, b] for a, b in zip(s, s[1:])))

    def route(self, stops, policy: str = "exact") -> tuple[float, list]:
        stops = [int(x) for x in stops]
        nodes = [self.depot, *stops]
        if policy == "exact":
            if len(stops) <= self.max_exact:
                return held_karp(self.D, nodes)
            return tour_heuristic(self.D, nodes, or_opt=True)
        if policy == "nn2opt":
            return tour_heuristic(self.D, nodes, or_opt=False)
        if policy in ("sshape", "largest_gap"):
            seq = self._aisle_policy(stops, policy)
            return self._length(seq), seq
        raise ValueError(policy)

    def _aisle_policy(self, stops, policy) -> list:
        bands = sorted({int(self.band[k]) for k in stops}, key=lambda b: (abs(b - self.band_dep), b))
        near_lo = abs(self.pos_dep - min(self.band_lo.values())) <= abs(self.pos_dep - max(self.band_hi.values()))
        by_band = {b: sorted((k for k in stops if self.band[k] == b), key=lambda k: self.pos[k]) for b in bands}
        if policy == "sshape":
            seq = []
            for i, b in enumerate(bands):
                ks = by_band[b] if (i % 2 == 0) == near_lo else by_band[b][::-1]
                seq += ks
            return seq
        out, back = [], []                                   # largest gap
        for b in bands:
            ks = by_band[b]
            p = [self.band_lo[b]] + [self.pos[k] for k in ks] + [self.band_hi[b]]
            g = int(np.argmax(np.diff(p)))                   # khoảng trống lớn nhất nằm giữa p[g] và p[g+1]
            lo_part, hi_part = ks[:g], ks[g:]
            near, far = (lo_part, hi_part[::-1]) if near_lo else (hi_part[::-1], lo_part)
            out += near
            back = far + back
        return out + back

    def lengths(self, perm, policy: str = "exact") -> np.ndarray:
        return np.array([self.route(s, policy)[0] for s in self.order_stops(perm)])

    def mean_length(self, perm, policy: str = "exact") -> float:
        return float(self.lengths(perm, policy).mean())

    def compare_policies(self, perm, policies=("exact", "nn2opt", "sshape", "largest_gap")) -> pd.DataFrame:
        stops = self.order_stops(perm)
        ex = np.array([self.route(s, "exact")[0] for s in stops])
        rows = []
        for pol in policies:
            L = ex if pol == "exact" else np.array([self.route(s, pol)[0] for s in stops])
            rows.append({"policy": pol, "mean_m": L.mean(), "gap_vs_exact": float(np.mean(L / np.maximum(ex, 1e-9) - 1)),
                         "share_exact_solved": float(np.mean([len(s) <= self.max_exact for s in stops]))})
        return pd.DataFrame(rows)


# ------------------------------------------------------------- gộp đơn
def batch_orders(pm: PickingModel, perm, B: int = 3, k_near: int = 8) -> dict:
    """Clarke–Wright: tiết kiệm s_ij = L_i + L_j − L(i ∪ j) trên k_near đơn gần nhất; ghép khi tổng cỡ ≤ B.
    Độ dài một xe = min(chu trình qua hợp các điểm dừng, các chu trình riêng nối tiếp) → không bao giờ tệ hơn."""
    stops = pm.order_stops(perm)
    N = len(stops)
    single = np.array([pm.route(s, "exact")[0] for s in stops])
    near_pairs = set()
    for i in range(N):
        d = np.array([pm.D[np.ix_(stops[i], stops[j])].min() if len(stops[i]) and len(stops[j]) else np.inf
                      for j in range(N)])
        d[i] = np.inf
        for j in np.argsort(d, kind="stable")[:k_near]:
            near_pairs.add((min(i, int(j)), max(i, int(j))))

    def tour(ids):
        u = np.unique(np.concatenate([stops[i] for i in ids]))
        return min(pm.route(u, "exact")[0], float(single[list(ids)].sum()))

    sav = sorted(((single[i] + single[j] - tour((i, j)), i, j) for i, j in near_pairs), reverse=True)
    batch_of = list(range(N))
    members = {i: [i] for i in range(N)}
    for s, i, j in sav:
        bi, bj = batch_of[i], batch_of[j]
        if s <= 0 or bi == bj or len(members[bi]) + len(members[bj]) > B:
            continue
        members[bi] += members.pop(bj)
        for o in members[bi]:
            batch_of[o] = bi
    batches = list(members.values())
    total = float(sum(tour(tuple(b)) for b in batches))
    st = float(single.sum())
    return {"batches": batches, "total": total, "single_total": st, "saving": 1 - total / st,
            "mean_per_order": total / N, "B": B}


# ------------------------------------------------------------- tránh khách
def node_density(fp, expo_fp: np.ndarray) -> np.ndarray:
    """Mật độ khách tương đối của mỗi ô lối đi: Σ lượt đi qua các slot có vùng tiếp xúc chứa ô / trung bình."""
    rho = np.zeros(fp.adj.shape[0])
    for s, slot in enumerate(fp.slots):
        for c in slot.zone:
            rho[fp.node_of[c]] += expo_fp[s]
    pos = rho[rho > 0]
    return rho / (pos.mean() if len(pos) else 1.0)


def avoidance_curve(pm: PickingModel, perm, customer_rm, kappas=(0.0, 0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    """Với mỗi κ: đường nhặt tối ưu theo chi phí (độ dài + κ·mật độ); báo quãng đường thật và chạm mặt
    (Σ mật độ khách tương đối trên các ô đi qua). Chạm mặt cùng thang với nhau giữa các κ."""
    fp = pm.fp
    customer_rm._run(np.asarray(perm, dtype=np.int64))
    rho = node_density(fp, customer_rm._expo)
    A = fp.adj.tocoo()
    src = np.array([fp.node_of[p] for p in fp.points])
    stops = pm.order_stops(perm)
    rows = []
    for kappa in kappas:
        # chi phí cạnh ĐỐI XỨNG: 1 + κ·(ρ_u + ρ_v)/2 → khoảng cách có trọng số là metric đối xứng
        w = 1.0 + kappa * 0.5 * (rho[A.row] + rho[A.col])
        W = sp.csr_matrix((w, (A.row, A.col)), shape=A.shape)
        dist, pred = dijkstra(W, directed=False, indices=src, return_predecessors=True)
        P = len(src)
        phys = np.zeros((P, P))
        enc = np.zeros((P, P))
        for a in range(P):                                   # cộng dồn dọc cây đường đi của nguồn a
            order = np.argsort(dist[a])
            pl = np.zeros(len(order))
            en = np.zeros(len(order))
            en[src[a]] = rho[src[a]]
            for v in order:
                u = pred[a, v]
                if u >= 0:
                    pl[v] = pl[u] + 1.0
                    en[v] = en[u] + rho[v]
            phys[a] = pl[src]
            enc[a] = en[src]
        DW = dist[:, src]
        d_tot = e_tot = 0.0
        for s in stops:
            nodes = [pm.depot, *[int(x) for x in s]]
            _, seq = held_karp(DW, nodes) if len(s) <= pm.max_exact else tour_heuristic(DW, nodes)
            q = [pm.depot, *seq, pm.depot]
            d_tot += sum(phys[a, b] for a, b in zip(q, q[1:]))
            e_tot += sum(enc[a, b] for a, b in zip(q, q[1:]))
        rows.append({"kappa": float(kappa), "distance": d_tot / len(stops), "encounters": e_tot / len(stops)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- lịch nhặt theo giờ
def wave_schedule(lam_p, lam_w, orders_per_day: float, capacity_per_hour: float, window_h: int = 3,
                  open_hours=(7, 22)) -> dict:
    """Bài vận tải: x[h, t] = số đơn đặt giờ h được nhặt giờ t. min Σ x[h,t]·λ_W(t)
    s.t. Σ_t x[h,t] = o_h, Σ_h x[h,t] ≤ K, chỉ t mở cửa trong [h, h+Δ] (đơn ngoài giờ: [mở, mở+Δ])."""
    lam_p, lam_w = np.asarray(lam_p, float), np.asarray(lam_w, float)
    o = orders_per_day * lam_p / lam_p.sum()
    t_open = np.arange(open_hours[0], open_hours[1])
    allowed = np.zeros((24, 24), dtype=bool)
    for h in range(24):
        ts = [t for t in t_open if h <= t <= h + window_h]
        if not ts:                                           # đặt ngoài giờ hoặc sát giờ đóng → sáng hôm sau
            ts = [t for t in t_open if t <= open_hours[0] + window_h]
        allowed[h, ts] = True
    hs, ts = np.nonzero(allowed)
    c = lam_w[ts]
    A_eq = sp.csr_matrix((np.ones(len(hs)), (hs, np.arange(len(hs)))), shape=(24, len(hs)))
    A_ub = sp.csr_matrix((np.ones(len(ts)), (ts, np.arange(len(ts)))), shape=(24, len(ts)))
    r = linprog(c, A_ub=A_ub, b_ub=np.full(24, capacity_per_hour), A_eq=A_eq, b_eq=o, bounds=(0, None),
                method="highs")
    if r.status != 0:
        raise ValueError(f"Lịch nhặt không khả thi với sức chứa {capacity_per_hour} đơn/giờ: {r.message}")
    X = np.zeros((24, 24))
    X[hs, ts] = r.x
    # đối chứng "nhặt ngay": mỗi giờ lấy đơn cũ nhất trước, trong sức chứa
    Xa = np.zeros((24, 24))
    left = o.copy()
    for t in range(24):
        cap = capacity_per_hour if t in t_open else 0.0
        for h in range(24):
            if cap <= 1e-12 or not allowed[h, t] or left[h] <= 1e-12:
                continue
            q = min(cap, left[h])
            Xa[h, t] += q
            left[h] -= q
            cap -= q
    cost_asap = float((Xa.sum(axis=0) * lam_w).sum() + left.sum() * lam_w.max())   # đơn trễ tính chi phí cao nhất
    cost = float(r.fun)
    return {"X": X, "allowed": allowed, "cost": cost, "cost_asap": cost_asap,
            "reduction": 1 - cost / cost_asap if cost_asap > 0 else 0.0,
            "picks_per_hour": X.sum(axis=0), "picks_per_hour_asap": Xa.sum(axis=0)}


# ------------------------------------------------------------- kiểm chứng xấp xỉ Z_P
def surrogate_check(pm: PickingModel, perms, picker_rm=None, surrogate=None) -> dict:
    """Tương quan hạng giữa quãng đường nhặt THẬT (TSP chính xác theo đơn) và (a) mô hình người nhặt của FlowBank
    (gần nhất + 2-opt, mẫu đơn khác), (b) bài thay thế QAP Z_P^lin (mục 5.4: chứng minh đại diện hợp lệ)."""
    exact = np.array([pm.mean_length(p, "exact") for p in perms])
    out = {"n_layouts": len(perms), "exact_mean": float(exact.mean())}
    if picker_rm is not None:
        z = np.array([picker_rm.evaluate(p)[0] for p in perms])
        out["spearman_picker_model"] = float(stats.spearmanr(z, exact).statistic)
        out["mean_ratio_picker_model"] = float(np.mean(z / exact))
    if surrogate is not None:
        z = np.array([surrogate.z1(p) for p in perms])
        out["spearman_surrogate_qap"] = float(stats.spearmanr(z, exact).statistic)
    return out
