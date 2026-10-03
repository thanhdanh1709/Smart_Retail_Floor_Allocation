"""Các phép tính lõi được biên dịch bằng numba (mục B5.3).

Biểu diễn: hoán vị perm độ dài m, perm[k] = nhóm hàng (đã đệm nhóm rỗng) ở slot k.
Hàm mục tiêu gộp dạng tổng quát:

    Z(perm) = cq * sum_{k<l} W[perm_k, perm_l] * D[k, l] + sum_k L[perm_k, k]

Khi hoán đổi hai slot r, s (a = perm_r, b = perm_s), với W, D đối xứng, W chéo = 0:

    Δ = cq * sum_{k≠r,s} (W[b, perm_k] - W[a, perm_k]) * (D[r, k] - D[s, k])
        + L[b, r] + L[a, s] - L[a, r] - L[b, s]                       (công thức 13)

tính trong O(m) thay vì O(m^2).
Ràng buộc mềm (hình phạt rho × số vi phạm): cặp tách xa và số nhóm phải dời vượt R.
Ràng buộc tương thích/cố định được giữ cứng qua ma trận allowed (không cho phép hoán đổi vi phạm).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def quad_value(perm, W, D):
    m = perm.shape[0]
    s = 0.0
    for k in range(m):
        a = perm[k]
        for l in range(k + 1, m):
            s += W[a, perm[l]] * D[k, l]
    return s


@njit(cache=True)
def lin_value(perm, L):
    s = 0.0
    for k in range(perm.shape[0]):
        s += L[perm[k], k]
    return s


@njit(cache=True)
def objective(perm, W, D, L, cq):
    return cq * quad_value(perm, W, D) + lin_value(perm, L)


@njit(cache=True)
def delta_swap(perm, W, D, L, cq, r, s):
    a = perm[r]
    b = perm[s]
    acc = 0.0
    for k in range(perm.shape[0]):
        if k == r or k == s:
            continue
        c = perm[k]
        acc += (W[b, c] - W[a, c]) * (D[r, k] - D[s, k])
    return cq * acc + L[b, r] + L[a, s] - L[a, r] - L[b, s]


# ------------------------------------------------------------- ràng buộc mềm
@njit(cache=True)
def inverse(perm):
    loc = np.empty_like(perm)
    for k in range(perm.shape[0]):
        loc[perm[k]] = k
    return loc


@njit(cache=True)
def soft_violations(perm, D, sep_i, sep_j, sep_d, k0, R):
    """(số cặp tách xa bị vi phạm, số nhóm phải dời vượt R)."""
    loc = inverse(perm)
    v_sep = 0
    for t in range(sep_i.shape[0]):
        if D[loc[sep_i[t]], loc[sep_j[t]]] < sep_d[t]:
            v_sep += 1
    excess = 0
    if R >= 0:
        moved = 0
        for i in range(k0.shape[0]):
            if k0[i] >= 0 and loc[i] != k0[i]:
                moved += 1
        if moved > R:
            excess = moved - R
    return v_sep, excess


@njit(cache=True)
def compat_violations(perm, allowed):
    v = 0
    for k in range(perm.shape[0]):
        if not allowed[perm[k], k]:
            v += 1
    return v


@njit(cache=True)
def fitness(perm, W, D, L, cq, allowed, sep_i, sep_j, sep_d, k0, R, rho):
    v_sep, excess = soft_violations(perm, D, sep_i, sep_j, sep_d, k0, R)
    return objective(perm, W, D, L, cq) + rho * (compat_violations(perm, allowed) + v_sep + excess)


@njit(cache=True)
def _soft_delta(perm, loc, D, sep_i, sep_j, sep_d, k0, R, moved, r, s):
    """Thay đổi (vi phạm tách xa + vượt R) khi hoán đổi slot r, s. Trả về (Δviol, Δmoved)."""
    a = perm[r]
    b = perm[s]
    dv = 0
    for t in range(sep_i.shape[0]):
        i = sep_i[t]
        j = sep_j[t]
        if i != a and i != b and j != a and j != b:
            continue
        li = loc[i]
        lj = loc[j]
        before = 1 if D[li, lj] < sep_d[t] else 0
        li2 = s if i == a else (r if i == b else li)
        lj2 = s if j == a else (r if j == b else lj)
        after = 1 if D[li2, lj2] < sep_d[t] else 0
        dv += after - before
    dm = 0
    if R >= 0:
        if a < k0.shape[0] and k0[a] >= 0:
            dm += (1 if s != k0[a] else 0) - (1 if r != k0[a] else 0)
        if b < k0.shape[0] and k0[b] >= 0:
            dm += (1 if r != k0[b] else 0) - (1 if s != k0[b] else 0)
        ex_before = moved - R if moved > R else 0
        ex_after = moved + dm - R if moved + dm > R else 0
        dv += ex_after - ex_before
    return dv, dm


@njit(cache=True)
def local_search(perm, W, D, L, cq, allowed, sep_i, sep_j, sep_d, k0, R, rho,
                 max_passes, first_improvement):
    """Tìm kiếm cục bộ 2-swap (cải thiện đầu tiên hoặc tốt nhất) dùng Δ(r, s).
    Sửa perm tại chỗ, trả về số nước đi đã thực hiện."""
    m = perm.shape[0]
    loc = inverse(perm)
    moved = 0
    if R >= 0:
        for i in range(k0.shape[0]):
            if k0[i] >= 0 and loc[i] != k0[i]:
                moved += 1
    n_moves = 0
    r0 = 0
    for _ in range(max_passes):
        improved = False
        best = -1e-12
        br = -1
        bs = -1
        bdm = 0
        for rr in range(m - 1):
            # cải thiện đầu tiên: quét vòng tiếp từ vị trí vừa cải thiện
            r = (r0 + rr) % (m - 1) if first_improvement else rr
            for s in range(r + 1, m):
                a = perm[r]
                b = perm[s]
                if a == b:
                    continue
                if not (allowed[a, s] and allowed[b, r]):
                    continue
                d = delta_swap(perm, W, D, L, cq, r, s)
                dv, dm = _soft_delta(perm, loc, D, sep_i, sep_j, sep_d, k0, R, moved, r, s)
                d += rho * dv
                if d < best:
                    best = d
                    br = r
                    bs = s
                    bdm = dm
                    if first_improvement:
                        break
            if first_improvement and br >= 0:
                break
        if br >= 0:
            a = perm[br]
            b = perm[bs]
            perm[br] = b
            perm[bs] = a
            loc[a] = bs
            loc[b] = br
            moved += bdm
            n_moves += 1
            improved = True
            r0 = br
        if not improved:
            break
    return n_moves


@njit(cache=True)
def swap_delta_full(perm, loc, W, D, L, cq, sep_i, sep_j, sep_d, k0, R, rho, moved, r, s):
    """Δ đầy đủ (mục tiêu + hình phạt) cho một hoán đổi – dùng trong SA/tabu."""
    d = delta_swap(perm, W, D, L, cq, r, s)
    dv, dm = _soft_delta(perm, loc, D, sep_i, sep_j, sep_d, k0, R, moved, r, s)
    return d + rho * dv, dm
