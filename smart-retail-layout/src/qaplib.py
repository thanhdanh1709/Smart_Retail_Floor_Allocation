"""Đọc bộ dữ liệu QAPLIB (Burkard và cs., 1997) để kiểm chứng cài đặt GA (thí nghiệm E1).

Định dạng tệp: n, ma trận A (n×n), ma trận B (n×n). Giá trị QAP của hoán vị p (cơ sở -> vị trí):
    sum_{i,j} a_ij b_{p(i) p(j)}.
Với biểu diễn của dự án (perm[k] = cơ sở ở vị trí k) và các nhân Δ yêu cầu ma trận đối xứng:
    W = A + Aᵀ (chéo 0), D = (B + Bᵀ)/2, L[i, k] = a_ii b_kk
cho đúng giá trị QAP khi ít nhất một trong A, B đối xứng.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QAPLIB_DIR = ROOT / "data" / "qaplib"

# lời giải tối ưu đã công bố (QAPLIB)
OPTIMA = {"nug12": 578, "chr12a": 9552, "had12": 1652, "tai12a": 224416, "esc16a": 68,
          "had20": 6922, "nug20": 2570, "tai20a": 703482, "kra30a": 88900, "nug30": 6124}


def read(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    nums = np.array(Path(path).read_text().split(), dtype=float)
    n = int(nums[0])
    A = nums[1:1 + n * n].reshape(n, n)
    B = nums[1 + n * n:1 + 2 * n * n].reshape(n, n)
    return A, B


class QAPInstance:
    """Instance tối giản tương thích với ga.run / baselines (không ràng buộc nghiệp vụ)."""

    def __init__(self, name: str, A: np.ndarray, B: np.ndarray):
        if not (np.allclose(A, A.T) or np.allclose(B, B.T)):
            raise ValueError(f"{name}: cả hai ma trận đều bất đối xứng – không hỗ trợ")
        self.name = name
        m = len(A)
        W = A + A.T
        np.fill_diagonal(W, 0.0)
        self.W = np.ascontiguousarray(W)
        self.D = np.ascontiguousarray((B + B.T) / 2)
        self.L = np.ascontiguousarray(np.outer(np.diag(A), np.diag(B)))
        self.allowed = np.ones((m, m), dtype=np.bool_)
        self.sep_i = np.zeros(0, dtype=np.int64)
        self.sep_j = np.zeros(0, dtype=np.int64)
        self.sep_d = np.zeros(0)
        self.k0 = -np.ones(m, dtype=np.int64)
        self.R = -1
        self.f = self.W.sum(axis=1)
        self.payoff = {"qap": True}
        self._m = m

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._m

    def weighted(self, alpha: float = 1.0):
        return self.L, 1.0, 0.0

    def kernel_args(self, alpha: float = 1.0, rho: float = 10.0):
        return (self.W, self.D, self.L, 1.0, self.allowed, self.sep_i, self.sep_j, self.sep_d,
                self.k0, -1, float(rho)), 0.0

    def value(self, perm: np.ndarray) -> float:
        from . import fastops
        return float(fastops.objective(np.asarray(perm, dtype=np.int64), self.W, self.D, self.L, 1.0))


def load(name: str) -> QAPInstance:
    A, B = read(QAPLIB_DIR / f"{name}.dat")
    return QAPInstance(name, A, B)


def value_direct(A: np.ndarray, B: np.ndarray, p_fac_to_loc: np.ndarray) -> float:
    """Giá trị QAP tính trực tiếp theo định nghĩa (dùng trong kiểm thử)."""
    p = np.asarray(p_fac_to_loc)
    return float((A * B[np.ix_(p, p)]).sum())
