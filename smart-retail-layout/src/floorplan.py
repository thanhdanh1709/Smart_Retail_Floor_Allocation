"""Mô hình hóa mặt bằng cửa hàng (mục B2 của đề cương).

Lưới ô vuông 1 m, mỗi ô có một loại:
    A – lối đi, S – kệ, R – kệ khu lạnh, X – tường/vật cản, E – cửa vào, C – quầy thu ngân.

Từ lưới, mô-đun tách các vị trí trưng bày (slot), dựng đồ thị lối đi và tính:
    D[k, l]  – khoảng cách đi bộ giữa điểm tiếp cận slot k và l (m)
    d_in[k]  – khoảng cách cửa vào -> slot k
    d_out[k] – khoảng cách slot k -> quầy thu ngân gần nhất
    e[k]     – mức tiếp xúc hình học của slot k (phương án (c) mục B2.4), chuẩn hóa [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

WALKABLE = set("AECP")             # P – khu tập kết đơn online (v4), đi lại được
SHELF = set("SR")
DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}


@dataclass
class Slot:
    cells: list[tuple[int, int]]      # các ô kệ thuộc slot
    face: str                         # hướng mặt kệ quay ra lối đi (N/S/W/E)
    access: tuple[int, int]           # điểm tiếp cận: ô lối đi trước tâm slot
    zone: list[tuple[int, int]]       # các ô lối đi trước mặt slot (vùng tiếp xúc)
    is_cold: bool


@dataclass
class FloorPlan:
    grid: list[str]
    name: str = ""
    seg_len: int = 2
    min_len: int = 2
    slots: list[Slot] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.grid = [row.rstrip("\n") for row in self.grid if row.strip()]
        widths = {len(r) for r in self.grid}
        if len(widths) != 1:
            raise ValueError(f"Các dòng lưới phải dài bằng nhau, nhận {sorted(widths)}")
        self.R, self.C = len(self.grid), widths.pop()
        self.slots = extract_slots(self.grid, self.seg_len, self.min_len)
        if not self.slots:
            raise ValueError("Không tìm thấy slot nào trên mặt bằng")
        self._build_graph()

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "FloorPlan":
        path = Path(path)
        return cls(path.read_text(encoding="utf-8").splitlines(), name=path.stem, **kw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.grid) + "\n", encoding="utf-8")

    def slots_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "slot_id": [f"K{k + 1:03d}" for k in range(self.m)],
            "cells": [";".join(f"({r},{c})" for r, c in s.cells) for s in self.slots],
            "face": [s.face for s in self.slots],
            "access_r": [s.access[0] for s in self.slots],
            "access_c": [s.access[1] for s in self.slots],
            "is_cold": [int(s.is_cold) for s in self.slots],
            "d_in": self.d_in, "d_out": self.d_out, "e_geom": self.e,
        })

    # ------------------------------------------------------------ properties
    @property
    def m(self) -> int:
        return len(self.slots)

    @property
    def is_cold(self) -> np.ndarray:
        return np.array([s.is_cold for s in self.slots], dtype=bool)

    def cells_of(self, ch: str) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.R) for c in range(self.C) if self.grid[r][c] == ch]

    # ----------------------------------------------------------------- graph
    def _build_graph(self) -> None:
        walk = [(r, c) for r in range(self.R) for c in range(self.C) if self.grid[r][c] in WALKABLE]
        self.node_of = {cell: i for i, cell in enumerate(walk)}
        self.cell_of = walk
        rows, cols = [], []
        for (r, c), i in self.node_of.items():
            for dr, dc in DIRS.values():
                j = self.node_of.get((r + dr, c + dc))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        n = len(walk)
        self.adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

        entrances = self.cells_of("E")
        self.checkouts = self.cells_of("C")
        if not entrances or not self.checkouts:
            raise ValueError("Mặt bằng cần ít nhất một ô E (cửa vào) và một ô C (thu ngân)")
        self.entrance = entrances[0]
        staging = self.cells_of("P")
        self.has_staging = bool(staging)
        self.staging = staging[0] if staging else self.entrance

        # điểm: m điểm tiếp cận + cửa vào (chỉ số m) + khu tập kết (chỉ số m + 1)
        self.points = [s.access for s in self.slots] + [self.entrance, self.staging]
        src = [self.node_of[p] for p in self.points]
        dist, pred = shortest_path(self.adj, method="D", unweighted=True,
                                   indices=src, return_predecessors=True)
        if not np.isfinite(dist[:, src]).all():
            raise ValueError("Có điểm tiếp cận không đi tới được từ cửa vào")
        self._dist, self._pred = dist, pred
        co_nodes = np.array([self.node_of[c] for c in self.checkouts])
        dco = dist[:, co_nodes]
        self._co_target = co_nodes[dco.argmin(axis=1)]   # quầy gần nhất của từng điểm
        P = dist[:, src]
        m = self.m
        self.D = P[:m, :m].copy()
        self.d_in = P[m, :m].copy()
        self.d_out = dco.min(axis=1)[:m].copy()
        self.d_0 = P[m + 1, :m].copy()                    # khu tập kết -> slot k (nhặt đơn)
        self.d_entry_exit = float(dco.min(axis=1)[m])

        # vùng tiếp xúc: ô lối đi -> danh sách slot
        self.zone_index: dict[int, list[int]] = {}
        for k, s in enumerate(self.slots):
            for cell in s.zone:
                self.zone_index.setdefault(self.node_of[cell], []).append(k)
        self._leg_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.e = geometric_exposure(self)

    # ------------------------------------------------------- path utilities
    def path_nodes(self, a: int, b: int) -> np.ndarray:
        """Đường đi ngắn nhất giữa điểm a và b (chỉ số trong self.points; b = -1 là thu ngân)."""
        target = self._co_target[a] if b == -1 else self.node_of[self.points[b]]
        pred = self._pred[a]
        out = [target]
        while out[-1] != self.node_of[self.points[a]]:
            out.append(pred[out[-1]])
        return np.array(out[::-1])

    def leg(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """(các nút trên đường a->b, các slot có vùng tiếp xúc nằm trên đường) – có cache."""
        key = (a, b)
        hit = self._leg_cache.get(key)
        if hit is None:
            nodes = self.path_nodes(a, b)
            exp = sorted({k for v in nodes for k in self.zone_index.get(int(v), ())})
            hit = (nodes, np.array(exp, dtype=np.int64))
            self._leg_cache[key] = hit
        return hit

    def leg_length(self, a: int, b: int) -> float:
        if b == -1:
            return float(self._dist[a, self._co_target[a]])
        return float(self._dist[a, self.node_of[self.points[b]]])

    def snake_order(self) -> np.ndarray:
        """Thứ tự slot dạng 'đường rắn': láng giềng gần nhất bắt đầu từ cửa vào."""
        m = self.m
        left = set(range(m))
        cur_d = self.d_in.copy()
        order = []
        while left:
            idx = np.array(sorted(left))
            k = int(idx[np.argmin(cur_d[idx])])
            order.append(k)
            left.remove(k)
            cur_d = self.D[k]
        return np.array(order)

    def wall_slots(self) -> list[int]:
        """Slot nằm sát tường (kệ tường) – ứng viên khu lạnh."""
        out = []
        for k, s in enumerate(self.slots):
            r, c = s.cells[0]
            dr, dc = DIRS[s.face]
            if 0 <= r - dr < self.R and 0 <= c - dc < self.C and self.grid[r - dr][c - dc] == "X":
                out.append(k)
        return out


# ----------------------------------------------------------------------------
def extract_slots(grid: list[str], seg_len: int = 2, min_len: int = 2) -> list[Slot]:
    """Tách slot: mỗi đoạn kệ liên tục quay ra lối đi theo một hướng được cắt thành
    các đoạn dài seg_len ô; phần dư ngắn hơn min_len được nhập vào đoạn trước.
    Kệ đảo hai mặt sinh hai slot (mỗi mặt một slot)."""
    R, C = len(grid), len(grid[0])

    def walk(r, c):
        return 0 <= r < R and 0 <= c < C and grid[r][c] in WALKABLE

    slots: list[Slot] = []
    for face, (dr, dc) in DIRS.items():
        horizontal = face in "NS"
        outer = range(R) if horizontal else range(C)
        for a in outer:
            inner = range(C) if horizontal else range(R)
            run: list[tuple[int, int]] = []
            run_type = None

            def flush():
                if len(run) >= min_len:
                    slots.extend(_cut(run, face, run_type == "R", seg_len, min_len))

            for b in inner:
                r, c = (a, b) if horizontal else (b, a)
                ch = grid[r][c]
                if ch in SHELF and walk(r + dr, c + dc):
                    if run and ch != run_type:
                        flush()
                        run = []
                    run.append((r, c))
                    run_type = ch
                else:
                    flush()
                    run = []
                    run_type = None
            flush()
    # thứ tự ổn định: theo hàng rồi cột của ô đầu tiên, rồi hướng
    slots.sort(key=lambda s: (s.cells[0], s.face))
    return slots


def _cut(run, face, cold, seg_len, min_len) -> list[Slot]:
    dr, dc = DIRS[face]
    pieces = [run[i:i + seg_len] for i in range(0, len(run), seg_len)]
    if len(pieces) > 1 and len(pieces[-1]) < min_len:
        last = pieces.pop()
        pieces[-1] = pieces[-1] + last
    out = []
    for cells in pieces:
        mid = cells[(len(cells) - 1) // 2]
        out.append(Slot(cells=list(cells), face=face,
                        access=(mid[0] + dr, mid[1] + dc),
                        zone=[(r + dr, c + dc) for r, c in cells],
                        is_cold=cold))
    return out


def geometric_exposure(fp: FloorPlan) -> np.ndarray:
    """Phương án (c) mục B2.4: đếm số lộ trình 'cửa vào -> slot l -> thu ngân'
    (với mọi l) đi qua vùng tiếp xúc của slot k. Chuẩn hóa min–max về [0, 1]."""
    m = fp.m
    cnt = np.zeros(m)
    for l in range(m):
        seen = set(fp.leg(m, l)[1].tolist()) | set(fp.leg(l, -1)[1].tolist())
        cnt[list(seen)] += 1
    return minmax(cnt)


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)
