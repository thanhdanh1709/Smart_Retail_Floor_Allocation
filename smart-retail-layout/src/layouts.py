"""Sinh mặt bằng mẫu cho thực nghiệm (mục B7.1): lưới song song (grid),
vòng (racetrack) và tự do (free-form). Mỗi hàm trả về danh sách chuỗi lưới.

Quy ước chung: tường X bao quanh, kệ tường ở cạnh trên/trái/phải, cửa vào E ở
góc dưới bên trái, dãy quầy thu ngân C ở hàng sát tường dưới.
"""
from __future__ import annotations

import numpy as np

from .floorplan import FloorPlan


def _blank(H: int, W: int) -> list[list[str]]:
    g = [["A"] * W for _ in range(H)]
    for r in range(H):
        g[r][0] = g[r][W - 1] = "X"
    for c in range(W):
        g[0][c] = g[H - 1][c] = "X"
    return g


def _perimeter_and_front(g: list[list[str]], n_checkouts: int, shelf_bottom: int) -> None:
    """Kệ tường (trên, trái, phải) tới hàng shelf_bottom; quầy thu ngân và cửa vào."""
    H, W = len(g), len(g[0])
    for c in range(1, W - 1):
        g[1][c] = "S"
    for r in range(1, shelf_bottom + 1):
        g[r][1] = "S"
        g[r][W - 2] = "S"
    # thu ngân: hàng H-2, xen kẽ C/A ở nửa bên phải
    start = W // 2
    for i in range(n_checkouts):
        c = start + 2 * i
        if c < W - 2:
            g[H - 2][c] = "C"
    g[H - 1][2] = "E"


def grid_layout(blocks_x: int, blocks_y: int, island_len: int,
                aisle_w: int = 2, cross_w: int = 2, n_checkouts: int = 4) -> list[str]:
    """Lưới song song: các đảo kệ ngang hai mặt (dày 1 ô) xếp thành blocks_y dãy,
    mỗi dãy blocks_x đoạn, ngăn cách bởi lối đi ngang (aisle_w) và lối cắt dọc (cross_w)."""
    W = 2 + 1 + aisle_w + blocks_x * (island_len + cross_w) - cross_w + aisle_w + 1
    H = 2 + aisle_w + blocks_y * (1 + aisle_w) + 3
    g = _blank(H, W)
    _perimeter_and_front(g, n_checkouts, shelf_bottom=H - 5)
    for by in range(blocks_y):
        r = 2 + aisle_w + by * (1 + aisle_w)
        for bx in range(blocks_x):
            c0 = 2 + aisle_w + bx * (island_len + cross_w)
            for c in range(c0, c0 + island_len):
                g[r][c] = "S"
    return ["".join(row) for row in g]


def racetrack_layout(n_islands: int, island_len: int, loop_w: int = 3,
                     inner_aisle: int = 2, n_checkouts: int = 4) -> list[str]:
    """Vòng (racetrack): lối chính rộng loop_w chạy vòng quanh khối giữa; khối giữa
    gồm các đảo kệ dọc dày 2 ô (mỗi cột một mặt), cách nhau bởi lối phụ."""
    inner_w = n_islands * 2 + (n_islands - 1) * inner_aisle
    W = 2 + 1 + loop_w + inner_w + loop_w + 1
    H = 2 + loop_w + island_len + loop_w + 2
    g = _blank(H, W)
    _perimeter_and_front(g, n_checkouts, shelf_bottom=H - 4)
    r0 = 2 + loop_w
    for i in range(n_islands):
        c0 = 2 + loop_w + i * (2 + inner_aisle)
        for r in range(r0, r0 + island_len):
            g[r][c0] = "S"
            g[r][c0 + 1] = "S"
    return ["".join(row) for row in g]


def free_layout(H: int, W: int, n_fixtures: int, seed: int = 0,
                n_checkouts: int = 4, max_tries: int = 4000) -> list[str]:
    """Tự do (free-flow): các khối kệ hình chữ nhật kích thước/hướng ngẫu nhiên đặt
    trên sàn, giữ khoảng cách tối thiểu 2 ô để mọi lối đi thông nhau."""
    rng = np.random.default_rng(seed)
    g = _blank(H, W)
    _perimeter_and_front(g, n_checkouts, shelf_bottom=H - 4)
    occ = np.zeros((H, W), dtype=bool)          # vùng đệm đã chiếm
    top, bottom, left, right = 4, H - 5, 4, W - 5
    placed, tries = 0, 0
    while placed < n_fixtures and tries < max_tries:
        tries += 1
        thick = int(rng.choice([1, 2]))
        length = int(rng.integers(4, 9))
        horiz = bool(rng.random() < 0.5)
        h, w = (thick, length) if horiz else (length, thick)
        r = int(rng.integers(top, bottom - h + 2))
        c = int(rng.integers(left, right - w + 2))
        if r + h - 1 > bottom or c + w - 1 > right:
            continue
        if occ[r - 2:r + h + 2, c - 2:c + w + 2].any():
            continue
        occ[r:r + h, c:c + w] = True
        for rr in range(r, r + h):
            for cc in range(c, c + w):
                g[rr][cc] = "S"
        placed += 1
    return ["".join(row) for row in g]


def assign_cold(grid: list[str], n_cold: int, seg_len: int = 2) -> list[str]:
    """Chuyển n_cold slot ở xa cửa vào nhất (ưu tiên kệ tường phía sau) thành khu lạnh R."""
    if n_cold <= 0:
        return grid
    fp = FloorPlan(grid, seg_len=seg_len)
    wall = sorted(fp.wall_slots(), key=lambda k: -fp.d_in[k])
    rest = sorted(set(range(fp.m)) - set(wall), key=lambda k: -fp.d_in[k])
    g = [list(r) for r in grid]
    for k in (wall + rest)[:n_cold]:     # ưu tiên kệ tường, thiếu thì dùng tủ đảo
        for r, c in fp.slots[k].cells:
            g[r][c] = "R"
    out = ["".join(r) for r in g]
    got = int(FloorPlan(out, seg_len=seg_len).is_cold.sum())
    if got < n_cold:
        raise ValueError(f"Chỉ tạo được {got}/{n_cold} slot lạnh – mặt bằng quá nhỏ")
    return out


def place_staging(grid: list[str], cell: tuple[int, int] | None = None) -> list[str]:
    """Đặt khu tập kết đơn online (ô P, v4). Mặc định: ô lối đi xa nhất bên phải ở hàng thu ngân
    (sát kho phía sau). Mỗi mặt bằng chỉ một ô P."""
    g = [list(r.replace("P", "A")) for r in grid]
    if cell is None:
        H = len(g)
        row = next(r for r in range(H - 2, 0, -1) if "C" in g[r])
        col = max(c for c, ch in enumerate(g[row]) if ch == "A")
        cell = (row, col)
    r, c = cell
    if g[r][c] != "A":
        raise ValueError(f"Ô khu tập kết {cell} phải là lối đi (A), đang là {g[r][c]!r}")
    g[r][c] = "P"
    return ["".join(x) for x in g]


# Cấu hình mặc định cho 3 kiểu × 3 quy mô (được kiểm tra số slot trong instance.py)
PRESETS = {
    ("grid", "small"): dict(fn="grid", blocks_x=2, blocks_y=1, island_len=6),
    ("grid", "medium"): dict(fn="grid", blocks_x=3, blocks_y=2, island_len=6),
    ("grid", "large"): dict(fn="grid", blocks_x=4, blocks_y=4, island_len=8),
    ("racetrack", "small"): dict(fn="racetrack", n_islands=2, island_len=6),
    ("racetrack", "medium"): dict(fn="racetrack", n_islands=3, island_len=10),
    ("racetrack", "large"): dict(fn="racetrack", n_islands=7, island_len=14),
    ("free", "small"): dict(fn="free", H=16, W=22, n_fixtures=3, seed=1),
    ("free", "medium"): dict(fn="free", H=22, W=30, n_fixtures=5, seed=2),
    ("free", "large"): dict(fn="free", H=34, W=46, n_fixtures=16, seed=3),
}


def build(kind: str, scale: str, **override) -> list[str]:
    cfg = {**PRESETS[(kind, scale)], **override}
    fn = cfg.pop("fn")
    return {"grid": grid_layout, "racetrack": racetrack_layout, "free": free_layout}[fn](**cfg)
