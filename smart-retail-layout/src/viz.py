"""Vẽ hình (mục B7.3): mặt bằng có nhãn nhóm hàng, bản đồ nhiệt, tập Pareto, boxplot,
đường hội tụ, đường cong cải thiện theo R."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CELL_COLORS = {"X": "#3b3b3b", "A": "#ffffff", "S": "#d9d4c7", "R": "#bfe0f2", "E": "#59a14f",
               "C": "#f28e2b"}


def plot_floorplan(fp, ax=None, inst=None, perm=None, title: str = "", label: str = "id",
                   highlight_moved: bool = False, fontsize: float = 6.0):
    """Vẽ lưới mặt bằng; nếu có (inst, perm) thì ghi nhãn nhóm hàng trên từng slot."""
    ax = ax or plt.gca()
    for r in range(fp.R):
        for c in range(fp.C):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=CELL_COLORS.get(fp.grid[r][c], "#fff"),
                                   edgecolor="#eeeeee", linewidth=0.3))
    if inst is not None and perm is not None:
        pos = {int(s): k for k, s in enumerate(inst.slot_idx)}
        for s, slot in enumerate(fp.slots):
            k = pos.get(s)
            if k is None:
                continue
            i = int(perm[k])
            if i >= inst.n:
                continue
            cells = np.array(slot.cells)
            cy, cx = cells[:, 0].mean() + 0.5, cells[:, 1].mean() + 0.5
            ar, ac = slot.access
            ty, tx = (cy + (ar + 0.5)) / 2, (cx + (ac + 0.5)) / 2
            txt = inst.meta.category_id.iat[i] if label == "id" else str(inst.names[i])[:10]
            moved = highlight_moved and inst.k0[i] >= 0 and inst.k0[i] != k
            ax.text(tx, ty, txt, ha="center", va="center", fontsize=fontsize,
                    color="#c0392b" if moved else "#1f1f1f", fontweight="bold" if moved else None)
    ax.set_xlim(0, fp.C)
    ax.set_ylim(fp.R, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_heatmap(fp, values: np.ndarray, ax=None, title: str = "", vmax: float | None = None,
                 cmap: str = "YlOrRd", colorbar: bool = True):
    """Bản đồ nhiệt trên các ô đi được (values: mảng R×C, NaN ở ô không đi được)."""
    ax = ax or plt.gca()
    base = np.array([[0 if ch == "X" else (1 if ch in "SR" else 2) for ch in row] for row in fp.grid])
    ax.imshow(base, cmap=ListedColormap(["#3b3b3b", "#d9d4c7", "#ffffff"]), vmin=0, vmax=2)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return im


def plot_pareto(fronts: dict, ax=None, knees: dict | None = None, normalized: bool = True,
                title: str = "Tập Pareto"):
    """fronts: {tên: mảng (k, 2)} – F chuẩn hóa (cực tiểu cả hai) hoặc (Z1, Z2)."""
    ax = ax or plt.gca()
    markers = ["o", "s", "^", "D", "v", "P"]
    for t, (name, F) in enumerate(fronts.items()):
        F = np.asarray(F)
        if len(F) == 0:
            continue
        o = np.argsort(F[:, 0])
        ax.plot(F[o, 0], F[o, 1], marker=markers[t % len(markers)], ms=4, lw=1, label=f"{name} ({len(F)})")
        if knees and name in knees:
            k = knees[name]
            ax.scatter([F[k, 0]], [F[k, 1]], s=120, facecolors="none", edgecolors="k", zorder=5)
    if normalized:
        ax.set_xlabel("F1 – quãng đường (chuẩn hóa, thấp = tốt)")
        ax.set_ylabel("F2 – giá trị bỏ lỡ (chuẩn hóa, thấp = tốt)")
    else:
        ax.set_xlabel("Z1 – quãng đường kỳ vọng")
        ax.set_ylabel("Z2 – giá trị kỳ vọng")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)
    return ax


def boxplot(df, value: str, by: str, ax=None, title: str = ""):
    ax = ax or plt.gca()
    groups = list(dict.fromkeys(df[by]))
    ax.boxplot([df.loc[df[by] == g, value].values for g in groups], labels=groups, showmeans=True)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=7)
    return ax


def convergence(histories: dict, ax=None, title: str = "Đường hội tụ", x: str = "time"):
    """histories: {tên: list[(thời gian, giá trị)]} hoặc list[(thế hệ, thời gian, giá trị)]."""
    ax = ax or plt.gca()
    for name, h in histories.items():
        h = np.asarray(h)
        xs = h[:, 1] if h.shape[1] == 3 and x == "time" else h[:, 0]
        ax.plot(xs, h[:, -1], lw=1.2, label=name)
    ax.set_xlabel("thời gian (s)" if x == "time" else "thế hệ")
    ax.set_ylabel("Z tốt nhất")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)
    return ax


def save(fig, path, dpi: int = 160):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
