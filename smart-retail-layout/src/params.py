"""Ước lượng tham số từ dữ liệu giỏ hàng (mục B3 của đề cương).

Đầu vào: bộ Instacart 2017 (data/raw/instacart) và bảng ánh xạ aisle -> nhóm hàng
(data/mappings/groups_vn.csv). Đầu ra cho mỗi mức nhóm ("group" – 40 nhóm kiểu cửa
hàng Việt Nam, "aisle" – 132 aisle của Instacart):

    w.npy     w_ij = n_ij / N         (luồng – tỷ lệ đơn chứa cả i và j), công thức (1)
    f.npy     f_i  = n_i / N          (độ phổ biến)
    lift.npy  lift_ij = w_ij / (f_i f_j)
    p.npy     p_i  = 1 - tỷ lệ mua lại (proxy (a) của hệ số mua ngẫu hứng, B3.4)
    v.npy     giá trị nhóm hàng (giả định có cơ sở – Instacart không có giá)
    categories.csv   theo mẫu PL1
    baskets.npz      mẫu đơn hàng (CSR) cho mô phỏng tác tử
    manifest.json    phiên bản dữ liệu + mã băm để tái lập
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "instacart"
PROCESSED = ROOT / "data" / "processed"
MAPPING = ROOT / "data" / "mappings" / "groups_vn.csv"

# trung vị thời gian dừng (giây) theo 3 mức "cân nhắc" – được hiệu chỉnh ở B3.5
DWELL_MEDIAN_S = {1: 6.0, 2: 10.0, 3: 18.0}
EXCLUDED_AISLES = {6, 100}          # "other", "missing"


def load_groups(path: Path = MAPPING) -> pd.DataFrame:
    g = pd.read_csv(path, encoding="utf-8")
    g["aisle_ids"] = g["aisle_ids"].astype(str).str.split(";").apply(lambda xs: [int(x) for x in xs])
    return g


def aisle_table(raw: Path = RAW, groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bảng aisle (đã bỏ other/missing) gắn với nhóm hàng và thuộc tính kế thừa."""
    groups = load_groups() if groups is None else groups
    aisles = pd.read_csv(raw / "aisles.csv")
    rows = []
    for _, g in groups.iterrows():
        for a in g.aisle_ids:
            rows.append(dict(aisle_id=a, group_id=g.group_id, group_name=g["name"],
                             needs_cold=g.needs_cold, value_v=g.value_v, dwell_level=g.dwell_level))
    t = pd.DataFrame(rows).merge(aisles, on="aisle_id")
    missing = set(aisles.aisle_id) - set(t.aisle_id) - EXCLUDED_AISLES
    if missing:
        raise ValueError(f"Aisle chưa được ánh xạ: {sorted(missing)}")
    return t.sort_values("aisle_id").reset_index(drop=True)


def load_order_aisles(raw: Path = RAW, max_orders: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Đọc order_products__prior, gắn aisle; trả về các cặp (order_id, aisle_id, reordered)."""
    t0 = time.time()
    prod = pd.read_csv(raw / "products.csv", usecols=["product_id", "aisle_id"])
    p2a = np.zeros(prod.product_id.max() + 1, dtype=np.int16)
    p2a[prod.product_id.values] = prod.aisle_id.values
    op = pd.read_csv(raw / "order_products__prior.csv", usecols=["order_id", "product_id", "reordered"],
                     dtype={"order_id": np.int32, "product_id": np.int32, "reordered": np.int8},
                     engine="c")
    if max_orders:
        ids = op.order_id.unique()
        keep = np.random.default_rng(seed).choice(ids, size=min(max_orders, len(ids)), replace=False)
        op = op[op.order_id.isin(keep)]
    op["aisle_id"] = p2a[op.product_id.values]
    op = op[~op.aisle_id.isin(list(EXCLUDED_AISLES))]
    print(f"  đọc {len(op):,} dòng đơn–sản phẩm trong {time.time() - t0:.1f}s")
    return op[["order_id", "aisle_id", "reordered"]]


def cooccurrence(order_ids: np.ndarray, unit_codes: np.ndarray, n_units: int):
    """Ma trận đơn × nhóm (0/1) -> (X, n_ij); đường chéo n_ij là n_i."""
    r = pd.factorize(order_ids)[0]
    X = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, unit_codes)),
                      shape=(r.max() + 1, n_units))
    X.sum_duplicates()
    X.data[:] = 1.0                                      # bỏ trùng trong cùng đơn
    co = (X.T @ X).toarray().astype(np.float64)
    return X, co


def compute(op: pd.DataFrame, level: str, atab: pd.DataFrame, groups: pd.DataFrame,
            n_baskets: int = 200_000, seed: int = 0) -> dict:
    if level == "aisle":
        units = atab.aisle_id.values
        code_of = {a: i for i, a in enumerate(units)}
        codes = op.aisle_id.map(code_of).values
        meta = pd.DataFrame({
            "category_id": [f"A{a:03d}" for a in units],
            "name": atab.aisle.values, "group_id": atab.group_id.values,
            "needs_cold": atab.needs_cold.values, "value_v": atab.value_v.values,
            "dwell_level": atab.dwell_level.values,
        })
    elif level == "group":
        a2g = dict(zip(atab.aisle_id, atab.group_id))
        units = groups.group_id.values
        code_of = {g: i for i, g in enumerate(units)}
        codes = op.aisle_id.map(a2g).map(code_of).values
        meta = pd.DataFrame({
            "category_id": units, "name": groups["name"].values, "group_id": units,
            "needs_cold": groups.needs_cold.values, "value_v": groups.value_v.values,
            "dwell_level": groups.dwell_level.values,
        })
    else:
        raise ValueError(level)
    n = len(units)
    codes = codes.astype(np.int64)
    X, co = cooccurrence(op.order_id.values, codes, n)
    N = X.shape[0]
    f = np.diag(co) / N
    w = co / N
    np.fill_diagonal(w, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(np.outer(f, f) > 0, w / np.outer(f, f), 0.0)
    rr = pd.Series(op.reordered.values).groupby(codes).mean().reindex(range(n)).fillna(0).values
    p = 1.0 - rr
    meta["n_slots"] = 1
    meta["impulse_p"] = p.round(4)
    meta["dwell_median_s"] = meta.dwell_level.map(DWELL_MEDIAN_S)
    meta["popularity_f"] = f.round(6)
    meta["reorder_rate"] = rr.round(4)

    # mẫu giỏ hàng cho mô phỏng
    rng = np.random.default_rng(seed)
    pick = rng.choice(N, size=min(n_baskets, N), replace=False)
    B = X[pick].tocsr()
    return dict(w=w, f=f, lift=lift, p=p, v=meta.value_v.values.astype(float),
                meta=meta, baskets=B, N=N)


def save(res: dict, out: Path, source_files: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k in ("w", "f", "lift", "p", "v"):
        np.save(out / f"{k}.npy", res[k])
    cols = ["category_id", "name", "group_id", "n_slots", "needs_cold", "value_v", "impulse_p",
            "dwell_median_s", "popularity_f", "reorder_rate"]
    res["meta"][cols].to_csv(out / "categories.csv", index=False, encoding="utf-8")
    B = res["baskets"]
    np.savez_compressed(out / "baskets.npz", indptr=B.indptr, indices=B.indices, shape=B.shape)
    man = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_orders": int(res["N"]), "n_units": int(len(res["f"])),
        "sources": {p.name: {"bytes": p.stat().st_size} for p in source_files},
        "outputs": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.suffix in (".npy", ".csv", ".npz")},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def build_all(raw: Path = RAW, out: Path = PROCESSED, max_orders: int | None = None) -> None:
    groups = load_groups()
    atab = aisle_table(raw, groups)
    op = load_order_aisles(raw, max_orders=max_orders)
    srcs = [raw / "products.csv", raw / "order_products__prior.csv", MAPPING]
    for level in ("group", "aisle"):
        t0 = time.time()
        res = compute(op, level, atab, groups)
        save(res, out / level, srcs)
        print(f"  mức {level}: {len(res['f'])} nhóm, N = {res['N']:,} đơn ({time.time() - t0:.1f}s)")


# ------------------------------------------------------------------ loading
def load(level: str = "group", root: Path = PROCESSED) -> dict:
    d = root / level
    if not (d / "w.npy").exists():
        raise FileNotFoundError(f"Chưa có tham số {d}; chạy: python -m experiments.build_data")
    meta = pd.read_csv(d / "categories.csv", encoding="utf-8")
    z = np.load(d / "baskets.npz")
    B = sp.csr_matrix((np.ones(len(z["indices"])), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return dict(w=np.load(d / "w.npy"), f=np.load(d / "f.npy"), lift=np.load(d / "lift.npy"),
                p=np.load(d / "p.npy"), v=np.load(d / "v.npy"), meta=meta, baskets=B)


def top_pairs(par: dict, k: int = 15, by: str = "w") -> pd.DataFrame:
    """Các cặp nhóm hàng liên kết mạnh nhất (theo w hoặc lift)."""
    M = par[by]
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    order = np.argsort(-M[iu])[:k]
    names = par["meta"]["name"].values
    return pd.DataFrame({
        "nhóm 1": names[iu[0][order]], "nhóm 2": names[iu[1][order]],
        "w (support)": par["w"][iu][order].round(4), "lift": par["lift"][iu][order].round(2),
    })


def from_uploaded(categories: pd.DataFrame, baskets: pd.DataFrame) -> dict:
    """Tham số từ dữ liệu người dùng tải lên (dashboard, dữ liệu POS của cửa hàng):
    categories theo mẫu PL1 (category_id, name, needs_cold, value_v, impulse_p, dwell_median_s);
    baskets dạng dài (order_id, category_id)."""
    meta = categories.copy()
    if "group_id" not in meta:
        meta["group_id"] = meta["category_id"]
    for col, default in (("needs_cold", 0), ("value_v", 0.5), ("impulse_p", 0.3), ("dwell_median_s", 10.0),
                         ("n_slots", 1)):
        if col not in meta:
            meta[col] = default
    code_of = {c: i for i, c in enumerate(meta.category_id)}
    b = baskets[baskets.category_id.isin(code_of)]
    codes = b.category_id.map(code_of).values.astype(np.int64)
    X, co = cooccurrence(b.order_id.values, codes, len(meta))
    N = X.shape[0]
    f = np.diag(co) / N
    w = co / N
    np.fill_diagonal(w, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(np.outer(f, f) > 0, w / np.outer(f, f), 0.0)
    meta["popularity_f"] = f
    return dict(w=w, f=f, lift=lift, p=meta.impulse_p.values.astype(float),
                v=meta.value_v.values.astype(float), meta=meta.reset_index(drop=True), baskets=X.tocsr())
