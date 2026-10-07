"""Dữ liệu cấp món (v4, tầng 3) từ Instacart.

    items.csv – một dòng/món: product_id, product_name, aisle_id, department_id, group_id,
                n_orders (số đơn chứa món), freq = n_orders / N, reorder_rate.
Chỉ giữ món thuộc aisle đã ánh xạ vào nhóm hàng (bỏ "other", "missing").
Lọc phần đuôi (min_orders) làm ở bước dùng (`load`), không phải lúc dựng, để báo được % doanh số giữ lại.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .params import PROCESSED, RAW, _sha, aisle_table

ITEMS_DIR = PROCESSED / "items"


def build(raw: Path = RAW, out: Path = ITEMS_DIR) -> pd.DataFrame:
    t0 = time.time()
    prod = pd.read_csv(raw / "products.csv")
    op = pd.read_csv(raw / "order_products__prior.csv", usecols=["order_id", "product_id", "reordered"],
                     dtype={"order_id": np.int32, "product_id": np.int32, "reordered": np.int8})
    N = int(op.order_id.nunique())
    g = op.groupby("product_id").reordered.agg(["size", "mean"])
    atab = aisle_table(raw)[["aisle_id", "group_id"]]
    t = (prod.merge(atab, on="aisle_id")
             .merge(g.rename(columns={"size": "n_orders", "mean": "reorder_rate"}),
                    left_on="product_id", right_index=True))
    t["freq"] = t.n_orders / N
    t = t.sort_values(["group_id", "aisle_id", "n_orders"], ascending=[True, True, False])
    cols = ["product_id", "product_name", "aisle_id", "department_id", "group_id",
            "n_orders", "freq", "reorder_rate"]
    out.mkdir(parents=True, exist_ok=True)
    t[cols].to_csv(out / "items.csv", index=False, encoding="utf-8")
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_orders": N, "n_items": int(len(t)),
           "outputs": {"items.csv": _sha(out / "items.csv")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  {len(t):,} món, N = {N:,} đơn ({time.time() - t0:.1f}s)")
    return t[cols]


def load(min_orders: int = 100, root: Path = ITEMS_DIR) -> tuple[pd.DataFrame, float]:
    """(bảng món có ≥ min_orders lượt mua, tỷ lệ lượt mua giữ lại)."""
    p = root / "items.csv"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p}; chạy: python -m experiments.build_data --items")
    t = pd.read_csv(p, encoding="utf-8")
    keep = t[t.n_orders >= min_orders].reset_index(drop=True)
    return keep, float(keep.n_orders.sum() / t.n_orders.sum())
