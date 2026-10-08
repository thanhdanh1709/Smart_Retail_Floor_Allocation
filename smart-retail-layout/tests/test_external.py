"""Dữ liệu ngoài (Open e-commerce, Tesco Grocery 1.0, Lyon Dense Crowd) – bỏ qua nếu chưa tải dữ liệu thô."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import external as X  # noqa: E402
from src.params import load_groups  # noqa: E402


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"chưa tải {p}")


def test_amazon_mapping_is_disjoint_and_known_groups():
    cats = [c for cs in X.AMAZON_TO_GROUP.values() for c in cs]
    assert len(cats) == len(set(cats))                                  # mỗi danh mục Amazon chỉ vào một nhóm
    assert set(X.AMAZON_TO_GROUP) <= set(load_groups().group_id)
    gids = [g for _, gs in X.TESCO_MERGE.values() for g in gs]
    assert len(gids) == len(set(gids))


def test_group_prices():
    _need(X.RAW / "open_ecommerce" / "amazon-purchases.csv")
    p = X.amazon_group_prices()
    ok = p.dropna(subset=["price_median"])
    assert len(p) == 40 and len(ok) >= 30
    assert (ok.n_purchases >= 500).all()
    assert (ok.price_q25 <= ok.price_median).all() and (ok.price_median <= ok.price_q75).all()
    assert ok.price_index.median() == pytest.approx(1.0)


def test_tesco_shares_comparable():
    _need(X.RAW / "tesco" / "year_borough_grocery.csv")
    df, s = X.tesco_check()
    assert df.tesco_share.sum() == pytest.approx(1.0) and df.instacart_share.sum() == pytest.approx(1.0)
    assert s["spearman"] > 0.6                                           # thứ hạng nhóm chuyển được sang London


def test_lyon_speed_decreases_with_density():
    _need(X.RAW / "lyon_crowd" / "extracted" / "Data_Madras" / "TopView_trajectories")
    tab, fd = X.lyon_fundamental_diagram()
    assert fd["spearman_speed_density"] < -0.3 and fd["gamma"] > 0
    assert np.all(np.diff(tab.speed.values[:5]) < 0)                    # giảm đơn điệu ở vùng có nhiều dữ liệu
    r = X.relative_speed(np.array([0.1, 1.0, 3.0, 5.4]), fd)
    assert r[0] > 0.99 and np.all(np.diff(r) < 0) and r[-1] == pytest.approx(0.0, abs=1e-9)
