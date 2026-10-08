"""Dữ liệu ngoài bổ trợ cho Instacart (kế hoạch v4, GĐ0 mở rộng). Mỗi bộ chỉ dùng cho đúng chỗ nó có thông tin:

    Open e-commerce 1.0 (Berke et al. 2024, CC0)  – giá thật theo danh mục Amazon → chỉ số giá TƯƠNG ĐỐI của
        40 nhóm (tầng 3: giá/lãi m_j; độ nhạy v_i). Giá Amazon là giá mua online (gói lớn, USD) nên chỉ dùng thứ hạng
        / tỷ lệ giữa các nhóm, không dùng mức tuyệt đối.
    Tesco Grocery 1.0 (Aiello et al. 2020, CC BY 4.0) – tỷ lệ món mua theo nhóm thực phẩm ở 411 cửa hàng London →
        kiểm tra tính khái quát của độ phổ biến f_i lấy từ Instacart (Mỹ) trên cửa hàng vật lý châu Âu.
    Lyon Dense Crowd (MADRAS, Fête des Lumières 2022, CC BY 4.0) – quỹ đạo người đi bộ đông đúc → giản đồ cơ bản
        tốc độ–mật độ v(ρ) (Weidmann) cho mô hình ùn tắc (SUE tầng 4, mô phỏng hai luồng GĐ6). Đám đông lễ hội
        khác khách siêu thị → chỉ dùng HÌNH DẠNG suy giảm tương đối v(ρ)/v0, không dùng v0.
Không dùng: RPC (ảnh nhận dạng sản phẩm – không có thông tin bố trí/luồng), "HRN4Customer" (không tồn tại).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from .params import PROCESSED, ROOT, load_groups

RAW = ROOT / "data" / "raw"
OUT = PROCESSED / "external"

# danh mục Amazon -> nhóm hàng (chỉ ánh xạ khi nghĩa rõ ràng; nhóm không có danh mục tương ứng để trống)
AMAZON_TO_GROUP = {
    "G01": ["VEGETABLE"], "G02": ["FRUIT"], "G03": ["MEAT"], "G04": ["FISH", "SHELLFISH"],
    "G05": ["JERKY"], "G06": ["DAIRY_BASED_DRINK", "MILK_SUBSTITUTE"], "G07": ["DAIRY_BASED_YOGURT"],
    "G08": ["DAIRY_BASED_CHEESE", "DAIRY_BASED_BUTTER", "DAIRY_BASED_CREAM", "NON_DAIRY_CREAM"],
    "G09": ["EGG"], "G13": ["DAIRY_BASED_ICE_CREAM"], "G15": ["BREAD"], "G16": ["CAKE", "PASTRY"],
    "G17": ["BREAKFAST_CEREAL", "CEREAL"], "G18": ["RICE_MIX"], "G19": ["NOODLE", "SAUCE"],
    "G20": ["PACKAGED_SOUP_AND_STEW"], "G21": ["EDIBLE_OIL_VEGETABLE", "VINEGAR"],
    "G22": ["HERB", "SEASONING", "CONDIMENT"], "G23": ["FLOUR", "SUGAR", "SUGAR_SUBSTITUTE", "SYRUP"],
    "G24": ["NUT_BUTTER", "HONEY"],
    "G26": ["SNACK_CHIP_AND_CRISP", "SNACK_FOOD_BAR", "CRACKER", "POPCORN", "PRETZEL"],
    "G27": ["COOKIE"], "G28": ["SUGAR_CANDY", "CHOCOLATE_CANDY"], "G29": ["NUT_AND_SEED", "FRUIT_SNACK"],
    "G30": ["WATER"], "G31": ["JUICE_AND_JUICE_DRINK", "DRINK_FLAVORED", "FLAVORED_DRINK_CONCENTRATE"],
    "G32": ["COFFEE", "TEA"], "G34": ["BABY_PRODUCT", "BABY_BOTTLE"],
    "G35": ["SHAMPOO", "SKIN_CLEANING_AGENT", "TOOTH_CLEANING_AGENT", "BODY_DEODORANT",
            "HAIR_CLEANER_CONDITIONER"],
    "G36": ["VITAMIN", "NUTRITIONAL_SUPPLEMENT", "MEDICATION", "HERBAL_SUPPLEMENT"],
    "G37": ["TOILET_PAPER", "PAPER_TOWEL", "FACIAL_TISSUE", "FOOD_STORAGE_BAG"],
    "G38": ["CLEANING_AGENT", "LAUNDRY_DETERGENT", "DISHWASHER_DETERGENT"],
    "G39": ["FOOD_STORAGE_CONTAINER", "KITCHEN", "DISHWARE_PLATE", "DRINKING_CUP"],
    "G40": ["PET_FOOD"],
}

# nhóm Tesco (gộp khi nhiều nhóm Tesco cùng rơi vào một nhóm của ta và ngược lại) -> (cột f_*, nhóm hàng)
TESCO_MERGE = {
    "rau củ, trái cây": (["fruit_veg"], ["G01", "G02"]),
    "thịt": (["meat_red", "poultry"], ["G03", "G05"]),
    "cá": (["fish"], ["G04"]),
    "sữa và chế phẩm": (["dairy"], ["G06", "G07", "G08"]),
    "trứng": (["eggs"], ["G09"]),
    "món làm sẵn": (["readymade"], ["G11", "G14"]),
    "ngũ cốc, bánh mì, mì": (["grains"], ["G15", "G17", "G18", "G19"]),
    "dầu mỡ": (["fats_oils"], ["G21"]),
    "nước sốt, gia vị": (["sauces"], ["G22"]),
    "đồ ngọt": (["sweets"], ["G13", "G16", "G26", "G27", "G28"]),
    "nước, nước ngọt": (["water", "soft_drinks"], ["G30", "G31"]),
    "cà phê, trà": (["tea_coffee"], ["G32"]),
    "đồ uống có cồn": (["beer", "wine", "spirits"], ["G33"]),
}


# ------------------------------------------------------------ Open e-commerce
def amazon_group_prices(raw: Path = RAW / "open_ecommerce", max_price: float = 500.0) -> pd.DataFrame:
    a = pd.read_csv(raw / "amazon-purchases.csv", usecols=["Category", "Purchase Price Per Unit", "Quantity"])
    a = a.rename(columns={"Purchase Price Per Unit": "price"})
    a = a[(a.price > 0) & (a.price <= max_price)]
    groups = load_groups()
    rows = []
    for g in groups.itertuples():
        cats = AMAZON_TO_GROUP.get(g.group_id, [])
        p = a[a.Category.isin(cats)].price
        rows.append({"group_id": g.group_id, "name": g.name, "value_v_assumed": g.value_v,
                     "amazon_categories": ";".join(cats), "n_purchases": int(len(p)),
                     "price_median": float(p.median()) if len(p) else np.nan,
                     "price_q25": float(p.quantile(0.25)) if len(p) else np.nan,
                     "price_q75": float(p.quantile(0.75)) if len(p) else np.nan})
    df = pd.DataFrame(rows)
    med = df.price_median.median()
    df["price_index"] = df.price_median / med                 # chỉ số tương đối (trung vị các nhóm = 1)
    return df


# ------------------------------------------------------------ Tesco
def tesco_check(raw: Path = RAW / "tesco", items_csv: Path = PROCESSED / "items" / "items.csv") -> tuple[pd.DataFrame, dict]:
    t = pd.read_csv(raw / "year_borough_grocery.csv")
    w = t.num_transactions.values / t.num_transactions.sum()   # gộp toàn London theo số giao dịch
    it = pd.read_csv(items_csv, usecols=["group_id", "n_orders"]).groupby("group_id").n_orders.sum()
    rows = []
    for name, (cols, gids) in TESCO_MERGE.items():
        rows.append({"category": name, "tesco_share": float(sum((t[f"f_{c}"].values * w).sum() for c in cols)),
                     "instacart_count": float(it.reindex(gids).fillna(0).sum()), "groups": ";".join(gids)})
    df = pd.DataFrame(rows)
    df["tesco_share"] /= df.tesco_share.sum()
    df["instacart_share"] = df.instacart_count / df.instacart_count.sum()
    rho = stats.spearmanr(df.tesco_share, df.instacart_share).statistic
    tv = 0.5 * float(np.abs(df.tesco_share - df.instacart_share).sum())
    return df, {"spearman": float(rho), "total_variation": tv, "n_categories": len(df)}


# ------------------------------------------------------------ Lyon
def weidmann(rho, v0, gamma, rho_max=5.4):
    """Giản đồ cơ bản Weidmann (1993): v = v0 (1 − exp(−γ (1/ρ − 1/ρ_max)))."""
    return v0 * (1.0 - np.exp(-gamma * (1.0 / rho - 1.0 / rho_max)))


def _read_topview(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=[0, 1, 2, 3],
                       names=["id", "frame", "x", "y"])


def lyon_fundamental_diagram(raw: Path = RAW / "lyon_crowd" / "extracted" / "Data_Madras" / "TopView_trajectories",
                             radius: float = 1.0, step: int = 15, fps: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """Mật độ cục bộ = số người trong bán kính r (kể cả chính mình) / πr²; tốc độ = quãng đường trong ±step khung.
    Trả bảng trung bình tốc độ theo khoảng mật độ và tham số Weidmann (ρ_max = 5,4 cố định)."""
    obs = []
    for p in sorted(raw.glob("TopView_*.txt")):
        d = _read_topview(p).sort_values(["id", "frame"])
        g = d.groupby("id")
        dx = g.x.shift(-step) - g.x.shift(step)
        dy = g.y.shift(-step) - g.y.shift(step)
        d["speed"] = np.hypot(dx, dy) / (2 * step / fps)
        frames = np.unique(d.frame.values)[::step]
        sub = d[d.frame.isin(frames)]
        for _, f in sub.groupby("frame"):
            xy = f[["x", "y"]].values
            if len(xy) < 2:
                continue
            cnt = np.array([len(n) for n in cKDTree(xy).query_ball_point(xy, radius)])
            rho = cnt / (np.pi * radius ** 2)
            ok = np.isfinite(f.speed.values)
            obs.append(np.column_stack([rho[ok], f.speed.values[ok]]))
    O = np.vstack(obs)
    O = O[(O[:, 1] < 3.0)]                                     # bỏ nhiễu theo dõi (> 3 m/s)
    edges = np.arange(0.25, O[:, 0].max() + 0.5, 0.5)
    b = np.digitize(O[:, 0], edges)
    tab = pd.DataFrame({"rho": O[:, 0], "speed": O[:, 1], "bin": b}).groupby("bin").agg(
        rho=("rho", "mean"), speed=("speed", "mean"), n=("speed", "size")).reset_index(drop=True)
    tab = tab[tab.n >= 50]
    (v0, gamma), _ = curve_fit(weidmann, tab.rho.values, tab.speed.values, p0=(1.0, 1.0), sigma=1 / np.sqrt(tab.n),
                               bounds=([0.1, 0.01], [3.0, 20.0]))
    rho_s = stats.spearmanr(O[:, 0], O[:, 1]).statistic
    return tab, {"v0": float(v0), "gamma": float(gamma), "rho_max": 5.4, "n_obs": int(len(O)),
                 "spearman_speed_density": float(rho_s), "radius_m": radius}


def relative_speed(rho, fd: dict) -> np.ndarray:
    """v(ρ)/v0 – hệ số giảm tốc dùng cho ùn tắc trong cửa hàng (chỉ dùng hình dạng)."""
    return np.clip(weidmann(np.maximum(rho, 1e-6), 1.0, fd["gamma"], fd["rho_max"]), 0.0, 1.0)


# ------------------------------------------------------------ dựng tất cả
def build_all(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    pr = amazon_group_prices()
    pr.to_csv(out / "group_prices.csv", index=False, encoding="utf-8")
    ok = pr.dropna(subset=["price_median"])
    summary["prices"] = {"groups_with_price": int(len(ok)), "n_purchases": int(ok.n_purchases.sum()),
                         "spearman_vs_assumed_v": float(stats.spearmanr(ok.price_median, ok.value_v_assumed).statistic)}
    tc, s = tesco_check()
    tc.to_csv(out / "tesco_check.csv", index=False, encoding="utf-8")
    summary["tesco"] = s
    fd, p = lyon_fundamental_diagram()
    fd.to_csv(out / "lyon_fd.csv", index=False)
    summary["lyon"] = p
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
