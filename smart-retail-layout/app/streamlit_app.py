"""Công cụ hỗ trợ ra quyết định bố trí mặt bằng (mục B8.4).

Chạy:  streamlit run app/streamlit_app.py
Luồng: (1) tải mặt bằng + dữ liệu → (2) phân tích liên kết, bản đồ nhiệt hiện trạng →
(3) chọn ràng buộc, α hoặc NSGA-II → (4) so sánh trước/sau → (5) xuất sơ đồ, danh sách kệ cần dời.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baselines, ga, moo, params, relocation, simulate as S, solvers, viz  # noqa: E402
from src.instance import make_instance  # noqa: E402

st.set_page_config(page_title="Smart Retail Floor Allocation", layout="wide")

SAMPLES = {f"{k} – {s}": (k, s) for s in ("small", "medium", "large") for k in ("grid", "racetrack", "free")}
SAMPLES["Tình huống: siêu thị mini (vẽ lại)"] = ("case", "minimart")
KPI_VI = {"distance_m": "Quãng đường TB (m)", "trip_time_min": "Thời gian chuyến TB (phút)",
          "exposed_slots": "Số slot tiếp xúc TB", "impulse_revenue": "Doanh thu ngẫu hứng TB",
          "impulse_items": "Số món ngẫu hứng TB", "basket_value": "Giá trị giỏ TB",
          "congestion_cells": "Điểm ùn tắc"}


# ------------------------------------------------------------------ cache
@st.cache_resource(show_spinner=False)
def warm():
    baselines.warmup()
    return True


@st.cache_resource(show_spinner="Đang dựng mô hình và bảng payoff…")
def build_instance(kind, scale, grid_text, level, n, uploaded_key, fixed_items, sep_items, R):
    par = None
    if uploaded_key is not None:
        par = st.session_state["uploaded_par"]
    elif level:
        par = params.load(level)
    grid = grid_text.split() if grid_text else None
    if kind == "case":
        grid = (ROOT / "data" / "floorplans" / "case_minimart.txt").read_text(encoding="utf-8").split()
    inst = make_instance(kind if kind != "upload" else "custom", scale, n=n, par=par, grid=grid,
                         level=level or "group", separate=list(sep_items) or [],
                         fixed=dict(fixed_items) or None, R=R)
    solvers.compute_payoff(inst, ga_cfg=ga.GAConfig(pop_size=60, max_gens=300, stall_gens=50, time_limit=30))
    return inst


@st.cache_resource(show_spinner=False)
def simulator(_inst, key):
    return S.Simulator(_inst)


@st.cache_data(show_spinner="Hiệu chỉnh λ trên sơ đồ hiện trạng…")
def calibrate(key, _sim, _perm, n_customers):
    return S.calibrate_lambda(_sim, np.asarray(_perm), 1.5, S.SimConfig(n_customers=n_customers),
                              n_customers=min(1500, n_customers))


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return buf.getvalue()


warm()
st.title("Smart Retail Floor Allocation")
st.caption("Tối ưu hóa không gian trưng bày và luồng di chuyển của khách hàng – công cụ hỗ trợ ra quyết định")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Mặt bằng và dữ liệu")
    src = st.radio("Mặt bằng", ["Mẫu có sẵn", "Tải tệp lưới (.txt)"])
    grid_text = None
    if src == "Mẫu có sẵn":
        label = st.selectbox("Chọn mặt bằng", list(SAMPLES), index=3)
        kind, scale = SAMPLES[label]
    else:
        up = st.file_uploader("Tệp lưới: X tường, A lối đi, S kệ, R kệ lạnh, E cửa vào, C thu ngân", type=["txt"])
        if up is None:
            st.info("Chưa có tệp – dùng mẫu grid – medium.")
            kind, scale = "grid", "medium"
        else:
            grid_text = up.getvalue().decode("utf-8")
            kind, scale = "upload", "custom"
    data_src = st.radio("Dữ liệu giỏ hàng", ["Instacart – 40 nhóm", "Instacart – 132 aisle",
                                             "Tải dữ liệu cửa hàng"])
    level, uploaded_key = None, None
    if data_src == "Instacart – 40 nhóm":
        level = "group"
    elif data_src == "Instacart – 132 aisle":
        level = "aisle"
    else:
        cu = st.file_uploader("categories.csv (mẫu PL1)", type=["csv"])
        bu = st.file_uploader("baskets.csv (order_id, category_id)", type=["csv"])
        if cu is not None and bu is not None:
            st.session_state["uploaded_par"] = params.from_uploaded(pd.read_csv(cu), pd.read_csv(bu))
            uploaded_key = f"{cu.name}|{bu.name}|{cu.size}|{bu.size}"
        else:
            st.info("Chưa đủ tệp – dùng Instacart 40 nhóm.")
            level = "group"
    if kind in ("grid", "racetrack", "free") and scale == "large" and level == "group":
        st.caption("Mặt bằng lớn dùng mức 132 aisle.")
        level = "aisle"
    base_meta = st.session_state["uploaded_par"]["meta"] if uploaded_key else params.load(level)["meta"]
    n_all = len(base_meta)
    default_n = {"small": 20, "minimart": 30}.get(scale, n_all)
    n = st.slider("Số nhóm hàng (phổ biến nhất)", 5, n_all, min(default_n, n_all))

    st.header("2. Ràng buộc nghiệp vụ")
    names = dict(zip(base_meta.category_id, base_meta["name"]))
    R_on = st.checkbox("Giới hạn số nhóm được dời (R)")
    R = st.slider("R", 0, n, min(8, n)) if R_on else -1
    sep_default = [f"{a} | {b}" for a, b in (("G38", "G01"), ("G38", "G03")) if a in names and b in names]
    ids = list(names)[:60]
    pair_opts = sep_default + [f"{a} | {b}" for a in ids for b in ids if a < b][:400]
    sep_sel = st.multiselect("Cặp nhóm cần tách xa", list(dict.fromkeys(pair_opts)), default=sep_default,
                             format_func=lambda s: " – ".join(names.get(x.strip(), x) for x in s.split("|")))
    delta = st.number_input("Khoảng cách tối thiểu δ (m)", 1, 30, 6)
    sep_items = tuple((s.split("|")[0].strip(), s.split("|")[1].strip(), int(delta)) for s in sep_sel)

try:
    inst0 = build_instance(kind, scale, grid_text, level, n, uploaded_key, (), sep_items, R)
except Exception as exc:  # dữ liệu tải lên sai định dạng
    st.error(f"Không dựng được mô hình: {exc}")
    st.stop()

with st.sidebar:
    slot_ids = inst0.fp.slots_frame().slot_id.values[inst0.slot_idx]
    fixed_cats = st.multiselect("Nhóm có vị trí cố định", list(inst0.meta.category_id),
                                format_func=lambda c: f"{c} – {names.get(c, c)}")
    fixed_items = []
    for c in fixed_cats:
        i = int(np.where(inst0.meta.category_id.values == c)[0][0])
        ok = [k for k in range(inst0.m) if inst0.allowed[i, k]]
        cur = int(inst0.k0[i]) if inst0.k0[i] in ok else ok[0]
        k = st.selectbox(f"Slot cho {names.get(c, c)}", ok, index=ok.index(cur), format_func=lambda k: slot_ids[k])
        fixed_items.append((c, int(k)))
inst = build_instance(kind, scale, grid_text, level, n, uploaded_key, tuple(fixed_items), sep_items, R) \
    if fixed_items else inst0
key = f"{kind}|{scale}|{level}|{n}|{uploaded_key}|{fixed_items}|{sep_items}|{R}"
sim = simulator(inst, key)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nhóm hàng (n)", inst.n)
c2.metric("Vị trí trưng bày (m)", inst.m)
c3.metric("Slot lạnh / nhóm lạnh", f"{int(inst.is_cold.sum())} / {int(inst.meta.needs_cold.sum())}")
c4.metric("Ràng buộc tách xa", len(inst.sep_i))

tab1, tab2, tab3 = st.tabs(["📊 Phân tích", "⚙️ Tối ưu", "🔁 So sánh & xuất"])

# ---------------------------------------------------------------- tab 1
with tab1:
    n_cust = st.slider("Số khách mô phỏng", 500, 10000, 3000, step=500)
    lam = calibrate(key + str(n_cust), sim, inst.current, n_cust)
    scfg = S.SimConfig(n_customers=n_cust, lam=lam)
    st.session_state["scfg"] = scfg
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Cặp nhóm hàng liên kết mạnh")
        f_ = inst.f[:inst.n]
        W_ = inst.W[:inst.n, :inst.n]
        par_view = {"w": W_, "lift": W_ / np.outer(f_, f_).clip(1e-12), "meta": inst.meta}
        by = st.radio("Xếp theo", ["w", "lift"], horizontal=True,
                      help="w: tỷ lệ đơn chứa cả hai nhóm; lift > 1: hai nhóm bổ trợ nhau")
        st.dataframe(params.top_pairs(par_view, 15, by=by), width="stretch", hide_index=True)
    with colB:
        st.subheader("Sơ đồ và bản đồ nhiệt hiện trạng")
        res0 = sim.run(inst.current, scfg)
        fig, ax = plt.subplots(2, 1, figsize=(7, 7))
        viz.plot_floorplan(inst.fp, ax[0], inst, inst.current, "Sơ đồ hiện trạng (mã nhóm)")
        viz.plot_heatmap(inst.fp, res0["traffic"], ax[1], "Lưu lượng (lượt đi qua/khách)")
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(inst.meta[["category_id", "name", "needs_cold", "value_v", "impulse_p", "dwell_median_s"]],
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 2
with tab2:
    method = st.radio("Phương pháp", ["GA – một phương án theo trọng số α", "NSGA-II – tập Pareto"], horizontal=True)
    budget = st.slider("Ngân sách thời gian (giây)", 3, 120, 15)
    if method.startswith("GA"):
        alpha = st.slider("α – trọng số mục tiêu tiện lợi (1 = chỉ quãng đường, 0 = chỉ giá trị)", 0.0, 1.0, 0.5, 0.05)
        if st.button("Chạy tối ưu", type="primary"):
            with st.spinner("GA đang chạy…"):
                r = ga.run(inst, alpha, ga.GAConfig(seed=0, time_limit=budget, stall_gens=150))
            st.session_state["plan"] = {"perm": r.perm, "alpha": alpha, "key": key, "label": f"GA α = {alpha}"}
            st.session_state.pop("front", None)
    else:
        if st.button("Chạy NSGA-II", type="primary"):
            with st.spinner("NSGA-II đang chạy…"):
                r = moo.nsga2(inst, time_limit=budget, ls_prob=0.1, seed=0)
            st.session_state["front"] = {"r": r, "key": key}
        fr = st.session_state.get("front")
        if fr and fr["key"] == key and len(fr["r"]["F"]):
            F, Z = fr["r"]["F"], fr["r"]["Z"]
            kn = moo.knee_point(F)
            idx = st.slider("Chọn phương án trên tập Pareto (0 = tiện lợi nhất)", 0, len(F) - 1, int(kn))
            figp = go.Figure()
            figp.add_trace(go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers+lines", name="Pareto",
                                      text=[f"#{i}" for i in range(len(F))]))
            figp.add_trace(go.Scatter(x=[Z[kn, 0]], y=[Z[kn, 1]], mode="markers", name="điểm gối",
                                      marker=dict(size=14, symbol="circle-open", color="black")))
            figp.add_trace(go.Scatter(x=[Z[idx, 0]], y=[Z[idx, 1]], mode="markers", name="đang chọn",
                                      marker=dict(size=12, color="red")))
            figp.add_trace(go.Scatter(x=[inst.z1(inst.current)], y=[inst.z2(inst.current)], mode="markers",
                                      name="hiện trạng", marker=dict(size=12, symbol="x", color="gray")))
            figp.update_layout(xaxis_title="Z1 – quãng đường kỳ vọng (thấp = tiện lợi)",
                               yaxis_title="Z2 – giá trị kỳ vọng (cao = tốt)", height=420)
            st.plotly_chart(figp, width="stretch")
            st.session_state["plan"] = {"perm": fr["r"]["perms"][idx], "alpha": 0.5, "key": key,
                                        "label": f"Pareto #{idx}" + (" (điểm gối)" if idx == kn else "")}
    plan = st.session_state.get("plan")
    if plan and plan["key"] == key:
        ev = inst.evaluate(plan["perm"], 0.5)
        cur = inst.evaluate(inst.current, 0.5)
        st.success(f"Đã có phương án: {plan['label']} – dời {ev['moved']} / {inst.n} nhóm; "
                   f"Z1 {cur['z1']:.1f} → {ev['z1']:.1f}; Z2 {cur['z2']:.3f} → {ev['z2']:.3f}; "
                   f"vi phạm: {ev['compat'] + ev['separate'] + ev['relocation_excess']}")

# ---------------------------------------------------------------- tab 3
with tab3:
    plan = st.session_state.get("plan")
    if not plan or plan["key"] != key:
        st.info("Chạy tối ưu ở thẻ ⚙️ trước.")
    else:
        scfg = st.session_state.get("scfg", S.SimConfig(n_customers=3000))
        perm = np.asarray(plan["perm"])
        with st.spinner("Mô phỏng hai sơ đồ…"):
            a = sim.run(inst.current, scfg)
            b = sim.run(perm, scfg)
        rows = []
        for k_, nm in KPI_VI.items():
            x0, x1 = a["kpi"][k_], b["kpi"][k_]
            rows.append({"Chỉ số": nm, "Hiện trạng": round(x0, 3), "Đề xuất": round(x1, 3),
                         "Thay đổi (%)": round(100 * (x1 - x0) / x0, 2) if x0 else np.nan})
        st.subheader("KPI mô phỏng (so sánh tương đối)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        vmax = np.nanmax([np.nanmax(a["traffic"]), np.nanmax(b["traffic"])])
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
        viz.plot_floorplan(inst.fp, ax[0, 0], inst, inst.current, "Hiện trạng")
        viz.plot_floorplan(inst.fp, ax[0, 1], inst, perm, f"Đề xuất – {plan['label']} (đỏ: phải dời)",
                           highlight_moved=True)
        viz.plot_heatmap(inst.fp, a["traffic"], ax[1, 0], "Lưu lượng – hiện trạng", vmax)
        viz.plot_heatmap(inst.fp, b["traffic"], ax[1, 1], "Lưu lượng – đề xuất", vmax)
        png = fig_png(fig)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Danh sách kệ cần dời theo thứ tự ưu tiên")
        seq = relocation.relocation_sequence(inst, inst.current, perm, plan["alpha"])
        st.dataframe(seq.round(4), hide_index=True, width="stretch")
        assign = inst.assignment_frame(perm)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Sơ đồ đề xuất (PNG)", png, "so_do_de_xuat.png", "image/png")
        d2.download_button("⬇️ Bảng gán nhóm → slot (CSV)", assign.to_csv(index=False).encode("utf-8-sig"),
                           "gan_nhom_slot.csv", "text/csv")
        d3.download_button("⬇️ Danh sách kệ cần dời (CSV)", seq.to_csv(index=False).encode("utf-8-sig"),
                           "danh_sach_doi_ke.csv", "text/csv")
