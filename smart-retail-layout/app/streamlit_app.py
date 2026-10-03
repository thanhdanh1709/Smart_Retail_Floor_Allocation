"""Công cụ hỗ trợ ra quyết định bố trí mặt bằng (mục B8.4), theo pipeline
Data → Model → Optimization → Simulation → Decision (src/pipeline.py).

Chạy:  streamlit run app/streamlit_app.py
Thẻ: ① Dữ liệu – liên kết nhóm hàng, sơ đồ hiện trạng · ② Mô hình – hiệu chỉnh bằng mô phỏng và kiểm định ·
③ Tối ưu – tập Pareto · ④ Mô phỏng – sàng lọc ứng viên, chọn theo hồ sơ, tinh chỉnh ·
⑤ Quyết định – KPI có khoảng tin cậy, sơ đồ, danh sách kệ cần dời, xuất tệp.
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

from src import baselines, ga, params, pipeline as P, relocation, routing, simulate as S, solvers, viz  # noqa: E402
from src.instance import make_instance  # noqa: E402

st.set_page_config(page_title="Smart Retail Floor Allocation", layout="wide")

SAMPLES = {f"{k} – {s}": (k, s) for s in ("small", "medium", "large") for k in ("grid", "racetrack", "free")}
SAMPLES["Tình huống: siêu thị mini (vẽ lại)"] = ("case", "minimart")
KPI_VI = {"distance_m": "Quãng đường TB (m)", "trip_time_min": "Thời gian chuyến TB (phút)",
          "exposed_slots": "Số slot tiếp xúc TB", "impulse_revenue": "Doanh thu ngẫu hứng TB",
          "impulse_items": "Số món ngẫu hứng TB", "basket_value": "Giá trị giỏ TB",
          "congestion_cells": "Điểm ùn tắc"}
PROFILE_HELP = {"Tiện lợi": "ưu tiên rút ngắn quãng đường của khách",
                "Cân bằng": "ưu tiên phương án không xấu hơn hiện trạng ở cả quãng đường và doanh thu ngẫu hứng",
                "Giá trị": "ưu tiên doanh thu mua ngẫu hứng"}


# ------------------------------------------------------------------ cache
@st.cache_resource(show_spinner=False)
def warm():
    baselines.warmup()
    return True


@st.cache_resource(show_spinner="Đang dựng dữ liệu và bảng payoff…")
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


@st.cache_resource(show_spinner="Hiệu chỉnh mô hình bằng mô phỏng trên sơ đồ hiện trạng…")
def calibrated(key, _inst, _sim, n_customers):
    return P.calibrate_model(_inst, _sim, S.SimConfig(n_customers=n_customers), 1.5, n_customers,
                             ga_cfg=ga.GAConfig(pop_size=60, max_gens=300, stall_gens=50, time_limit=30))


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return buf.getvalue()


def stage(name):
    """Kết quả một khâu trong session_state (chỉ hợp lệ với cấu hình hiện tại)."""
    v = st.session_state.get(name)
    return v["data"] if v and v["key"] == key else None


def put(name, data):
    st.session_state[name] = {"key": key, "data": data}


warm()
st.title("Smart Retail Floor Allocation")
st.caption("Tối ưu hóa không gian trưng bày và luồng di chuyển của khách hàng – công cụ hỗ trợ ra quyết định. "
           "Pipeline: Dữ liệu → Mô hình → Tối ưu → Mô phỏng → Quyết định")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Mặt bằng và dữ liệu")
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

    st.header("Ràng buộc nghiệp vụ")
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
    st.header("Mô phỏng")
    n_cust = st.slider("Số khách mỗi lần mô phỏng", 500, 5000, 1500, step=500)
inst = build_instance(kind, scale, grid_text, level, n, uploaded_key, tuple(fixed_items), sep_items, R) \
    if fixed_items else inst0
key = f"{kind}|{scale}|{level}|{n}|{uploaded_key}|{fixed_items}|{sep_items}|{R}|{n_cust}"
sim = simulator(inst, key)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nhóm hàng (n)", inst.n)
c2.metric("Vị trí trưng bày (m)", inst.m)
c3.metric("Slot lạnh / nhóm lạnh", f"{int(inst.is_cold.sum())} / {int(inst.meta.needs_cold.sum())}")
c4.metric("Ràng buộc tách xa", len(inst.sep_i))

tabs = st.tabs(["① Dữ liệu", "② Mô hình", "③ Tối ưu", "④ Mô phỏng", "⑤ Quyết định"])

# ------------------------------------------------------------ ① Dữ liệu
with tabs[0]:
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
        st.subheader("Sơ đồ hiện trạng")
        fig, ax = plt.subplots(figsize=(7, 4))
        viz.plot_floorplan(inst.fp, ax, inst, inst.current, "Sơ đồ hiện trạng (mã nhóm)")
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(inst.meta[["category_id", "name", "needs_cold", "value_v", "impulse_p", "popularity_f",
                            "dwell_median_s"]], width="stretch", hide_index=True)

# ------------------------------------------------------------- ② Mô hình
with tabs[1]:
    st.markdown("Mô hình được **hiệu chỉnh bằng mô phỏng trên sơ đồ hiện trạng**: λ khớp 1,5 món ngẫu hứng/khách; "
                "e_k = tỷ lệ khách đi qua vùng tiếp xúc của slot k; "
                "q_i = p_i(1 − f_i)(1 − e^(−λ·t_i)) – xác suất một lần đi qua dẫn tới mua ngẫu hứng. "
                "Khi đó Z2 ≈ doanh thu ngẫu hứng kỳ vọng mỗi khách.")
    cal, calib = calibrated(key, inst, sim, n_cust)
    cfg = calib.sim_cfg
    m1, m2, m3 = st.columns(3)
    m1.metric("λ (hiệu chỉnh)", f"{calib.lam:.4f}")
    m2.metric("Z2 dự báo – hiện trạng", f"{cal.z2(inst.current):.3f}")
    res0 = sim.run(inst.current, cfg)
    m3.metric("Doanh thu ngẫu hứng mô phỏng – hiện trạng", f"{res0['kpi']['impulse_revenue']:.3f}")
    colA, colB = st.columns([1, 1])
    with colA:
        fig, ax = plt.subplots(figsize=(7, 4))
        viz.plot_heatmap(inst.fp, res0["traffic"], ax, "Lưu lượng hiện trạng (lượt đi qua/khách)")
        st.pyplot(fig)
        plt.close(fig)
    with colB:
        coef = pd.DataFrame({"name": inst.names, "p_i (gốc)": inst.p[:inst.n].round(3),
                             "q_i (hiệu chỉnh)": calib.q[:inst.n].round(4), "v_i": inst.v[:inst.n]})
        st.dataframe(coef.sort_values("q_i (hiệu chỉnh)", ascending=False), hide_index=True, width="stretch",
                     height=330)
    st.subheader("Kiểm định mô hình")
    k_val = st.slider("Số sơ đồ ngẫu nhiên dùng kiểm định", 10, 60, 20, step=10)
    if st.button("Kiểm định: mục tiêu mô hình có xếp hạng đúng KPI mô phỏng?"):
        perms = P.sample_layouts(inst, k_val) + [inst.current]
        bar = st.progress(0.0)
        rows = []
        for i, p in enumerate(perms):
            rows.append(P.simulate_mean(sim, p, cfg, 1, P.SCREEN_SEED))
            bar.progress((i + 1) / len(perms))
        val = P.model_validity({"gốc (e hình học, p)": inst, "hiệu chỉnh (e mô phỏng, q)": cal},
                               pd.DataFrame(rows), perms)
        put("validity", val)
    val = stage("validity")
    if val is not None:
        st.dataframe(val.drop(columns="n_layouts").round(3), hide_index=True, width="stretch")
        st.caption("Hệ số Spearman: 1 = mô hình xếp hạng sơ đồ đúng như mô phỏng, 0 = không liên quan.")

# --------------------------------------------------------------- ③ Tối ưu
with tabs[2]:
    objective = st.radio("Hàm mục tiêu", ["Định tuyến – độ tiếp xúc phụ thuộc sơ đồ (khuyến nghị)",
                                          "Mô hình QAP hiệu chỉnh – độ tiếp xúc cố định"], horizontal=True)
    budget = st.slider("Ngân sách NSGA-II (giây)", 3, 120, 15)
    if st.button("Tạo tập phương án Pareto", type="primary"):
        with st.spinner("NSGA-II lai + GA hai đầu trên mô hình QAP hiệu chỉnh…"):
            front = P.candidates(cal, budget, seed=0)
            obj = cal
        if objective.startswith("Định tuyến"):
            with st.spinner("NSGA-II trên Z1/Z2 theo định tuyến (khởi tạo từ tập Pareto QAP + hiện trạng)…"):
                rm = routing.RouteModel(inst, calib.lam, cfg.sigma, 1500)
                front, obj = P.candidates_route(cal, rm, front, budget, seed=0)
        put("front", front)
        put("obj", obj)
        for s_ in ("screen", "picks", "final"):
            st.session_state.pop(s_, None)
    front, obj = stage("front"), stage("obj")
    if front is not None:
        Z = front["Z"]
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers+lines", name="Pareto",
                                  text=[f"#{i}" for i in range(len(Z))]))
        for prof, i in front["model_picks"].items():
            figp.add_trace(go.Scatter(x=[Z[i, 0]], y=[Z[i, 1]], mode="markers", name=f"{prof} (theo mô hình)",
                                      marker=dict(size=11, symbol="circle-open")))
        figp.add_trace(go.Scatter(x=[obj.z1(inst.current)], y=[obj.z2(inst.current)], mode="markers",
                                  name="hiện trạng", marker=dict(size=12, symbol="x", color="gray")))
        figp.update_layout(xaxis_title="Z1 – quãng đường kỳ vọng (thấp = tiện lợi)",
                           yaxis_title="Z2 – doanh thu ngẫu hứng kỳ vọng (cao = tốt)", height=420)
        st.plotly_chart(figp, width="stretch")
        st.success(f"{len(Z)} phương án không trội – chuyển sang ④ để đánh giá bằng mô phỏng.")

# ------------------------------------------------------------- ④ Mô phỏng
with tabs[3]:
    front = stage("front")
    if front is None:
        st.info("Tạo tập Pareto ở thẻ ③ trước.")
    else:
        cands = list(front["perms"]) + [np.asarray(inst.current)]       # hiện trạng cũng là một ứng viên
        if st.button("Mô phỏng sàng lọc mọi phương án", type="primary"):
            bar = st.progress(0.0)
            ref = P.simulate_mean(sim, inst.current, cfg, 1, P.SCREEN_SEED)
            rows = []
            for i, p in enumerate(cands):
                rows.append({"cand": i, **P.simulate_mean(sim, p, cfg, 1, P.SCREEN_SEED)})
                bar.progress((i + 1) / len(cands))
            put("screen", {"df": pd.DataFrame(rows), "ref": ref})
            st.session_state.pop("picks", None)
        scr = stage("screen")
        if scr is not None:
            df, ref = scr["df"].copy(), scr["ref"]
            df["Δ quãng đường (%)"] = 100 * (df.distance_m / ref["distance_m"] - 1)
            df["Δ doanh thu ngẫu hứng (%)"] = 100 * (df.impulse_revenue / ref["impulse_revenue"] - 1)
            figs = go.Figure()
            figs.add_trace(go.Scatter(x=df["Δ quãng đường (%)"], y=df["Δ doanh thu ngẫu hứng (%)"],
                                      mode="markers", name="ứng viên Pareto", text=[f"#{i}" for i in df.cand]))
            figs.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="hiện trạng",
                                      marker=dict(size=13, symbol="x", color="gray")))
            figs.update_layout(xaxis_title="Δ quãng đường so với hiện trạng (%) – trái = tốt",
                               yaxis_title="Δ doanh thu ngẫu hứng (%) – trên = tốt", height=420)
            figs.add_hline(y=0, line_width=.5)
            figs.add_vline(x=0, line_width=.5)
            st.plotly_chart(figs, width="stretch")
            st.caption("Góc trên bên trái: phương án tốt hơn hiện trạng ở cả hai KPI.")
            n_ref = st.slider("Số bước tinh chỉnh 2-swap bằng mô phỏng / hồ sơ (0 = bỏ qua)", 0, 300, 60, step=20)
            if st.button("Chọn theo hồ sơ và tinh chỉnh"):
                picks = {}
                for prof in P.PROFILES:
                    si = P.select(scr["df"], ref, prof)
                    perm = cands[si]
                    if n_ref:
                        with st.spinner(f"Tinh chỉnh hồ sơ {prof}…"):
                            perm, _ = P.refine(sim, cal, perm, ref, prof, cfg, n_ref, 1)
                    lab = "hiện trạng" if si == len(cands) - 1 else f"Pareto #{si}"
                    picks[prof] = {"perm": perm, "source": lab + (f" + {n_ref} bước 2-swap" if n_ref else "")}
                put("picks", picks)
                st.session_state.pop("final", None)
            picks = stage("picks")
            if picks:
                st.success("Đã chọn: " + "; ".join(f"{p}: {v['source']}" for p, v in picks.items())
                           + " – xem kết quả ở thẻ ⑤.")

# ----------------------------------------------------------- ⑤ Quyết định
with tabs[4]:
    picks = stage("picks")
    if not picks:
        st.info("Hoàn tất ④ (sàng lọc + chọn theo hồ sơ) trước.")
    else:
        prof = st.radio("Hồ sơ quyết định", list(P.PROFILES), index=1, horizontal=True,
                        format_func=lambda p: f"{p} – {PROFILE_HELP[p]}")
        n_rep = st.slider("Số lần lặp đánh giá cuối (hạt giống độc lập)", 5, 30, 10)
        if st.button("Đánh giá cuối", type="primary"):
            plans = {"Hiện trạng": inst.current, **{p: v["perm"] for p, v in picks.items()}}
            with st.spinner("Mô phỏng các phương án trên hạt giống độc lập…"):
                put("final", P.final_eval(sim, plans, cfg, n_rep)[1])
        fin = stage("final")
        perm = np.asarray(picks[prof]["perm"])
        if fin is not None:
            sub = fin[fin.plan.isin(["Hiện trạng", prof])].pivot_table(index="kpi", columns="plan",
                                                                        values=["mean", "ci95"])
            cmp_ = fin[fin.plan == prof].set_index("kpi")
            rows = [{"Chỉ số": KPI_VI[k], "Hiện trạng": f"{sub.loc[k, ('mean', 'Hiện trạng')]:.3f}",
                     "Đề xuất": f"{sub.loc[k, ('mean', prof)]:.3f} ± {sub.loc[k, ('ci95', prof)]:.3f}",
                     "Thay đổi (%)": round(cmp_.loc[k, "change_pct"], 2),
                     "p (Wilcoxon)": round(cmp_.loc[k, "p_value"], 4)} for k in KPI_VI]
            st.subheader(f"KPI mô phỏng – phương án {prof} ({picks[prof]['source']})")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        a, b = sim.run(inst.current, cfg), sim.run(perm, cfg)
        vmax = np.nanmax([np.nanmax(a["traffic"]), np.nanmax(b["traffic"])])
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
        viz.plot_floorplan(inst.fp, ax[0, 0], inst, inst.current, "Hiện trạng")
        viz.plot_floorplan(inst.fp, ax[0, 1], inst, perm, f"Đề xuất – {prof} (đỏ: phải dời)", highlight_moved=True)
        viz.plot_heatmap(inst.fp, a["traffic"], ax[1, 0], "Lưu lượng – hiện trạng", vmax)
        viz.plot_heatmap(inst.fp, b["traffic"], ax[1, 1], "Lưu lượng – đề xuất", vmax)
        png = fig_png(fig)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Danh sách kệ cần dời theo thứ tự ưu tiên")
        alpha = 1.0 - P.PROFILES[prof]["beta"]
        seq = relocation.relocation_sequence(cal, inst.current, perm, alpha)
        st.dataframe(seq.round(4), hide_index=True, width="stretch")
        assign = cal.assignment_frame(perm)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Sơ đồ đề xuất (PNG)", png, "so_do_de_xuat.png", "image/png")
        d2.download_button("⬇️ Bảng gán nhóm → slot (CSV)", assign.to_csv(index=False).encode("utf-8-sig"),
                           "gan_nhom_slot.csv", "text/csv")
        d3.download_button("⬇️ Danh sách kệ cần dời (CSV)", seq.to_csv(index=False).encode("utf-8-sig"),
                           "danh_sach_doi_ke.csv", "text/csv")
