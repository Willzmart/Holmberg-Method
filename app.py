"""Diseño de mallas de perforación y voladura para avance horizontal — Método Holmberg.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import fragmentation as fr
from core import sequencing as sq
from core import viz
from core.holmberg import (CARTRIDGES, EXPLOSIVES, ZONE_COLORS, ZONES, DesignOptions, Drill, Explosive, Rock,
                           run_design, summarize)
from core.sections import GALLERY_TYPES, PRESETS, build_section

st.set_page_config(page_title="Mallas Holmberg — Avance subterráneo", layout="wide")
ASSETS = Path(__file__).parent / "assets"
SYSTEMS = ["Carmex (mecha de seguridad + fulminante)", "No eléctrico", "Electrónico"]

ss = st.session_state
ss.setdefault("B_override", None)
ss.setdefault("SB_override", None)
ss.setdefault("seq", None)
ss.setdefault("seq_sig", None)

# =============================================================================
# BARRA LATERAL — DATOS DE ENTRADA (Fase 0)
# =============================================================================
st.sidebar.title("Datos de entrada")

with st.sidebar.expander("Sección de la galería", expanded=True):
    kind = st.selectbox("Tipo de galería", GALLERY_TYPES, index=0)
    preset = st.selectbox("Sección típica", list(PRESETS), index=6)
    A0, H0 = PRESETS[preset] or (3.5, 3.5)
    c1, c2 = st.columns(2)
    A = c1.number_input("Ancho A (m)", 1.5, 10.0, float(A0), 0.05)
    Hs = c2.number_input("Alto H (m)", 1.5, 10.0, float(H0), 0.05)
    r_arch, top_w = None, None
    if kind.startswith("Baúl"):
        r_arch = st.number_input("Radio del baúl r (m)", 0.1, A / 2, min(1.25, A / 2), 0.05,
                                 help="Radio de las esquinas en arco; r = A/2 equivale a bóveda de medio punto.")
    if kind.startswith("Trapezoidal"):
        top_w = st.number_input("Ancho en el techo (m)", 0.5, A, round(0.8 * A, 2), 0.05)
    if kind.startswith("Herradura") and Hs > 1.5 * A:
        st.caption("En herradura la altura se limita a 1.5·A.")

with st.sidebar.expander("Macizo rocoso", expanded=False):
    c1, c2 = st.columns(2)
    RMR = c1.number_input("RMR", 5, 100, 50)
    GSI = c2.number_input("GSI", 5, 100, 45)
    RQD = c1.number_input("RQD (%)", 1, 100, 55)
    UCS = c2.number_input("σci (MPa)", 5.0, 350.0, 75.0, 5.0)
    rho_r = c1.number_input("Densidad (t/m³)", 1.8, 4.5, 2.75, 0.01)
    Qb = c2.number_input("Q de Barton", 0.001, 1000.0, 1.95)
    jsf_auto = 1.0 if RMR > 60 else 0.9 if RMR > 40 else 0.8 if RMR > 20 else 0.7
    JSF = st.number_input("JSF (resistencia de discontinuidades)", 0.5, 1.0, jsf_auto, 0.05,
                          help="Tabla del libro: buena 1.0 · regular 0.9 · mala 0.8 · muy mala 0.7")

with st.sidebar.expander("Perforación", expanded=False):
    c1, c2 = st.columns(2)
    d1 = c1.number_input("Ø producción (mm)", 25.0, 76.0, 45.0, 1.0)
    d2 = c2.number_input("Ø alivio / rimado (mm)", 45.0, 152.0, 102.0, 1.0)
    NB = c1.number_input("N° de alivios", 1, 4, 2)
    bar_ft = c2.number_input("Longitud de barra (pies)", 4.0, 20.0, 11.0, 0.5)
    eff = st.slider("Eficiencia de perforación (%)", 70, 100, 95)
    F_method = st.selectbox("Error de perforación F", ["Dato directo", "Ec. 26", "Ec. 27"])
    F_val, alpha, e_c = 0.04, 10.0, 20.0
    if F_method == "Dato directo":
        F_val = st.number_input("F (m)", 0.0, 0.5, 0.04, 0.005)
    elif F_method == "Ec. 26":
        c1, c2 = st.columns(2)
        alpha = c1.number_input("α desviación (mm/m)", 0.0, 50.0, 10.0)
        e_c = c2.number_input("e emboquille (mm)", 0.0, 100.0, 20.0)
    else:
        st.caption("La Ec. 27 da valores altos de F; revise el resultado.")


def explosive_picker(label, default_name, default_cart, key):
    names = list(EXPLOSIVES) + ["Personalizado"]
    name = st.selectbox(label, names, index=names.index(default_name), key=key + "n")
    if name == "Personalizado":
        c1, c2 = st.columns(2)
        vod = c1.number_input("VOD (m/s)", 1500, 8000, 5000, key=key + "v")
        rho = c2.number_input("Densidad (g/cm³)", 0.6, 1.6, 1.10, key=key + "r")
        rws = c1.number_input("RWS (% ANFO)", 40, 180, 90, key=key + "w") / 100
        w = c2.number_input("Peso cartucho (kg)", 0.02, 1.0, 0.20, key=key + "c")
        Lc = c1.number_input("Long. cartucho (m)", 0.1, 0.8, 0.3048, key=key + "l")
        return Explosive(name="Personalizado", vod=vod, rho=rho, rws=rws, cart="—", w_cart=w, L_cart=Lc)
    e = EXPLOSIVES[name]
    cart = st.selectbox("Cartucho", list(CARTRIDGES), index=list(CARTRIDGES).index(default_cart), key=key + "k")
    return Explosive(name=name, vod=e["vod"], rho=e["rho"], rws=e["rws"], cart=cart,
                     w_cart=e["w"][cart], L_cart=CARTRIDGES[cart][1])


with st.sidebar.expander("Explosivos", expanded=False):
    exp_main = explosive_picker("Explosivo principal (cuele, ayudas, hastiales, arrastre)", "Emulnor 1000",
                                '1 1/4" x 12"', "em")
    st.caption(f"VOD {exp_main.vod:.0f} m/s · ρ {exp_main.rho} g/cm³ · RWS {exp_main.rws*100:.0f} % · "
               f"{exp_main.w_cart:.3f} kg/cart.")
    exp_cor = explosive_picker("Explosivo de corona (voladura controlada)", "Emulnor 500", '1" x 12"', "ec")

with st.sidebar.expander("Parámetros de diseño", expanded=False):
    cut_method = st.radio("Burden del arranque", ["Holmberg", "Pearse"], horizontal=True)
    cut_factor = st.radio("B_max arranque (Holmberg)", [1.7, 1.5], horizontal=True,
                          format_func=lambda v: f"{v}·Ø_equiv") if cut_method == "Holmberg" else 1.7
    nq_mode = st.radio("N° de cuadrantes", ["Automático (A_h ≤ √H')", "Manual"], horizontal=False)
    n_quad = None if nq_mode.startswith("Auto") else st.slider("Cuadrantes", 1, 4, 3)
    c1, c2 = st.columns(2)
    gamma = c1.number_input("Ángulo de contorno γ (°)", 0.0, 10.0, 3.0, 0.5)
    K_cor = c2.number_input("K corona (S = K·Ø1)", 10.0, 20.0, 15.0, 0.5)
    c_rule = st.selectbox("Constante corregida Ĉ", ["Iterativa", "c + 0.05"],
                          help="Iterativa: Ĉ = c+0.05 si B ≥ 1.4 m; c+0.07/B si B < 1.4 m. "
                               "'c + 0.05' reproduce el Ejemplo N°06 del libro.")
    offset = st.number_input("Separación de los taladros de contorno al perfil (m)", 0.0, 0.3, 0.10, 0.01)
    ch_mode = st.checkbox("Fijar altura del centro del cuele", False)
    cut_h = st.number_input("Altura del centro del cuele (m)", 0.5, 8.0, 1.6, 0.05) if ch_mode else None
    blast_eff = st.slider("Eficiencia de voladura (avance/H') %", 70, 100, 95) / 100

with st.sidebar.expander("Iniciación", expanded=False):
    system = st.selectbox("Sistema de iniciación", SYSTEMS)
    primer = st.radio("Posición del cebo", ["Fondo", "Boca"], horizontal=True,
                      help="Fondo = cebado inverso (recomendado).")

with st.sidebar.expander("Costos unitarios", expanded=False):
    c_drill = st.number_input("Perforación (US$/m)", 0.0, 50.0, 2.5, 0.1)
    c_exp = st.number_input("Explosivo (US$/kg)", 0.0, 50.0, 4.0, 0.1)
    c_acc = st.number_input("Accesorios por taladro (US$)", 0.0, 50.0, 3.0, 0.1)

# =============================================================================
# CÁLCULO
# =============================================================================
try:
    section = build_section(kind, A, Hs, r_arch, top_w)
except ValueError as e:
    st.error(str(e))
    st.stop()

rock = Rock(RMR=RMR, GSI=GSI, RQD=RQD, UCS=UCS, rho=rho_r, Q=Qb, JSF=JSF)
drill = Drill(d1_mm=d1, d2_mm=d2, NB=int(NB), bar_m=bar_ft * 0.3048, eff=eff / 100, F_method=F_method,
              F_value=F_val, alpha_mm_m=alpha, e_mm=e_c)
opts = DesignOptions(cut_method=cut_method, cut_factor=float(cut_factor), n_quadrants=n_quad, gamma_deg=gamma,
                     K_corona=K_cor, c_hat_rule=c_rule, contour_offset=offset, cut_height=cut_h,
                     blast_eff=blast_eff, B_prod_override=ss.B_override, SB_prod=ss.SB_override or 1.25)
try:
    design = run_design(section, rock, drill, exp_main, exp_cor, opts)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo generar la malla con estos datos: {e}")
    st.stop()

P = design.p
H = design.holes
summary = summarize(design, dict(drill=c_drill, exp=c_exp, acc=c_acc))
sig = (len(H), round(float(H.x.sum()), 4), round(float(H.kg.sum()), 3), system)
if ss.seq_sig != sig:
    ss.seq = None

# =============================================================================
# ENCABEZADO
# =============================================================================
st.title("Diseño de malla de perforación y voladura — Método Holmberg")
st.caption(f"{kind} · {A:.2f} × {section.height:.2f} m · área {section.area:.2f} m² · "
           f"H' = {P['Hp']:.2f} m · {len(H)} taladros cargados + {int(NB)} de alivio")
m = st.columns(6)
m[0].metric("Taladros", f"{len(H)} + {int(NB)}")
m[1].metric("Avance", f"{summary['Avance por disparo (m)']:.2f} m")
m[2].metric("Explosivo", f"{summary['Explosivo total (kg)']:.1f} kg")
m[3].metric("Factor de carga", f"{summary['Factor de carga (kg/m³)']:.2f} kg/m³")
m[4].metric("Perforación específica", f"{summary['Perforación específica (m/m³)']:.2f} m/m³")
m[5].metric("Costo", f"US$ {summary['Costo por metro de avance (US$/m)']:.0f}/m")
if ss.B_override:
    st.info(f"Burden de producción impuesto por la optimización de P80: B = {P['Bp_prod']:.2f} m · "
            f"S/B = {opts.SB_prod:.2f}. Puede restablecerlo en la pestaña de predicción.")
for w in design.warnings:
    st.warning(w)

tabs = st.tabs(["Flujo de cálculo", "Predicción de P80", "Frente 2D", "Vista 3D y carguío",
                "Secuenciamiento", "Resumen y exportación"])

# =============================================================================
# TAB 1 — FLUJO DE CÁLCULO
# =============================================================================
with tabs[0]:
    flow = ASSETS / "diagrama_flujo.png"
    if flow.exists():
        with st.expander("Diagrama de flujo del método (referencia)"):
            st.image(str(flow), width="stretch")
    log = design.log.df()
    st.markdown("Cada fase reproduce las ecuaciones del libro (número entre paréntesis). "
                "Las filas marcadas *(libro)* son las reglas geométricas del texto; la malla usa los burdens "
                "de Holmberg y distribuye los taladros sobre la geometría real de la sección.")
    for fase in log.Fase.unique():
        with st.expander(fase, expanded=fase.startswith(("1.", "2.", "3."))):
            st.dataframe(log[log.Fase == fase].drop(columns="Fase"), hide_index=True, width="stretch")
    z = H.groupby("zona", sort=False).agg(N=("id", "count"), B_m=("B", "mean"), S_m=("S", "mean"),
                                         kg_tal=("kg", "mean"), kg=("kg", "sum")).reindex(
        [zz for zz in ZONES if zz in set(H.zona)])
    st.subheader("Distribución resultante por zona")
    st.dataframe(z.round(3), width="stretch")

# =============================================================================
# TAB 2 — PREDICCIÓN DE P80
# =============================================================================
with tabs[1]:
    st.markdown("La predicción se aplica a la **zona de producción** (ayudas). El cuele es geometría fija y la "
                "corona es voladura controlada, por lo que no se optimizan por fragmentación.")
    route = st.radio("Ruta de predicción", ["Ruta A — Kuz-Ram", "Ruta B — Modelo ML híbrido"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    P80_t = c1.number_input("P80 objetivo (cm)", 2.0, 150.0, 20.0, 0.5)
    B_lim = P["Bp_limit"]
    B_min = c2.number_input("Burden mínimo a evaluar (m)", 0.1, max(0.15, float(B_lim) - 0.05),
                             float(min(0.35, max(0.1, B_lim - 0.15))), 0.05)
    c3.metric("Límite Holmberg B_p (ayudas)", f"{B_lim:.2f} m")
    base_vals = dict(d_mm=d1, q=P["q0"], H=P["Hp"], Lc=P["Lc"], Lt=P["Lt"], RMR=RMR, GSI=GSI, RQD=RQD, UCS=UCS,
                     rho_r=rho_r, VOD=exp_main.vod, rho_e=exp_main.rho, RWS=exp_main.rws * 100, A_sec=section.area)

    if route.startswith("Ruta A"):
        with st.expander("Factor de roca A (Lilly)", expanded=True):
            mode = st.radio("Cálculo de A", ["Lilly (RMD, JF, RDI, HF)", "Valor directo"], horizontal=True)
            if mode.startswith("Lilly"):
                c1, c2, c3 = st.columns(3)
                rmd = c1.selectbox("Descripción del macizo (RMD)", list(fr.RMD_OPTS), index=1)
                jps = c2.selectbox("Espaciamiento de juntas (JPS)", list(fr.JPS_OPTS), index=1)
                jpa = c3.selectbox("Orientación de juntas (JPA)", list(fr.JPA_OPTS), index=1)
                E = c1.number_input("Módulo de Young E (GPa)", 1.0, 120.0, 35.0)
                Arf, parts = fr.rock_factor(rmd, jps, jpa, rho_r, E, UCS)
                c2.metric("Factor de roca A", f"{Arf:.2f}")
                c3.caption(" · ".join(f"{k} = {v:.1f}" for k, v in parts.items()))
            else:
                Arf = st.number_input("A", 1.0, 22.0, 7.0, 0.1)
        W = st.number_input("Desviación estándar de perforación W (m)", 0.0, 0.5, float(P["F"]), 0.005)
        k = fr.KuzRamInput(A=Arf, q=P["q0"], Lc=P["Lc"], H=P["Hp"], d_mm=d1, W=W, rws_pct=exp_main.rws * 100,
                           SB=opts.SB_prod)
        cur = fr.kuzram(P["Bp_prod"], k)
        opt_r, curve = fr.optimize_burden_kuzram(k, P80_t, B_min, B_lim)
        c = st.columns(5)
        c[0].metric("B actual (malla)", f"{cur['B']:.2f} m")
        c[1].metric("K actual", f"{cur['K']:.2f} kg/m³")
        c[2].metric("X50 (Kuznetsov)", f"{cur['X50']:.1f} cm")
        c[3].metric("n (Cunningham)", f"{cur['n']:.2f}")
        c[4].metric("P80 actual", f"{cur['P80']:.1f} cm", f"{cur['P80'] - P80_t:+.1f} vs objetivo", delta_color="inverse")
        st.latex(r"X_{50}=A\,K^{-0.8}\,Q^{1/6}\left(\tfrac{115}{RWS}\right)^{19/30}\quad"
                 r"n=\left(2.2-14\tfrac{B}{d}\right)\sqrt{\tfrac{1+S/B}{2}}\left(1-\tfrac{W}{B}\right)\tfrac{L_c}{H'}\quad"
                 r"P_{80}=X_{50}\left(\tfrac{\ln 5}{\ln 2}\right)^{1/n}")
        g1, g2 = st.columns(2)
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=curve.B, y=curve.P80, name="P80(B)", line=dict(width=3, color="#c2416f")))
        f1.add_trace(go.Scatter(x=curve.B, y=curve.X50, name="X50(B)", line=dict(dash="dot", color="#888")))
        f1.add_hline(y=P80_t, line=dict(dash="dash", color="#2ca02c"), annotation_text="P80 objetivo")
        f1.add_vline(x=B_lim, line=dict(dash="dash", color="#d62728"), annotation_text="Límite Holmberg")
        f1.add_trace(go.Scatter(x=[opt_r["B"]], y=[opt_r["P80"]], mode="markers", name="Óptimo",
                                marker=dict(size=14, symbol="star", color="#ff7f0e")))
        f1.update_layout(height=380, xaxis_title="Burden de producción B (m)", yaxis_title="Tamaño (cm)",
                         plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
        g1.plotly_chart(f1, width="stretch")
        f2 = go.Figure()
        for lab, rr, col in [("Actual", cur, "#888"), ("Óptimo", opt_r, "#c2416f")]:
            x, R = fr.rosin_rammler_curve(rr["X50"], rr["n"], xmax=max(cur["P80"], opt_r["P80"]) * 2.2)
            f2.add_trace(go.Scatter(x=x, y=R, name=f"{lab}: P80 = {rr['P80']:.1f} cm", line=dict(color=col, width=3)))
        f2.add_hline(y=80, line=dict(dash="dot", color="#555"))
        f2.update_layout(height=380, xaxis_title="Tamaño x (cm)", yaxis_title="% pasante (Rosin-Rammler)",
                         plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
        g2.plotly_chart(f2, width="stretch")
        if opt_r["converged"]:
            st.success(f"Burden óptimo B = {opt_r['B']:.2f} m · S = {opt_r['S']:.2f} m → P80 = {opt_r['P80']:.1f} cm "
                       f"(K = {opt_r['K']:.2f} kg/m³).")
        else:
            st.warning(f"El P80 objetivo no es alcanzable dentro de [{B_min:.2f}, {B_lim:.2f}] m; el más cercano es "
                       f"B = {opt_r['B']:.2f} m con P80 = {opt_r['P80']:.1f} cm. Cambie el explosivo, el diámetro "
                       "o el objetivo.")
        best_B, best_SB = opt_r["B"], opts.SB_prod
    else:
        st.markdown("Cargue un modelo entrenado (scikit-learn, XGBoost, LightGBM… serializado con **joblib** o "
                    "**pickle**). Sólo cargue archivos propios o de confianza: deserializar un modelo ejecuta código.")
        up = st.file_uploader("Modelo (.joblib / .pkl)", type=["joblib", "pkl", "pickle"])
        demo = st.checkbox("Usar el sustituto de DEMOSTRACIÓN (sólo para probar el flujo; no es un modelo real)",
                           value=up is None)
        model = None
        if up is not None and not demo:
            try:
                raw = up.getvalue()
                try:
                    import joblib
                    model = joblib.load(io.BytesIO(raw))
                except Exception:  # noqa: BLE001
                    model = pickle.loads(raw)
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo cargar el modelo: {e}")
        elif demo:
            model = fr.DemoSurrogate()
        if model is not None:
            c1, c2, c3 = st.columns(3)
            target_kind = c1.radio("El modelo predice", ["P80", "X50"], horizontal=True)
            unit = c2.selectbox("Unidad de salida", ["cm", "mm", "m"])
            scale = {"cm": 1.0, "mm": 0.1, "m": 100.0}[unit]
            n_unif = c3.number_input("n de uniformidad (si predice X50)", 0.5, 3.0, 1.5, 0.05)
            names = fr.model_feature_names(model)
            if names:
                missing = [f for f in names if f not in fr.ML_VARIABLES]
                st.caption("Variables del modelo: " + ", ".join(names))
                if missing:
                    st.error("El modelo usa variables no disponibles en la app: " + ", ".join(missing) +
                             f". Variables disponibles: {', '.join(fr.ML_VARIABLES)}.")
                    names = None
            else:
                names = st.multiselect("Orden de las variables de entrada del modelo", list(fr.ML_VARIABLES),
                                       default=["B", "S", "d_mm", "q", "Lc", "H", "RMR", "RWS"],
                                       format_func=lambda v: f"{v} — {fr.ML_VARIABLES[v]}")
            if names:
                c1, c2, c3 = st.columns(3)
                SB_lo, SB_hi = c1.slider("Rango S/B", 1.0, 1.6, (1.0, 1.5), 0.05)
                lam = c2.slider("Peso del costo λ", 0.0, 1.0, 0.15, 0.05)
                seed = c3.number_input("Semilla", 0, 9999, 0)
                v_now = fr.ml_features(P["Bp_prod"], opts.SB_prod, base_vals)
                p_now, _ = fr.ml_predict(model, names, v_now, scale, target_kind, n_unif)
                st.metric("P80 predicho con la malla actual", f"{p_now:.1f} cm", f"{p_now - P80_t:+.1f} vs objetivo",
                          delta_color="inverse")
                key = (P80_t, B_min, round(B_lim, 4), SB_lo, SB_hi, lam, int(seed), target_kind, unit, n_unif,
                       tuple(names), up.name if (up is not None and not demo) else "demo")
                if st.button("Optimizar con Evolución Diferencial (DE)"):
                    with st.spinner("Evolución Diferencial…"):
                        ss.ml_res = (key,) + fr.optimize_ml(model, names, base_vals, P80_t, (B_min, B_lim),
                                                            (SB_lo, SB_hi), scale, target_kind, n_unif, lam, int(seed))
                if ss.get("ml_res") is None or ss.ml_res[0] != key:
                    st.info("Pulse el botón para ejecutar la optimización inversa con el modelo.")
                    res = None
                else:
                    _, res, hist = ss.ml_res
            if names and res is not None:
                c = st.columns(4)
                c[0].metric("B óptimo", f"{res['B']:.2f} m")
                c[1].metric("S/B óptimo", f"{res['SB']:.2f}")
                c[2].metric("S óptimo", f"{res['S']:.2f} m")
                c[3].metric("P80 predicho", f"{res['P80']:.1f} cm")
                fh = go.Figure(go.Scatter(x=hist.B, y=hist.SB, mode="markers",
                                          marker=dict(color=hist.P80, colorscale="Viridis", size=6, showscale=True,
                                                      colorbar=dict(title="P80 cm")),
                                          hovertemplate="B=%{x:.2f}<br>S/B=%{y:.2f}<extra></extra>"))
                fh.add_trace(go.Scatter(x=[res["B"]], y=[res["SB"]], mode="markers", name="Óptimo",
                                        marker=dict(size=16, symbol="star", color="#ff4b00")))
                fh.update_layout(height=380, xaxis_title="B (m)", yaxis_title="S/B", plot_bgcolor="white",
                                 title=f"Evaluaciones de la Evolución Diferencial ({res['nfev']})",
                                 margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                st.plotly_chart(fh, width="stretch")
                best_B, best_SB = res["B"], res["SB"]
            else:
                best_B, best_SB = None, None
        else:
            best_B, best_SB = None, None

    c1, c2 = st.columns(2)
    if c1.button("Aplicar burden óptimo a la malla", type="primary", disabled=best_B is None):
        ss.B_override, ss.SB_override = float(best_B), float(best_SB)
        st.rerun()
    if c2.button("Restablecer burden de Holmberg", disabled=ss.B_override is None):
        ss.B_override, ss.SB_override = None, None
        st.rerun()

# =============================================================================
# TAB 3 — FRENTE 2D
# =============================================================================
with tabs[2]:
    c1, c2, c3, c4 = st.columns(4)
    label = c1.selectbox("Etiqueta", ["id", "periodo", "kg", "ninguna"],
                         format_func={"id": "N° de taladro", "periodo": "N° de retardo", "kg": "Carga (kg)",
                                      "ninguna": "Sin etiqueta"}.get)
    show_open = c2.checkbox("Mostrar aberturas del cuele", True)
    show_vec = c3.checkbox("Vectores de salida (requiere secuencia)", False)
    show_dims = c4.checkbox("Cotas", True)
    if label == "periodo" and ss.seq is None:
        st.info("Calcule la secuencia en la pestaña de secuenciamiento para ver los retardos.")
    st.plotly_chart(viz.front_2d(design, label, show_open, show_dims, ss.seq, show_vec and ss.seq is not None),
                    width="stretch")
    with st.expander("Detalle del cuele"):
        cut = H[H.zona.str.startswith(("Arranque", "Cuadrante"))]
        st.dataframe(cut[["id", "zona", "x", "y", "B", "S", "kg"]].round(3), hide_index=True,
                     width="stretch")

# =============================================================================
# TAB 4 — VISTA 3D Y CARGUÍO
# =============================================================================
with tabs[3]:
    c1, c2, c3, c4 = st.columns(4)
    show_shell = c1.checkbox("Túnel excavado", True)
    show_init = c2.checkbox("Iniciación (cebo + línea)", True)
    show_stem = c3.checkbox("Taco", True)
    zf = c4.multiselect("Zonas a mostrar", [z for z in ZONES if z in set(H.zona)], default=[], placeholder="Todas")
    st.plotly_chart(viz.view_3d(design, system, primer, show_shell, show_init, show_stem, zf or None, ss.seq),
                    width="stretch")
    st.plotly_chart(viz.hole_profiles(design, system, primer), width="stretch")
    st.caption(f"Taco Lt = 10·Ø1 = {P['Lt']:.2f} m (Ec. 31) · Longitud de carga Lc = H' − Lt = {P['Lc']:.2f} m (Ec. 32) · "
               f"q₀ = {P['q0']:.3f} kg/m · q corona = 90·Ø1² = {P['q_cor']:.3f} kg/m (Ec. 75).")

# =============================================================================
# TAB 5 — SECUENCIAMIENTO
# =============================================================================
with tabs[4]:
    st.markdown("Cada taladro sale hacia la cara libre creada por el taladro anterior (árbol de alivio de camino "
                "mínimo desde los alivios). Los retardos se asignan respetando precedencia, orden por zonas, carga "
                "máxima por retardo y, si se elige, corona simultánea.")
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.radio("Método", ["MILP", "Heurística"],
                          format_func={"MILP": "MILP exacto (HiGHS)", "Heurística": "Heurística rápida"}.get)
        tl = st.slider("Tiempo límite MILP (s)", 5, 120, 20) if method == "MILP" else 20
        classic = st.checkbox("Orden clásico por zonas", True,
                              help="Arranque → cuadrantes → ayudas → hastiales → corona → arrastre")
        cut_single = st.checkbox("Un taladro por retardo en el cuele", True)
        corona_sim = st.checkbox("Corona simultánea (voladura controlada)", True)
    with c2:
        nmax = st.number_input("Máx. taladros por retardo", 1, 30, 4)
        q_mode = st.radio("Carga máxima por retardo", ["Directa", "Por vibraciones (PPV)"], horizontal=True)
        if q_mode == "Directa":
            qmax = st.number_input("Q máx por retardo (kg)", 1.0, 500.0, 12.0, 0.5)
        else:
            K_s = st.number_input("K del sitio", 50.0, 3000.0, 700.0, 10.0)
            beta = st.number_input("β", 0.8, 2.5, 1.5, 0.05)
            Dm = st.number_input("Distancia a la estructura (m)", 5.0, 1000.0, 30.0, 1.0)
            ppv_lim = st.number_input("PPV admisible (mm/s)", 1.0, 500.0, 50.0, 1.0)
            qmax = sq.qmax_from_ppv(ppv_lim, Dm, K_s, beta)
            st.caption(f"Q máx = D²·(PPV/K)^(2/β) = **{qmax:.1f} kg** (constantes a calibrar en campo)")
        link = st.slider("Radio de alivio k (× burden)", 1.2, 2.5, 1.6, 0.1)
    with c3:
        st.markdown(f"**Sistema:** {system}")
        cfg = {}
        if system.startswith("Carmex"):
            cfg["burn"] = st.number_input("Tiempo de combustión de la mecha (s/m)", 80.0, 250.0, 150.0, 5.0,
                                          help="Valor referencial; use el del fabricante del lote.")
            cfg["dt_s"] = st.number_input("Retardo entre salidas (s)", 0.5, 10.0, 2.0, 0.5)
            cfg["tail"] = st.number_input("Mecha sobrante fuera del taladro (m)", 0.1, 1.5, 0.35, 0.05)
            cfg["primer"], cfg["Lt"] = primer, P["Lt"]
        elif system == "No eléctrico":
            txt = st.text_area("Tiempos nominales por número (ms)",
                               "0, 200, 400, 600, 1000, 1400, 1800, 2400, 3000, 3800, 4600, 5500, 6400, 7400, 8500, 9600",
                               help="Serie de periodo largo referencial: reemplácela por la tabla del fabricante.")
            try:
                cfg["table"] = [float(t) for t in txt.replace(";", ",").split(",") if t.strip()]
            except ValueError:
                st.error("La tabla debe contener números separados por comas.")
                cfg["table"] = [0, 200, 400]
        else:
            cfg["t0"] = st.number_input("Tiempo del primer taladro (ms)", 0.0, 1000.0, 0.0)
            cfg["dt_ms"] = st.number_input("Intervalo entre retardos (ms)", 1.0, 1000.0, 50.0, 1.0)
    if st.button("Calcular secuencia óptima", type="primary"):
        opt_s = dict(classic=classic, qmax=qmax, nmax=int(nmax), cut_single=cut_single, corona_sim=corona_sim,
                     method=method, time_limit=tl, link=link)
        if H.kg.max() > qmax:
            st.error(f"Un solo taladro carga {H.kg.max():.2f} kg > Q máx ({qmax:.1f} kg). Aumente Q máx.")
        else:
            with st.spinner("Optimizando la secuencia…"):
                seq, info = sq.run_sequencing(H, design.relief, opt_s, system, cfg)
            ss.seq, ss.seq_sig, ss.seq_info, ss.seq_qmax = seq, sig, info, qmax
            st.rerun()
    if ss.seq is not None:
        seq, info = ss.seq, ss.seq_info
        c = st.columns(5)
        c[0].metric("Retardos", info["n_periods"])
        c[1].metric("Heurística", f"{info['heur_periods']} retardos")
        c[2].metric("Duración", f"{seq.t_ms.max()/1000:.2f} s")
        c[3].metric("Carga máx. por retardo", f"{seq.groupby('periodo').kg.sum().max():.1f} kg")
        c[4].metric("Radio de alivio usado", f"{info['link']:.2f}·B")
        st.caption(f"Método: {info['method']}" + (f" · {info.get('milp_status', '')}" if 'milp_status' in info else ""))
        g1, g2 = st.columns([1.1, 1])
        g1.plotly_chart(viz.sequence_animation(design, seq), width="stretch")
        g2.plotly_chart(viz.front_2d(design, "periodo", False, False, seq, True, height=640), width="stretch")
        st.plotly_chart(viz.charge_per_delay(seq, ss.seq_qmax), width="stretch")
        cols = ["orden", "id", "zona", "periodo", "t_ms", "predecesor", "kg", "n_cart", "x", "y"]
        if "mecha_m" in seq:
            cols.append("mecha_m")
            st.info("Con carmex, el orden de chispeo debe seguir la columna 'orden'. La mecha tiene una dispersión "
                    "de combustión de alrededor de ±10 %, así que el escalonamiento por longitud sólo es referencial.")
        st.dataframe(seq[cols].round(3), hide_index=True, width="stretch")
        st.download_button("Descargar secuencia (CSV)", seq[cols].to_csv(index=False).encode("utf-8"),
                           "secuencia_voladura.csv", "text/csv")

# =============================================================================
# TAB 6 — RESUMEN Y EXPORTACIÓN
# =============================================================================
with tabs[5]:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Indicadores del disparo")
        st.dataframe(pd.DataFrame({"Indicador": list(summary), "Valor": [round(float(v), 3) for v in summary.values()]}),
                     hide_index=True, width="stretch")
    with c2:
        st.subheader("Carguío por zona")
        zt = H.groupby("zona", sort=False).agg(Taladros=("id", "count"), Cartuchos=("n_cart", "sum"),
                                              Explosivo=("explosivo", "first"), kg=("kg", "sum")).reindex(
            [zz for zz in ZONES if zz in set(H.zona)])
        st.dataframe(zt.round(2), width="stretch")
        pie = go.Figure(go.Pie(labels=zt.index, values=zt.kg, marker=dict(colors=[ZONE_COLORS[z] for z in zt.index]),
                               hole=0.45, sort=False))
        pie.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(pie, width="stretch")
    st.subheader("Tabla de taladros")
    cols = ["id", "zona", "x", "y", "L", "Lt", "Lc", "lookout", "B", "S", "explosivo", "n_cart", "kg"]
    st.dataframe(H[cols].round(3), hide_index=True, width="stretch")
    params = dict(seccion=dict(tipo=kind, ancho=A, alto=section.height, radio=section.radius, area=section.area),
                  roca=rock.__dict__, perforacion=drill.__dict__, explosivo_principal=exp_main.__dict__,
                  explosivo_corona=exp_cor.__dict__, opciones=opts.__dict__,
                  resultados={k: float(v) for k, v in summary.items()})
    d1c, d2c, d3c = st.columns(3)
    d1c.download_button("Taladros (CSV)", H[cols].to_csv(index=False).encode("utf-8"), "malla_taladros.csv", "text/csv")
    d2c.download_button("Memoria de cálculo (CSV)", design.log.df().to_csv(index=False).encode("utf-8"),
                        "memoria_calculo_holmberg.csv", "text/csv")
    d3c.download_button("Parámetros (JSON)", json.dumps(params, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                        "parametros_diseno.json", "application/json")
