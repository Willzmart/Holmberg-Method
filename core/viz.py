"""Figuras Plotly: frente 2D, vista 3D con carguío, perfil de carga y secuencia."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .holmberg import CONTOUR_ZONES, ZONE_COLORS, ZONES

STEM = "#9c8f7a"
FUSE = "#f2c200"
PRIMER = "#ff2d2d"
ROCK_FILL = "rgba(160,160,160,0.18)"


def _section_trace(section, name="Sección"):
    x, y = section.polygon.exterior.xy
    return go.Scatter(x=list(x), y=list(y), mode="lines", fill="toself", fillcolor=ROCK_FILL,
                      line=dict(color="#333", width=2.5), name=name, hoverinfo="skip")


# ----------------------------------------------------------------------------
# Frente 2D
# ----------------------------------------------------------------------------
def front_2d(design, label="id", show_openings=True, show_dims=True, seq: pd.DataFrame | None = None,
             show_vectors=False, height=720):
    s, h, rel = design.section, design.holes, design.relief
    fig = go.Figure()
    fig.add_trace(_section_trace(s))
    if show_openings:
        for name, poly in design.openings:
            ox, oy = poly.exterior.xy
            fig.add_trace(go.Scatter(x=list(ox), y=list(oy), mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=ZONE_COLORS.get(name, "#999"), width=1, dash="dot")))
    # alivios
    r_pt = design.p["d2"] / 2
    for _, r in rel.iterrows():
        fig.add_shape(type="circle", x0=r.x - r_pt, x1=r.x + r_pt, y0=r.y - r_pt, y1=r.y + r_pt,
                      line=dict(color="#222", width=1.5), fillcolor="white")
    fig.add_trace(go.Scatter(x=rel.x, y=rel.y, mode="markers", name=f"Alivio Ø{design.drill.d2_mm:.0f} mm",
                             marker=dict(size=12, color="white", line=dict(color="#222", width=2)),
                             hovertemplate="Alivio<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>"))
    data = h.copy()
    if seq is not None:
        data = data.merge(seq[["id", "periodo", "t_ms", "predecesor"]], on="id", how="left")
    if show_vectors and seq is not None:
        pos = data.set_index("id")[["x", "y"]]
        for _, r in data.iterrows():
            if r.predecesor and r.predecesor in pos.index:
                px, py = pos.loc[r.predecesor]
            else:
                px, py = design.p["cut_center"]
            fig.add_annotation(x=r.x, y=r.y, ax=px, ay=py, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="rgba(0,0,0,0.35)")
    for z in ZONES:
        g = data[data.zona == z]
        if g.empty:
            continue
        if label == "id":
            txt = g.id.astype(str)
        elif label == "periodo" and "periodo" in g:
            txt = g.periodo.astype(int).astype(str)
        elif label == "kg":
            txt = g.kg.round(2).astype(str)
        else:
            txt = None
        hov = ("N° %{customdata[0]}<br>" + z + "<br>x=%{x:.2f} m, y=%{y:.2f} m<br>B=%{customdata[1]:.2f} m"
               "<br>Carga=%{customdata[2]:.2f} kg (%{customdata[3]} cart.)")
        cd = np.c_[g.id, g.B.fillna(0), g.kg, g.n_cart]
        if "periodo" in g:
            hov += "<br>Retardo N° %{customdata[4]} — %{customdata[5]:.0f} ms"
            cd = np.c_[cd, g.periodo, g.t_ms]
        fig.add_trace(go.Scatter(
            x=g.x, y=g.y, mode="markers+text" if txt is not None else "markers", text=txt,
            textposition="top center", textfont=dict(size=10, color="#111"), name=f"{z} ({len(g)})",
            marker=dict(size=11, color=ZONE_COLORS[z], line=dict(color="#111", width=0.8)),
            customdata=cd, hovertemplate=hov + "<extra></extra>"))
    if show_dims:
        b = s.polygon.bounds
        fig.add_annotation(x=0, y=b[1] - 0.25, text=f"A = {s.width:.2f} m", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=b[2] + 0.35, y=s.height / 2, text=f"H = {s.height:.2f} m", showarrow=False,
                           textangle=-90, font=dict(size=12))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="Altura (m)", zeroline=False)
    fig.update_xaxes(title="Ancho (m)", zeroline=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      legend=dict(orientation="v", x=1.02, y=1), title="Vista frontal del frente — malla de perforación")
    return fig


# ----------------------------------------------------------------------------
# Vista 3D
# ----------------------------------------------------------------------------
def _face_mesh(section, y0, color, opacity, name):
    pts = np.asarray(section.polygon.exterior.coords)[:-1]
    cx, cz = section.polygon.centroid.x, section.polygon.centroid.y
    X = np.r_[cx, pts[:, 0]]
    Z = np.r_[cz, pts[:, 1]]
    Y = np.full_like(X, y0)
    n = len(pts)
    i = np.zeros(n, dtype=int)
    j = np.arange(1, n + 1)
    k = np.r_[np.arange(2, n + 1), 1]
    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=color, opacity=opacity, name=name,
                     hoverinfo="skip", showlegend=True, flatshading=True)


def _shell(section, y0, y1, opacity=0.12):
    pts = np.asarray(section.polygon.exterior.coords)
    n = len(pts)
    X = np.r_[pts[:, 0], pts[:, 0]]
    Z = np.r_[pts[:, 1], pts[:, 1]]
    Y = np.r_[np.full(n, y0), np.full(n, y1)]
    I, J, K = [], [], []
    for a in range(n - 1):
        I += [a, a + 1]
        J += [a + 1, n + a + 1]
        K += [n + a, n + a]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color="#8a8a8a", opacity=opacity, name="Túnel excavado",
                     hoverinfo="skip", showlegend=True)


def _ring_line(section, y, name, dash=None, color="#444", show=True):
    x, z = section.polygon.exterior.xy
    return go.Scatter3d(x=list(x), y=[y] * len(x), z=list(z), mode="lines", name=name, showlegend=show,
                        line=dict(color=color, width=4, dash=dash), hoverinfo="skip")


def view_3d(design, system="Carmex (mecha de seguridad + fulminante)", primer="Fondo", show_shell=True,
            show_init=True, show_stem=True, zone_filter=None, seq: pd.DataFrame | None = None, height=760):
    s, h, P = design.section, design.holes, design.p
    Hp = P["Hp"]
    adv = Hp * design.opts.blast_eff
    gamma = math.radians(design.opts.gamma_deg)
    fig = go.Figure()
    fig.add_trace(_face_mesh(s, 0.0, "#c9b79c", 0.35, "Frente (cara libre)"))
    fig.add_trace(_ring_line(s, 0.0, "Contorno del frente"))
    fig.add_trace(_ring_line(s, adv, f"Avance esperado ({adv:.2f} m)", dash="dash", color="#1f77b4"))
    if show_shell:
        fig.add_trace(_shell(s, -2.0, 0.0, opacity=0.07))
    data = h if zone_filter is None else h[h.zona.isin(zone_filter)]
    if seq is not None:
        data = data.merge(seq[["id", "periodo", "t_ms"] + (["mecha_m"] if "mecha_m" in seq else [])],
                          on="id", how="left")

    def axis(r):
        c = np.array([r.x, 0.0, r.y])
        if r.zona in CONTOUR_ZONES:
            u = np.array([r.nx * math.sin(gamma), math.cos(gamma), r.ny * math.sin(gamma)])
        else:
            u = np.array([0.0, 1.0, 0.0])
        return c, u / np.linalg.norm(u)

    stem_xyz, fuse_xyz, prim, coll = [], [], [], []
    charge = {z: [] for z in ZONES}
    for _, r in data.iterrows():
        c, u = axis(r)
        L, Lt = r.L, r.Lt
        a, b = c, c + u * Lt
        stem_xyz += [a, b, [None] * 3]
        charge[r.zona] += [c + u * Lt, c + u * L, [None] * 3]
        sp = L - 0.12 if primer == "Fondo" else Lt + 0.12
        p_pt = c + u * sp
        prim.append(p_pt)
        off = np.array([0.035, 0.0, 0.035])
        tail = 0.35
        if system.startswith("Carmex") and "mecha_m" in data and not pd.isna(r.get("mecha_m", np.nan)):
            tail = max(0.2, float(r.mecha_m) - sp)  # mecha que sobresale del collar
        fuse_xyz += [p_pt + off, c + off, c + off - np.array([0, tail, 0]), [None] * 3]
        coll.append(c)
    for z in ZONES:
        if charge[z]:
            arr = np.array(charge[z], dtype=object)
            fig.add_trace(go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="lines",
                                       line=dict(color=ZONE_COLORS[z], width=9), name=f"Carga — {z}",
                                       legendgroup=z, hoverinfo="skip"))
    if show_stem and stem_xyz:
        arr = np.array(stem_xyz, dtype=object)
        fig.add_trace(go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="lines",
                                   line=dict(color=STEM, width=9), name=f"Taco ({P['Lt']:.2f} m)", hoverinfo="skip"))
    if show_init and fuse_xyz:
        arr = np.array(fuse_xyz, dtype=object)
        lbl = {"Carmex (mecha de seguridad + fulminante)": "Carmex (mecha de seguridad)",
               "No eléctrico": "Tubo de choque (no eléctrico)", "Electrónico": "Cable del detonador electrónico"}
        fig.add_trace(go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="lines",
                                   line=dict(color=FUSE, width=3), name=lbl.get(system, "Iniciador"), hoverinfo="skip"))
        pr = np.array(prim)
        fig.add_trace(go.Scatter3d(x=pr[:, 0], y=pr[:, 1], z=pr[:, 2], mode="markers",
                                   marker=dict(size=4.5, color=PRIMER, symbol="diamond"),
                                   name=f"Cebo + fulminante ({primer.lower()})", hoverinfo="skip"))
    co = np.array(coll)
    if len(co):
        txt = [f"N° {r.id} — {r.zona}<br>L={r.L:.2f} m · taco {r.Lt:.2f} m · carga {r.Lc:.2f} m"
               f"<br>{r.n_cart} cart. {r.explosivo} = {r.kg:.2f} kg"
               + (f"<br>Retardo {int(r.periodo)} — {r.t_ms:.0f} ms" if "periodo" in data and not pd.isna(r.get('periodo')) else "")
               + (f"<br>Mecha: {r.mecha_m:.2f} m" if "mecha_m" in data and not pd.isna(r.get('mecha_m', np.nan)) else "")
               for _, r in data.iterrows()]
        fig.add_trace(go.Scatter3d(x=co[:, 0], y=co[:, 1], z=co[:, 2], mode="markers",
                                   marker=dict(size=3, color="#111"), name="Boca de taladro",
                                   text=txt, hovertemplate="%{text}<extra></extra>"))
    rel = design.relief
    rx, ry, rz = [], [], []
    for _, r in rel.iterrows():
        rx += [r.x, r.x, None]
        ry += [0, Hp + 0.1, None]
        rz += [r.y, r.y, None]
    fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(color="#555", width=14),
                               name=f"Alivio Ø{design.drill.d2_mm:.0f} mm (vacío)", hoverinfo="skip"))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=40, b=0), title="Vista 3D del frente: taladros, taco, carga e iniciación",
        scene=dict(xaxis_title="Ancho X (m)", yaxis_title="Profundidad Y (m)", zaxis_title="Altura Z (m)",
                   aspectmode="data", camera=dict(eye=dict(x=-1.25, y=-2.1, z=0.75), up=dict(x=0, y=0, z=1))),
        legend=dict(orientation="h", x=0.0, y=-0.02, yanchor="top", font=dict(size=11)))
    return fig


# ----------------------------------------------------------------------------
# Perfil de carga por zona
# ----------------------------------------------------------------------------
def hole_profiles(design, system, primer):
    h = design.holes
    zones = [z for z in ZONES if z in set(h.zona)]
    fig = go.Figure()
    for k, z in enumerate(zones):
        r = h[h.zona == z].iloc[0]
        y = -k
        L, Lt = r.L, r.Lt
        fig.add_shape(type="rect", x0=0, x1=L, y0=y - 0.18, y1=y + 0.18, line=dict(color="#333"), fillcolor="#f4f4f4",
                      layer="below")
        fig.add_shape(type="rect", x0=0, x1=Lt, y0=y - 0.14, y1=y + 0.14, line=dict(width=0), fillcolor=STEM,
                      layer="below")
        n = int(r.n_cart)
        lc = (L - Lt) / max(n, 1)
        for c in range(n):
            x0 = L - (c + 1) * lc
            fig.add_shape(type="rect", x0=x0 + 0.01, x1=x0 + lc - 0.01, y0=y - 0.12, y1=y + 0.12,
                          line=dict(width=0.5, color="#fff"), fillcolor=ZONE_COLORS[z], layer="below")
        px = L - lc / 2 if primer == "Fondo" else Lt + lc / 2
        fig.add_trace(go.Scatter(x=[px], y=[y], mode="markers", marker=dict(symbol="diamond", size=11, color=PRIMER,
                                 line=dict(color="#000", width=1)), showlegend=(k == 0), name="Cebo + fulminante"))
        fig.add_trace(go.Scatter(x=[px, -0.4], y=[y + 0.05, y + 0.05], mode="lines", line=dict(color=FUSE, width=3),
                                 showlegend=(k == 0), name="Carmex / tubo de choque" if not system.startswith("Elec") else "Cable"))
        fig.add_annotation(x=-0.5, y=y, text=f"<b>{z}</b>", xanchor="right", showarrow=False, font=dict(size=11))
        fig.add_annotation(x=L + 0.08, y=y, xanchor="left", showarrow=False, font=dict(size=10),
                           text=f"{n} cart. · {r.kg:.2f} kg · q={r.q_teo:.2f} kg/m")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=12, color=STEM),
                             name="Taco"))
    fig.update_xaxes(range=[-2.3, h.L.max() + 2.0], title="Distancia desde la boca del taladro (m)", zeroline=False)
    fig.update_yaxes(visible=False, range=[-len(zones) + 0.4, 0.6])
    fig.update_layout(height=70 * len(zones) + 160, margin=dict(l=10, r=10, t=40, b=60), plot_bgcolor="white",
                      title="Columna de carga por zona (boca → fondo)",
                      legend=dict(orientation="h", y=-0.12, x=0.3))
    return fig


# ----------------------------------------------------------------------------
# Secuencia
# ----------------------------------------------------------------------------
def sequence_animation(design, seq, height=700):
    s = design.section
    per = sorted(seq.periodo.unique())
    x, y = s.polygon.exterior.xy
    base = [go.Scatter(x=list(x), y=list(y), mode="lines", fill="toself", fillcolor=ROCK_FILL,
                       line=dict(color="#333", width=2.5), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=design.relief.x, y=design.relief.y, mode="markers", showlegend=False,
                       marker=dict(size=12, color="white", line=dict(color="#222", width=2)), hoverinfo="skip")]

    def state(p):
        fired = seq.periodo < p
        now = seq.periodo == p
        wait = seq.periodo > p
        return [
            go.Scatter(x=seq.x[wait], y=seq.y[wait], mode="markers+text", text=seq.periodo[wait].astype(str),
                       textposition="top center", marker=dict(size=11, color="#d9d9d9", line=dict(color="#777", width=1)),
                       name="Por salir", hoverinfo="skip"),
            go.Scatter(x=seq.x[fired], y=seq.y[fired], mode="markers", marker=dict(size=13, color="#444", symbol="x"),
                       name="Detonado", hoverinfo="skip"),
            go.Scatter(x=seq.x[now], y=seq.y[now], mode="markers+text", text=seq.periodo[now].astype(str),
                       textposition="top center", marker=dict(size=18, color="#ff4b00", line=dict(color="#000", width=1)),
                       name="Detona ahora", hoverinfo="skip"),
        ]

    fig = go.Figure(data=base + state(per[0]),
                    frames=[go.Frame(data=base + state(p), name=str(p),
                                     layout=go.Layout(title=f"Retardo N° {p} — t = {seq[seq.periodo == p].t_ms.iloc[0]:.0f} ms"))
                            for p in per])
    fig.update_layout(
        height=height, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10),
        title=f"Retardo N° {per[0]}",
        updatemenus=[dict(type="buttons", x=0.0, y=-0.06, direction="left", buttons=[
            dict(label="▶ Reproducir", method="animate",
                 args=[None, dict(frame=dict(duration=550, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pausa", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, y=-0.02, currentvalue=dict(prefix="Retardo N° "),
                      steps=[dict(method="animate", label=str(p),
                                  args=[[str(p)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for p in per])])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def charge_per_delay(seq, qmax):
    g = seq.groupby("periodo").agg(kg=("kg", "sum"), n=("id", "count"), t=("t_ms", "first")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g.periodo, y=g.kg, text=g.n.astype(str) + " tal.", textposition="outside",
                         marker_color="#1f77b4", name="Carga por retardo",
                         hovertemplate="Retardo %{x}<br>%{y:.2f} kg<extra></extra>"))
    fig.add_hline(y=qmax, line=dict(color="#d62728", dash="dash"), annotation_text=f"Q máx por retardo = {qmax:.1f} kg")
    fig.update_layout(height=360, xaxis_title="N° de retardo", yaxis_title="kg por retardo", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
