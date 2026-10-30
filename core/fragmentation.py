"""Predicción de fragmentación (P80) para la zona de producción.

Ruta A: Kuz-Ram (Kuznetsov + Cunningham + Rosin-Rammler).
Ruta B: modelo ML entrenado por el usuario + metaheurística (Evolución Diferencial).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar

LN5_LN2 = math.log(5) / math.log(2)  # (x/X50)^n para 80 % pasante = ln(5)/0.693


# ----------------------------------------------------------------------------
# Factor de roca A (Lilly / Cunningham 1987)
# ----------------------------------------------------------------------------
RMD_OPTS = {"Pulverulento / friable (10)": 10, "Fracturado verticalmente (usa JF)": None, "Masivo (50)": 50}
JPS_OPTS = {"< 0.1 m (10)": 10, "0.1 m – sobretamaño (20)": 20, "> sobretamaño (50)": 50}
JPA_OPTS = {"Buza hacia afuera del frente (20)": 20, "Rumbo perpendicular al frente (30)": 30,
            "Buza hacia el frente (40)": 40}


def rock_factor(rmd_key, jps_key, jpa_key, rho_t_m3, E_gpa, ucs_mpa):
    rmd = RMD_OPTS[rmd_key]
    jf = JPS_OPTS[jps_key] + JPA_OPTS[jpa_key]
    if rmd is None:
        rmd = jf
    rdi = 25 * rho_t_m3 - 50
    hf = E_gpa / 3 if E_gpa < 50 else ucs_mpa / 5
    A = 0.06 * (rmd + jf + rdi + hf)
    return A, dict(RMD=rmd, JF=jf, RDI=rdi, HF=hf)


# ----------------------------------------------------------------------------
# Kuz-Ram
# ----------------------------------------------------------------------------
@dataclass
class KuzRamInput:
    A: float          # factor de roca
    q: float          # densidad lineal de carga en producción (kg/m)
    Lc: float         # longitud de carga (m)
    H: float          # longitud de perforación efectiva H' (m)
    d_mm: float       # diámetro de perforación (mm)
    W: float          # desviación estándar de perforación (m)
    rws_pct: float    # RWS (% respecto al ANFO = 100)
    SB: float = 1.25


def kuzram(B: float, k: KuzRamInput, S: float | None = None) -> dict:
    S = S if S else k.SB * B
    Q = k.q * k.Lc                          # kg por taladro
    V = B * S * k.H                         # m³ por taladro
    K = Q / V                               # factor de carga (kg/m³)
    X50 = k.A * K ** -0.8 * Q ** (1 / 6) * (115 / k.rws_pct) ** (19 / 30)   # cm (Kuznetsov)
    n = (2.2 - 14 * B / k.d_mm) * math.sqrt((1 + S / B) / 2) * (1 - k.W / B) * (k.Lc / k.H)  # Cunningham
    n = max(n, 0.3)
    P80 = X50 * (LN5_LN2) ** (1 / n)
    return dict(B=B, S=S, Q=Q, V=V, K=K, X50=X50, n=n, P80=P80)


def rosin_rammler_curve(X50, n, xmax=None, npts=200):
    xmax = xmax or X50 * 4
    x = np.linspace(0.01, xmax, npts)
    R = 1 - np.exp(-0.693 * (x / X50) ** n)
    return x, R * 100


def optimize_burden_kuzram(k: KuzRamInput, P80_target: float, B_min: float, B_max: float):
    """Busca el burden que hace P80(B) = objetivo (búsqueda acotada)."""
    f = lambda B: (kuzram(B, k)["P80"] - P80_target) ** 2
    res = minimize_scalar(f, bounds=(B_min, B_max), method="bounded", options=dict(xatol=1e-4))
    r = kuzram(res.x, k)
    r["converged"] = abs(r["P80"] - P80_target) / P80_target < 0.02
    hist = []
    Bs = np.linspace(B_min, B_max, 60)
    for B in Bs:
        hist.append(kuzram(B, k))
    return r, pd.DataFrame(hist)


# ----------------------------------------------------------------------------
# Ruta ML
# ----------------------------------------------------------------------------
ML_VARIABLES = {
    "B": "Burden (m)", "S": "Espaciamiento (m)", "SB": "Relación S/B", "d_mm": "Diámetro (mm)",
    "q": "Densidad de carga (kg/m)", "Q": "Carga por taladro (kg)", "K": "Factor de carga (kg/m³)",
    "H": "Longitud de perforación (m)", "Lc": "Longitud de carga (m)", "Lt": "Taco (m)",
    "RMR": "RMR", "GSI": "GSI", "RQD": "RQD (%)", "UCS": "σci (MPa)", "rho_r": "Densidad roca (t/m³)",
    "VOD": "VOD (m/s)", "rho_e": "Densidad explosivo (g/cm³)", "RWS": "RWS (%)", "A_sec": "Área sección (m²)",
}


def ml_features(B, SB, base: dict) -> dict:
    S = SB * B
    Q = base["q"] * base["Lc"]
    v = dict(base)
    v.update(B=B, S=S, SB=SB, Q=Q, K=Q / (B * S * base["H"]))
    return v


class DemoSurrogate:
    """Sustituto de DEMOSTRACIÓN (Kuz-Ram con término no lineal). No usar para diseño real."""
    feature_names_in_ = np.array(["B", "S", "d_mm", "q", "Lc", "H", "RMR", "RWS"])

    def __init__(self, A=7.0):
        self.A = A

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = []
        for _, r in X.iterrows():
            k = KuzRamInput(A=self.A * (1 + 0.004 * (60 - r.RMR)), q=r.q, Lc=r.Lc, H=r.H, d_mm=r.d_mm,
                            W=0.04, rws_pct=r.RWS, SB=r.S / r.B)
            p = kuzram(r.B, k, r.S)["P80"]
            out.append(p * (1 + 0.08 * math.sin(3 * r.B)))
        return np.array(out)


def model_feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "steps"):
        names = getattr(model.steps[-1][1], "feature_names_in_", None)
    return list(names) if names is not None else None


def ml_predict(model, features: list, values: dict, out_scale: float, target_kind: str, n_unif: float):
    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    y = float(np.ravel(model.predict(row))[0]) * out_scale  # → cm
    if target_kind == "X50":
        return y * LN5_LN2 ** (1 / n_unif), y
    return y, None


def optimize_ml(model, features, base, P80_target, B_bounds, SB_bounds, out_scale, target_kind,
                n_unif, lam=0.15, seed=0):
    hist = []

    def obj(x):
        B, SB = x
        v = ml_features(B, SB, base)
        p80, _ = ml_predict(model, features, v, out_scale, target_kind, n_unif)
        drill_index = 1.0 / (B * SB * B)          # perforación por m³ (∝ costo)
        ref = 1.0 / (B_bounds[1] ** 2 * SB_bounds[1])
        f = abs(p80 - P80_target) / P80_target + lam * (drill_index / ref - 1) * 0.1
        hist.append(dict(B=B, SB=SB, P80=p80, objetivo=f))
        return f

    res = differential_evolution(obj, [B_bounds, SB_bounds], seed=seed, maxiter=60, popsize=15,
                                 tol=1e-6, polish=True)
    B, SB = res.x
    v = ml_features(B, SB, base)
    p80, _ = ml_predict(model, features, v, out_scale, target_kind, n_unif)
    return dict(B=B, SB=SB, S=B * SB, P80=p80, fun=res.fun, nfev=res.nfev), pd.DataFrame(hist)
