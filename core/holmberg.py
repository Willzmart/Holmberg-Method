"""Motor de cálculo del método Holmberg (cuele de taladros paralelos) y
generación geométrica de la malla para cualquier sección de galería.

Ecuaciones según: Carpio Chávez, A. (2025). Diseño de mallas de perforación y
voladura de labores de avance (secciones 3.2.1 a 3.2.10). El número de cada
ecuación del libro se indica entre paréntesis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient

from .sections import Section

# ----------------------------------------------------------------------------
# Catálogo de explosivos (datos del Ejemplo N°06 del libro, Emulnor - FAMESA)
# ----------------------------------------------------------------------------
CARTRIDGES = {  # nombre: (diámetro m, longitud m)
    '1 1/4" x 12"': (0.03175, 0.3048),
    '1" x 8"': (0.0254, 0.2032),
    '1" x 12"': (0.0254, 0.3048),
}
EXPLOSIVES = {
    "Emulnor 500": dict(vod=4000, rho=0.90, rws=0.70,
                        w={'1 1/4" x 12"': 0.200, '1" x 8"': 0.084, '1" x 12"': 0.128}),
    "Emulnor 1000": dict(vod=5800, rho=1.13, rws=0.87,
                         w={'1 1/4" x 12"': 0.250, '1" x 8"': 0.108, '1" x 12"': 0.166}),
    "Emulnor 3000": dict(vod=5700, rho=1.14, rws=1.02,
                         w={'1 1/4" x 12"': 0.265, '1" x 8"': 0.109, '1" x 12"': 0.173}),
    "Emulnor 5000": dict(vod=5500, rho=1.16, rws=1.12,
                         w={'1 1/4" x 12"': 0.265, '1" x 8"': 0.115, '1" x 12"': 0.176}),
}

ZONES = ["Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3", "Cuadrante 4",
         "Ayuda de hastiales", "Ayuda de corona", "Ayuda de arrastre",
         "Hastiales", "Corona", "Arrastre"]
ZONE_COLORS = {
    "Alivio": "#6b6b6b",
    "Arranque": "#d62728",
    "Cuadrante 1": "#ff7f0e",
    "Cuadrante 2": "#e6b800",
    "Cuadrante 3": "#9467bd",
    "Cuadrante 4": "#c51b8a",
    "Ayuda de hastiales": "#1f77b4",
    "Ayuda de corona": "#17becf",
    "Ayuda de arrastre": "#8c564b",
    "Hastiales": "#e377c2",
    "Corona": "#2ca02c",
    "Arrastre": "#7f7f00",
}
CUT_ZONES = {"Arranque", "Cuadrante 1", "Cuadrante 2", "Cuadrante 3", "Cuadrante 4"}
HELPER_ZONES = {"Ayuda de hastiales", "Ayuda de corona", "Ayuda de arrastre"}
CONTOUR_ZONES = {"Hastiales", "Corona", "Arrastre"}


# ----------------------------------------------------------------------------
# Datos de entrada
# ----------------------------------------------------------------------------
@dataclass
class Rock:
    RMR: float = 50
    GSI: float = 45
    RQD: float = 55
    UCS: float = 75          # MPa
    rho: float = 2.75        # t/m3
    Q: float = 1.95          # Barton
    JSF: float = 0.9


@dataclass
class Drill:
    d1_mm: float = 45        # diámetro de producción
    d2_mm: float = 102       # diámetro de alivio (rimado)
    NB: int = 2              # taladros de alivio
    bar_m: float = 3.35      # longitud de barra
    eff: float = 0.95        # eficiencia de perforación
    F_method: str = "Dato directo"  # "Dato directo" | "Ec. 26" | "Ec. 27"
    F_value: float = 0.04
    alpha_mm_m: float = 10.0  # desviación angular
    e_mm: float = 20.0        # error de emboquille


@dataclass
class Explosive:
    name: str = "Emulnor 1000"
    vod: float = 5800
    rho: float = 1.13        # g/cm3
    rws: float = 0.87        # potencia relativa en peso referida al ANFO (fracción)
    cart: str = '1 1/4" x 12"'
    w_cart: float = 0.25     # kg/cartucho
    L_cart: float = 0.3048   # m


@dataclass
class DesignOptions:
    cut_method: str = "Holmberg"     # "Holmberg" | "Pearse"
    cut_factor: float = 1.7          # 1.5 ó 1.7 (Ec. 28-29)
    n_quadrants: int | None = None   # None = automático (Ah ≤ √H')
    gamma_deg: float = 3.0           # ángulo de contorno
    K_corona: float = 15.0           # S = K·Ø1
    c_hat_rule: str = "Iterativa"    # "Iterativa" | "c + 0.05"
    contour_offset: float = 0.10     # separación del contorno al emboquillar (m)
    cut_height: float | None = None  # altura del centro del cuele (None = libro)
    blast_eff: float = 0.95          # eficiencia de voladura (avance / H')
    B_prod_override: float | None = None  # burden de producción impuesto (Kuz-Ram/ML)
    SB_prod: float = 1.25


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
class Log:
    """Registro de cálculos (fase, parámetro, símbolo, valor, unidad, ecuación)."""

    def __init__(self):
        self.rows = []

    def add(self, fase, par, sym, val, unit="", eq=""):
        self.rows.append(dict(Fase=fase, Parámetro=par, Símbolo=sym,
                              Valor=(round(val, 4) if isinstance(val, (int, float, np.floating)) else val),
                              Unidad=unit, Ecuación=eq))
        return val

    def df(self):
        return pd.DataFrame(self.rows)


def c_hat(c, B, rule):
    if rule == "c + 0.05":
        return c + 0.05
    return c + 0.05 if B >= 1.4 else c + 0.07 / B


def bmax_tajeo(q0, rws, c, f, sb, rule):
    """B_max = 0.9·√(q0·RWS / (Ĉ·f·S/B))  (Ec. 48, 54, 61, 66, 78)."""
    B = 0.9 * math.sqrt(q0 * rws / ((c + 0.05) * f * sb))
    for _ in range(50):
        ch = c_hat(c, B, rule)
        Bn = 0.9 * math.sqrt(q0 * rws / (ch * f * sb))
        if abs(Bn - B) < 1e-6:
            break
        B = Bn
    return B, c_hat(c, B, rule)


def _sample_chain(line: LineString, n_pts: int, include_ends=True):
    if n_pts <= 0:
        return []
    L = line.length
    if include_ends:
        if n_pts == 1:
            ts = [0.5 * L]
        else:
            ts = np.linspace(0, L, n_pts)
    else:
        ts = np.linspace(0, L, n_pts + 2)[1:-1]
    out = []
    for t in ts:
        p = line.interpolate(t)
        p2 = line.interpolate(min(L, t + 0.01))
        p1 = line.interpolate(max(0, t - 0.01))
        tx, ty = p2.x - p1.x, p2.y - p1.y
        nrm = math.hypot(tx, ty) or 1.0
        out.append((p.x, p.y, tx / nrm, ty / nrm))
    return out


def _chains(poly: Polygon, springline: float, floor_tol=0.03):
    """Divide el contorno (antihorario) en cadenas: piso, hastial der., corona, hastial izq."""
    ring = LineString(poly.exterior.coords)
    L = ring.length
    n = max(400, int(L / 0.01))
    ts = np.linspace(0, L, n, endpoint=False)
    pts = [ring.interpolate(t) for t in ts]
    ymin = min(p.y for p in pts)

    def cls(p):
        if p.y <= ymin + floor_tol:
            return "piso"
        if p.y > springline:
            return "corona"
        return "der" if p.x > 0 else "izq"

    labels = [cls(p) for p in pts]
    # rotar para empezar en un cambio de clase
    k0 = next((i for i in range(n) if labels[i] != labels[i - 1]), 0)
    labels = labels[k0:] + labels[:k0]
    pts = pts[k0:] + pts[:k0]
    groups = []
    cur, cur_pts = labels[0], [pts[0]]
    for lab, p in zip(labels[1:], pts[1:]):
        if lab == cur:
            cur_pts.append(p)
        else:
            groups.append((cur, cur_pts + [p]))
            cur, cur_pts = lab, [p]
    groups.append((cur, cur_pts + [pts[0]]))
    chains = {}
    for lab, gp in groups:
        if len(gp) < 2:
            continue
        ls = LineString([(p.x, p.y) for p in gp])
        if lab in chains:  # unir tramos partidos
            if chains[lab].length < ls.length:
                chains[lab] = ls
        else:
            chains[lab] = ls
    return chains


# ----------------------------------------------------------------------------
# Cálculo Holmberg
# ----------------------------------------------------------------------------
@dataclass
class Design:
    section: Section
    rock: Rock
    drill: Drill
    exp_main: Explosive
    exp_cor: Explosive
    opts: DesignOptions
    log: Log = field(default_factory=Log)
    p: dict = field(default_factory=dict)      # parámetros calculados
    holes: pd.DataFrame | None = None
    relief: pd.DataFrame | None = None
    openings: list = field(default_factory=list)  # [(nombre, Polygon)]
    warnings: list = field(default_factory=list)


def compute_parameters(d: Design) -> Design:
    s, rk, dr, ex, ec, o, lg = d.section, d.rock, d.drill, d.exp_main, d.exp_cor, d.opts, d.log
    P = d.p
    A, Ht = s.width, s.height
    d1, d2 = dr.d1_mm / 1000, dr.d2_mm / 1000

    # ---- Fase 0-1: parámetros generales ----
    F0 = "1. Parámetros generales"
    lg.add(F0, "Área de la sección", "S", s.area, "m²")
    lg.add(F0, "Perímetro de la sección", "P", s.perimeter, "m")
    Hp = lg.add(F0, "Longitud de perforación efectiva", "H'", dr.bar_m * dr.eff, "m", "(23)")
    deq = lg.add(F0, "Diámetro equivalente de alivio", "Ø_equiv", d2 * math.sqrt(dr.NB), "m", "(25)")
    Lmax = lg.add(F0, "Longitud máxima alcanzable con el alivio", "L", 0.15 + 34.1 * deq - 39.4 * deq**2, "m", "(24)")
    if Hp > Lmax:
        d.warnings.append(f"H' = {Hp:.2f} m supera la longitud alcanzable L = {Lmax:.2f} m: aumente NB o Ø2.")
    if dr.F_method == "Ec. 26":
        F = dr.alpha_mm_m / 1000 * Hp + dr.e_mm / 1000
        eqF = "(26) F = α·H' + e"
    elif dr.F_method == "Ec. 27":
        F = 0.0013 * Hp**2 + 0.078 * Hp + 0.0014
        eqF = "(27)"
    else:
        F, eqF = dr.F_value, "dato"
    F = lg.add(F0, "Error de perforación", "F", F, "m", eqF)
    ERQD = rk.RQD * rk.JSF
    Kv = lg.add(F0, "Factor de volabilidad (ERQD = RQD·JSF)", "K_v", 1.96 - 0.27 * math.log(max(ERQD, 1e-3)), "", "Kv = 1.96 − 0.27·ln(ERQD)")
    PD = lg.add(F0, "Presión de detonación", "PD", ex.rho * 1000 * 1e-6 * ex.vod**2 / 4, "MPa", "PD = ρe·10⁻⁶·VOD²/4")
    St = 0.08 * rk.UCS
    Std = lg.add(F0, "Resistencia a tracción dinámica", "S_td", 4.5 * 10.197 * St, "MPa", "Std = 4.5·10.197·(0.08·σci)")
    CE = 0.56 * rk.rho * math.tan(math.radians(5 + rk.RMR / 2)) / ((115 - rk.RQD) / 3.3) ** (1 / 3)
    CE = lg.add(F0, "Consumo específico de explosivo", "C.E", CE, "kg/m³", "0.56·ρr·tan(5+RMR/2) / ∛((115−RQD)/3.3)")
    c = lg.add(F0, "Constante sueca de roca", "c", 0.8784 * CE + 0.0052, "kg/m³", "c = 0.8784·C.E + 0.0052")
    Lt = lg.add(F0, "Longitud de taco", "Lt", 10 * d1, "m", "(31)")
    Lc = lg.add(F0, "Longitud de carga", "Lc", Hp - Lt, "m", "(32)")
    VP = 3500 + 1000 * math.log10(max(rk.Q, 1e-3))
    lg.add(F0, "Impedancia explosivo / roca", "n_z", ex.rho * ex.vod / (rk.rho * VP), "", "ρe·VOD/(ρr·Vp), Vp = 3500+1000·log Q")

    # ---- Fase 2: arranque ----
    F2 = "2. Arranque"
    if o.cut_method == "Pearse":
        Bmax0 = Kv * dr.d1_mm / 1000 * math.sqrt(PD / Std)
        eqB = "Pearse: Kv·Ø1/1000·√(PD/Std)"
    else:
        Bmax0 = o.cut_factor * deq
        eqB = f"(28-29) {o.cut_factor}·Ø_equiv"
    Bmax0 = lg.add(F2, "Burden máximo del arranque", "B_max", Bmax0, "m", eqB)
    Bp0 = lg.add(F2, "Burden práctico del arranque", "B_p", Bmax0 - F, "m", "(30)")
    if Bp0 <= deq / 2:
        d.warnings.append("El burden práctico del arranque es menor que el radio equivalente del alivio: revise F o Ø2.")
    q0 = 55 * d1 * (Bmax0 / deq) ** 1.5 * (Bmax0 - deq / 2) * (c / 0.4) / ex.rws
    q0 = lg.add(F2, "Densidad de carga lineal del arranque", "q₀", q0, "kg/m", "(26)")
    q_cart = ex.w_cart / ex.L_cart
    lg.add(F2, "Densidad lineal del cartucho elegido", "q_cart", q_cart, "kg/m", "peso / longitud")
    Ah = lg.add(F2, "Ancho de abertura del arranque", "A_h0", math.sqrt(2) * Bp0, "m", "(33)")

    cut_secs = [dict(name="Arranque", B=Bp0, Bmax=Bmax0, dist=Bp0, Ah=Ah, rot=45.0)]

    # ---- Fase 3: cuadrantes ----
    F3 = "3. Cuadrantes"
    sqrtH = math.sqrt(Hp)
    nmax = o.n_quadrants if o.n_quadrants is not None else 4
    rot = 0.0
    for N in range(1, nmax + 1):
        Bmax = 8.8e-2 * math.sqrt(Ah * q0 * ex.rws / (d1 * c))
        Bmax = min(Bmax, 2 * Ah)  # límite de Holmberg B ≤ 2·Ah
        Bp = Bmax - F
        X = Ah / 2 + Bp
        Ahn = math.sqrt(2) * X
        # criterio automático (Ec. 46): se acepta el cuadrante si A_hN ≤ √H' (tolerancia 5 %)
        if o.n_quadrants is None and N > 1 and Ahn > 1.05 * sqrtH:
            break
        if Ahn > 0.8 * min(A, Ht):
            d.warnings.append(f"El cuadrante {N} excede la sección; se detiene en {N-1}.")
            break
        lg.add(F3, f"Burden máximo cuadrante {N}", f"B_max{N}", Bmax, "m", "(34/38/42)")
        lg.add(F3, f"Burden práctico cuadrante {N}", f"B_p{N}", Bp, "m", "(35/39/43)")
        lg.add(F3, f"Distancia centro-esquina cuadrante {N}", f"X{N}", X, "m", "(36/40/44)")
        lg.add(F3, f"Ancho de abertura cuadrante {N}", f"A_h{N}", Ahn, "m", "(37/41/45)")
        cut_secs.append(dict(name=f"Cuadrante {N}", B=Bp, Bmax=Bmax, dist=X, Ah=Ahn, rot=rot))
        rot = 45.0 - rot
        Ah = Ahn
    lg.add(F3, "Verificación A_h último ≤ √H'", "√H'", sqrtH, "m",
           "cumple" if Ah <= sqrtH * 1.02 else "A_h > √H' (se acepta como límite)")

    # ---- Fases 4-7: tajeo ----
    gs = math.sin(math.radians(o.gamma_deg))
    zone_params = {}
    rows = [("4. Ayuda de hastiales", "Ayuda de hastiales", 1.45, 1.25),
            ("5. Ayuda de corona", "Ayuda de corona", 1.20, 1.25),
            ("6. Ayuda de arrastre", "Ayuda de arrastre", 1.45, 1.25),
            ("7. Hastiales", "Hastiales", 1.45, 1.25),
            ("9. Arrastre", "Arrastre", 1.45, 1.00)]
    for fase, z, f, sb in rows:
        Bm, ch = bmax_tajeo(q0, ex.rws, c, f, sb, o.c_hat_rule)
        lg.add(fase, f"Constante corregida {z.lower()}", "Ĉ", ch, "kg/m³", "c+0.05 (B≥1.4) / c+0.07/B (B<1.4)")
        lg.add(fase, f"Burden máximo {z.lower()}", "B_max", Bm, "m", f"0.9·√(q₀·RWS/(Ĉ·{f}·{sb}))")
        zone_params[z] = dict(Bmax=Bm, f=f, sb=sb)

    Ahl = Ah
    # referencias geométricas del libro
    fa = "4. Ayuda de hastiales"
    lg.add(fa, "Burden práctico (libro)", "B_p", (A - Ahl) / 4, "m", "(49)")
    lg.add(fa, "N° taladros (libro)", "N°", (A - Ahl) / 1.25 + 2, "", "(50)")
    fa = "5. Ayuda de corona"
    Hd_dn = lg.add(fa, "Altura disponible hacia abajo", "H_disp(↓)", 0.64 * (Ht - Ahl), "m", "(53)")
    lg.add(fa, "Burden práctico (libro)", "B_p", 0.51 * Hd_dn, "m", "(55)")
    lg.add(fa, "Longitud de segmento", "L_segm", 0.6533 * A, "m", "(56)")
    fa = "6. Ayuda de arrastre"
    Hd_up = lg.add(fa, "Altura disponible hacia arriba", "H_disp(↑)", 0.36 * (Ht - Ahl), "m", "(60)")
    lg.add(fa, "Burden práctico (libro)", "B_p", Hd_up / 2, "m", "(62)")
    lg.add(fa, "Longitud de segmento", "L_segm", 0.72 * A, "m", "(63)")
    fa = "7. Hastiales"
    Bp_h = zone_params["Hastiales"]["Bmax"] - Hp * gs - F
    lg.add(fa, "Burden práctico (libro)", "B_pract", 0.2755 * A / 2, "m", "(67)")
    lg.add(fa, "Burden práctico de contorno", "B_p", Bp_h, "m", "B_max − H'·senγ − F")
    lg.add(fa, "Altura disponible en hastiales", "H_disp", s.wall_height, "m", "(68)")

    # ---- Fase 8: corona (smooth blasting) ----
    F8 = "8. Corona"
    S_cor = lg.add(F8, "Espaciamiento de corona", "S", o.K_corona * d1, "m", "(71) S = K·Ø1")
    Bm_cor = lg.add(F8, "Burden máximo de corona", "B_max", S_cor / 0.8, "m", "(72)")
    Bp_cor = lg.add(F8, "Burden práctico de corona", "B_pract", Bm_cor - Hp * gs - F, "m", "(73)")
    lg.add(F8, "N° taladros de corona (libro)", "N°", A / (Bm_cor * 0.8) + 2, "", "(74)")
    q_cor = lg.add(F8, "Densidad de carga de corona", "q_corona", 90 * d1**2, "kg/m", "(75)")

    # ---- Fase 9: arrastre ----
    F9 = "9. Arrastre"
    Bm_arr = zone_params["Arrastre"]["Bmax"]
    Bp_arr = lg.add(F9, "Burden práctico de arrastre", "B_pract", Bm_arr - Hp * gs - F, "m", "(79)")
    N_arr = int((s.floor_width + 2 * Hp * gs) / Bm_arr + 2)
    lg.add(F9, "N° taladros de arrastre", "N°", N_arr, "", "(80)")
    S_arr = lg.add(F9, "Espaciamiento central de arrastre", "S_central", (s.floor_width + 2 * Hp * gs) / (N_arr - 1), "m", "(81)")
    lg.add(F9, "Espaciamiento en esquinas", "S_esq", S_arr - Hp * gs, "m", "S − H'·senγ (Holmberg)")

    # burden de producción usado en la malla
    Bp_limit = zone_params["Ayuda de hastiales"]["Bmax"] - F
    Bp_help = Bp_limit
    if o.B_prod_override:
        if o.B_prod_override > Bp_limit:
            d.warnings.append(f"El burden impuesto ({o.B_prod_override:.2f} m) supera el límite de Holmberg "
                              f"({Bp_limit:.2f} m); se usa el límite.")
            Bp_help = Bp_limit
        else:
            Bp_help = o.B_prod_override
        lg.add("10. Predicción P80", "Burden de producción impuesto (Kuz-Ram / ML)", "B_prod", Bp_help, "m",
               "optimización por P80")
    P["Bp_limit"] = Bp_limit
    P.update(dict(Hp=Hp, deq=deq, Lmax=Lmax, F=F, c=c, CE=CE, Kv=Kv, PD=PD, q0=q0, q_cor=q_cor, Lt=Lt, Lc=Lc,
                  cut_secs=cut_secs, Ah_last=Ahl, zone_params=zone_params, S_cor=S_cor, Bp_cor=Bp_cor,
                  Bp_h=Bp_h, Bm_arr=Bm_arr, Bp_arr=Bp_arr, N_arr=N_arr, S_arr=S_arr, Bp_prod=Bp_help,
                  gamma=o.gamma_deg, gs=gs, d1=d1, d2=d2, q_cart=q_cart))
    return d


# ----------------------------------------------------------------------------
# Generación geométrica de la malla
# ----------------------------------------------------------------------------
def _square(cx, cy, half_diag, rot_deg):
    pts = []
    for k in range(4):
        a = math.radians(rot_deg + 90 * k)
        pts.append((cx + half_diag * math.cos(a), cy + half_diag * math.sin(a)))
    return Polygon(pts)


def build_layout(d: Design) -> Design:
    s, o, P = d.section, d.opts, d.p
    poly = s.polygon
    Hp, gs, F = P["Hp"], P["gs"], P["F"]
    holes = []

    def add(zone, x, y, nx=0.0, ny=0.0, B=None, S=None):
        holes.append(dict(zona=zone, x=float(x), y=float(y), nx=float(nx), ny=float(ny),
                          B=float(B) if B else np.nan, S=float(S) if S else np.nan))

    # ---- centro del cuele ----
    Ahl = P["Ah_last"]
    yc = o.cut_height if o.cut_height else 0.36 * (s.height - Ahl) + Ahl / 2
    yc = float(np.clip(yc, Ahl / 2 + 0.3, s.height - Ahl / 2 - 0.3))
    xc = 0.0
    P["cut_center"] = (xc, yc)

    # ---- alivios ----
    d2 = P["d2"]
    NB = d.drill.NB
    rel = []
    if NB == 1:
        rel = [(xc, yc)]
    else:
        rr = (d2 + 0.02) / (2 * math.sin(math.pi / NB)) if NB > 2 else (d2 + 0.02) / 2
        for k in range(NB):
            a = math.pi / 2 + 2 * math.pi * k / NB if NB > 2 else math.pi * k
            rel.append((xc + rr * math.cos(a), yc + rr * math.sin(a)))
    d.relief = pd.DataFrame(rel, columns=["x", "y"])

    # ---- arranque y cuadrantes ----
    d.openings = []
    for sec in P["cut_secs"]:
        sq = _square(xc, yc, sec["dist"], sec["rot"])
        for (x, y) in list(sq.exterior.coords)[:4]:
            add(sec["name"], x, y, B=sec["B"], S=sec["Ah"])
        d.openings.append((sec["name"], sq))
    opening = d.openings[-1][1]

    # ---- contorno ----
    off = o.contour_offset
    cpoly = orient(poly.buffer(-off, join_style=2), sign=1.0) if off > 0 else poly
    spring_c = min(s.springline, cpoly.bounds[3] - 0.02)
    chains = _chains(cpoly, spring_c)
    # corona (smooth blasting): S = K·Ø1 sobre la longitud real del arco
    if "corona" in chains:
        ch = chains["corona"]
        n = max(2, math.ceil(ch.length / P["S_cor"]) + 1)
        for x, y, tx, ty in _sample_chain(ch, n, True):
            add("Corona", x, y, ty, -tx, B=P["Bp_cor"], S=ch.length / (n - 1))
    # arrastre
    if "piso" in chains:
        ch = chains["piso"]
        n = max(3, P["N_arr"])
        for x, y, tx, ty in _sample_chain(ch, n, True):
            add("Arrastre", x, y, 0.0, -1.0, B=P["Bp_arr"], S=ch.length / (n - 1))
    # hastiales
    S_h = 1.25 * max(P["Bp_h"], 0.3)
    for side in ("der", "izq"):
        if side in chains:
            ch = chains[side]
            n_int = max(0, math.ceil(ch.length / S_h) - 1)
            for x, y, tx, ty in _sample_chain(ch, n_int, False):
                add("Hastiales", x, y, ty, -tx, B=P["Bp_h"], S=ch.length / (n_int + 1))

    # ---- anillos de ayudas (producción) ----
    B_side = max(P["Bp_h"], 0.25)
    B_top = max(P["Bp_cor"], 0.25)
    B_floor = max(P["Bp_arr"], 0.25)
    d_ring = (B_side + B_top) / 2
    ring1 = cpoly.buffer(-d_ring, join_style=2)
    ring1 = ring1.intersection(box(-50, B_floor + off, 50, 50))
    if ring1.geom_type == "MultiPolygon":
        ring1 = max(ring1.geoms, key=lambda g: g.area)
    Bprod = P["Bp_prod"]
    SB = P.get("SB_prod", o.SB_prod)
    S_ring = SB * Bprod
    P["ring_step"] = Bprod
    existing = [(h["x"], h["y"]) for h in holes]
    # fila 1 de ayudas: paralela al contorno, a la distancia de su burden
    if not ring1.is_empty:
        excl = opening.buffer(0.6 * Bprod, join_style=2)
        ls = LineString(orient(ring1, 1.0).exterior.coords)
        L = ls.length
        n = max(4, math.ceil(L / S_ring))
        if n % 2:
            n += 1
        t0 = ls.project(Point(0.0, ring1.bounds[1] - 1))  # inicia en el centro inferior (simetría)
        for i in range(n):
            pt = ls.interpolate((t0 + i * L / n) % L)
            if excl.contains(pt):
                continue
            if min(math.hypot(pt.x - ex, pt.y - ey) for ex, ey in existing) < 0.6 * Bprod:
                continue
            add("_ayuda", pt.x, pt.y, B=Bprod, S=L / n)
            existing.append((pt.x, pt.y))

    # ---- relleno simétrico de máxima distancia (cobertura de burden) ----
    # Cada punto de roca debe quedar a ≤ R_cov de algún taladro; para una malla B × S
    # el punto más alejado está a ½·√(B² + S²) ≈ 0.80·B (S/B = 1.25).
    R_cov = 0.5 * math.hypot(Bprod, S_ring)
    minx, miny, maxx, maxy = poly.bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, 0.04), np.arange(miny, maxy, 0.04))
    G = np.c_[gx.ravel(), gy.ravel()]
    from shapely import contains_xy
    inner = cpoly.buffer(-0.35 * min(B_side, B_top, B_floor))
    mask = contains_xy(inner, G[:, 0], G[:, 1]) & ~contains_xy(opening, G[:, 0], G[:, 1])
    G = G[mask]
    if len(G):
        Hh = np.array(existing + list(map(tuple, d.relief.values)))
        dmin = np.min(np.hypot(G[:, None, 0] - Hh[None, :, 0], G[:, None, 1] - Hh[None, :, 1]), axis=1)
        for _ in range(60):
            k = int(np.argmax(dmin))
            if dmin[k] <= R_cov:
                break
            x, y = G[k]
            pts = [(x, y)] if abs(x - xc) < 0.25 * Bprod else [(x, y), (2 * xc - x, y)]
            if len(pts) == 1:
                pts = [(xc, y)]
            for (px, py) in pts:
                add("_ayuda", px, py, B=Bprod, S=S_ring)
                dmin = np.minimum(dmin, np.hypot(G[:, 0] - px, G[:, 1] - py))

    df = pd.DataFrame(holes)
    # clasificación de ayudas por sector respecto al centro del cuele
    for i in df.index[df.zona == "_ayuda"]:
        dx, dy = df.at[i, "x"] - xc, df.at[i, "y"] - yc
        if dy >= abs(dx) * 0.8:
            df.at[i, "zona"] = "Ayuda de corona"
        elif -dy >= abs(dx) * 0.8:
            df.at[i, "zona"] = "Ayuda de arrastre"
        else:
            df.at[i, "zona"] = "Ayuda de hastiales"
    d.holes = df
    return d


# ----------------------------------------------------------------------------
# Carguío
# ----------------------------------------------------------------------------
def charge_holes(d: Design) -> Design:
    P, o = d.p, d.opts
    df = d.holes.copy()
    Hp, Lt, gs = P["Hp"], P["Lt"], P["gs"]
    look = df.zona.isin(CONTOUR_ZONES)
    df["L"] = Hp
    df["lookout"] = np.where(look, Hp * gs, 0.0)
    df["Lt"] = Lt
    df["Lc"] = Hp - Lt
    exm, exc = d.exp_main, d.exp_cor
    df["explosivo"] = np.where(df.zona == "Corona", f"{exc.name} {exc.cart}", f"{exm.name} {exm.cart}")
    df["q_teo"] = np.where(df.zona == "Corona", P["q_cor"], P["q0"])
    w = np.where(df.zona == "Corona", exc.w_cart, exm.w_cart)
    Lcart = np.where(df.zona == "Corona", exc.L_cart, exm.L_cart)
    df["kg_teo"] = df.q_teo * df.Lc
    df["n_cart"] = np.maximum(1, np.round(df.kg_teo / w)).astype(int)
    df["kg"] = df.n_cart * w
    df["llenado"] = df.n_cart * Lcart / df.Lc
    if (df.llenado > 1.15).any():
        d.warnings.append("Algunos taladros requieren más cartuchos de los que caben sin atacar (>115 % de Lc). "
                          "Considere un cartucho de mayor peso o un explosivo de mayor densidad.")
    # numeración: por zona y ángulo
    order = {z: i for i, z in enumerate(ZONES)}
    xc, yc = P["cut_center"]
    df["ang"] = np.degrees(np.arctan2(df.y - yc, df.x - xc)) % 360
    df["rank"] = df.zona.map(order)
    df = df.sort_values(["rank", "ang"]).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    d.holes = df.drop(columns=["ang"])
    return d


def summarize(d: Design, cost: dict | None = None) -> dict:
    df, P, s = d.holes, d.p, d.section
    adv = P["Hp"] * d.opts.blast_eff
    vol = s.area * adv
    n = len(df)
    kg = df.kg.sum()
    drill_m = (n + d.drill.NB) * P["Hp"]
    out = {
        "N° taladros cargados": n,
        "N° taladros de alivio": d.drill.NB,
        "Avance por disparo (m)": adv,
        "Volumen roto (m³)": vol,
        "Tonelaje roto (t)": vol * d.rock.rho,
        "Explosivo total (kg)": kg,
        "Factor de carga (kg/m³)": kg / vol,
        "Carga por metro de avance (kg/m)": kg / adv,
        "Perforación total (m)": drill_m,
        "Perforación específica (m/m³)": drill_m / vol,
    }
    if cost:
        c_drill = drill_m * cost["drill"]
        c_exp = kg * cost["exp"]
        c_acc = n * cost["acc"]
        tot = c_drill + c_exp + c_acc
        out.update({"Costo perforación (US$)": c_drill, "Costo explosivo (US$)": c_exp,
                    "Costo accesorios (US$)": c_acc, "Costo total por disparo (US$)": tot,
                    "Costo por metro de avance (US$/m)": tot / adv})
    return out


def run_design(section, rock, drill, exp_main, exp_cor, opts) -> Design:
    d = Design(section, rock, drill, exp_main, exp_cor, opts)
    d.p["SB_prod"] = opts.SB_prod
    compute_parameters(d)
    build_layout(d)
    charge_holes(d)
    return d
