"""Secuenciamiento óptimo de salida de los taladros.

1. Grafo de alivio: cada taladro "sale" hacia la cara libre más cercana. Se construye
   un grafo entre taladros vecinos (distancia ≤ k·B) y se calcula el camino mínimo
   (Dijkstra) desde los taladros de alivio. El predecesor de cada taladro en ese árbol
   es el taladro que debe salir antes para abrirle cara libre.
2. Asignación de retardos con restricciones:
   - precedencia (el taladro sale después de su predecesor),
   - orden clásico por zonas (arranque → cuadrantes → ayudas → hastiales → corona → arrastre),
   - un solo taladro por retardo en el cuele (opcional),
   - carga máxima por retardo (criterio de vibraciones),
   - corona simultánea para voladura controlada (opcional).
   Métodos: heurística de programación por listas o MILP exacto (HiGHS) que minimiza
   el número de retardos.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

from .holmberg import CUT_ZONES, HELPER_ZONES

GROUP_OF = {"Arranque": 0, "Cuadrante 1": 1, "Cuadrante 2": 2, "Cuadrante 3": 3, "Cuadrante 4": 4,
            "Ayuda de hastiales": 5, "Ayuda de corona": 5, "Ayuda de arrastre": 5,
            "Hastiales": 6, "Corona": 7, "Arrastre": 8}


def relief_graph(holes: pd.DataFrame, relief: pd.DataFrame, link=1.6):
    n, m = len(holes), len(relief)
    XY = np.vstack([relief[["x", "y"]].values, holes[["x", "y"]].values])
    Bv = np.concatenate([np.full(m, holes.B.min()), holes.B.fillna(holes.B.median()).values])
    D = np.hypot(XY[:, None, 0] - XY[None, :, 0], XY[:, None, 1] - XY[None, :, 1])
    for _ in range(12):
        # arista i → j sólo si i está dentro del radio de alivio del taladro j (k·B_j);
        # peso = d² para preferir caminos de saltos cortos (cara libre progresiva)
        R = link * Bv[None, :]
        W = np.where((D <= R) & (D > 0), D**2, 0.0)
        W[:, :m] = 0.0
        dist, pred, _src = dijkstra(W, directed=True, indices=list(range(m)), min_only=True,
                                    return_predecessors=True)
        if np.all(np.isfinite(dist[m:])):
            break
        link += 0.25
    parent = []
    for j in range(m, m + n):
        p = pred[j]
        parent.append(int(p - m) if p >= m else -1)  # -1 = sale contra el alivio
    return dist[m:], np.array(parent), link


def schedule_heuristic(h: pd.DataFrame, parent, dist, opt):
    n = len(h)
    period = np.zeros(n, dtype=int)
    load, count, cut_used = {}, {}, set()
    classic = opt["classic"]
    grp = h.zona.map(GROUP_OF).values if classic else np.where(h.zona == "Arrastre", 1, 0)
    order = np.lexsort((h.x.values, dist, grp))
    group_end = {}
    corona_idx = [i for i in range(n) if h.zona.iat[i] == "Corona"]
    corona_done = False

    def fits(p, idxs):
        q = sum(h.kg.iat[i] for i in idxs)
        if load.get(p, 0) + q > opt["qmax"] + 1e-9:
            return False
        is_cor_block = opt["corona_sim"] and len(idxs) > 1
        if not is_cor_block and count.get(p, 0) + len(idxs) > opt["nmax"]:
            return False
        if is_cor_block and count.get(p, 0) > 0:
            return False
        if opt["cut_single"] and any(h.zona.iat[i] in CUT_ZONES for i in idxs):
            if p in cut_used or len(idxs) > 1:
                return False
        return True

    def assign(p, idxs):
        for i in idxs:
            period[i] = p
            load[p] = load.get(p, 0) + h.kg.iat[i]
            count[p] = count.get(p, 0) + 1
            if h.zona.iat[i] in CUT_ZONES:
                cut_used.add(p)
            group_end[grp[i]] = max(group_end.get(grp[i], 0), p)

    for i in order:
        if period[i]:
            continue
        g = grp[i]
        start = max([group_end[k] for k in group_end if k < g], default=0) + 1
        earliest = start
        if parent[i] >= 0 and period[parent[i]]:
            earliest = max(earliest, period[parent[i]] + 1)
        if opt["corona_sim"] and h.zona.iat[i] == "Corona" and not corona_done:
            corona_done = True
            e = max([earliest] + [period[parent[j]] + 1 for j in corona_idx if parent[j] >= 0 and period[parent[j]]])
            p = e
            while p < e + 200 and not fits(p, corona_idx):
                p += 1
            if fits(p, corona_idx):
                assign(p, corona_idx)
                continue
        p = earliest
        while not fits(p, [i]):
            p += 1
            if p > 1000:
                raise RuntimeError("No es posible asignar retardos: aumente la carga máxima por retardo.")
        assign(p, [i])
    return period


def schedule_milp(h: pd.DataFrame, parent, opt, P: int, time_limit=20):
    n = len(h)
    classic = opt["classic"]
    grp = h.zona.map(GROUP_OF).values if classic else np.where(h.zona == "Arrastre", 1, 0)
    groups = sorted(set(grp))
    G = len(groups)
    gidx = {g: k for k, g in enumerate(groups)}
    nx = n * P
    nv = nx + G + 1  # x, E_g (fin de grupo), z (makespan)
    X = lambda i, p: i * P + p
    rows, lb, ub = [], [], []

    def add_row(coefs, lo, hi):
        r = lil_matrix((1, nv))
        for j, v in coefs:
            r[0, j] = v
        rows.append(r)
        lb.append(lo)
        ub.append(hi)

    pr = np.arange(1, P + 1)
    for i in range(n):  # un período por taladro
        add_row([(X(i, p), 1) for p in range(P)], 1, 1)
    for i in range(n):  # precedencia
        j = parent[i]
        same_cor = opt["corona_sim"] and h.zona.iat[i] == "Corona" and j >= 0 and h.zona.iat[j] == "Corona"
        if j >= 0 and grp[j] <= grp[i] and not same_cor:
            add_row([(X(i, p), pr[p]) for p in range(P)] + [(X(j, p), -pr[p]) for p in range(P)], 1, np.inf)
    for i in range(n):  # fin de grupo y makespan
        e = nx + gidx[grp[i]]
        add_row([(X(i, p), pr[p]) for p in range(P)] + [(e, -1)], -np.inf, 0)
        add_row([(X(i, p), pr[p]) for p in range(P)] + [(nv - 1, -1)], -np.inf, 0)
        k = gidx[grp[i]]
        if k > 0:  # empieza después del grupo anterior
            add_row([(X(i, p), pr[p]) for p in range(P)] + [(nx + k - 1, -1)], 1, np.inf)
    for p in range(P):  # capacidad por retardo
        add_row([(X(i, p), h.kg.iat[i]) for i in range(n)], -np.inf, opt["qmax"])
        cnt = [i for i in range(n) if not (opt["corona_sim"] and h.zona.iat[i] == "Corona")]
        add_row([(X(i, p), 1) for i in cnt], -np.inf, opt["nmax"])
        if opt["cut_single"]:
            cut = [i for i in range(n) if h.zona.iat[i] in CUT_ZONES]
            if cut:
                add_row([(X(i, p), 1) for i in cut], -np.inf, 1)
    if opt["corona_sim"]:
        cor = [i for i in range(n) if h.zona.iat[i] == "Corona"]
        for i in cor[1:]:
            add_row([(X(i, p), pr[p]) for p in range(P)] + [(X(cor[0], p), -pr[p]) for p in range(P)], 0, 0)
    from scipy.sparse import vstack
    A = vstack(rows).tocsr()
    c = np.zeros(nv)
    c[nv - 1] = 1000.0
    for i in range(n):
        for p in range(P):
            c[X(i, p)] = pr[p]
    integrality = np.r_[np.ones(nx), np.zeros(G + 1)]
    bounds = Bounds(np.zeros(nv), np.r_[np.ones(nx), np.full(G + 1, P)])
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, disp=False))
    if res.x is None:
        return None, res
    x = res.x[:nx].reshape(n, P)
    return x.argmax(axis=1) + 1, res


def delay_times(period, system, cfg):
    p = np.asarray(period)
    if system == "Carmex (mecha de seguridad + fulminante)":
        t = (p - 1) * cfg["dt_s"] * 1000.0
    elif system == "Electrónico":
        t = cfg["t0"] + (p - 1) * cfg["dt_ms"]
    else:  # no eléctrico con tabla del fabricante
        tab = list(cfg["table"])
        if p.max() > len(tab):
            step = tab[-1] - tab[-2] if len(tab) > 1 else 100
            tab += [tab[-1] + step * (k + 1) for k in range(p.max() - len(tab))]
        t = np.array([tab[k - 1] for k in p], dtype=float)
    return t


def fuse_lengths(period, cfg, H):
    """Longitud de mecha de seguridad por taladro (carmex escalonado)."""
    base = cfg["tail"] + (H if cfg["primer"] == "Fondo" else cfg["Lt"] + 0.2)
    dL = cfg["dt_s"] / cfg["burn"]  # m de mecha por período
    return base + (np.asarray(period) - 1) * dL


def ppv(Q, D, K, beta):
    return K * (D / np.sqrt(np.maximum(Q, 1e-9))) ** (-beta)


def qmax_from_ppv(ppv_lim, D, K, beta):
    return D**2 * (ppv_lim / K) ** (2 / beta)


def run_sequencing(holes, relief, opt, system, cfg):
    h = holes.reset_index(drop=True)
    dist, parent, link = relief_graph(h, relief, opt.get("link", 1.6))
    period_h = schedule_heuristic(h, parent, dist, opt)
    method = "Heurística (listas con restricciones)"
    period = period_h
    info = {}
    if opt.get("method") == "MILP":
        P = int(period_h.max())
        per_m, res = schedule_milp(h, parent, opt, P, time_limit=opt.get("time_limit", 20))
        info["milp_status"] = res.message
        if per_m is not None:
            # compactar períodos vacíos
            used = sorted(set(per_m))
            remap = {p: k + 1 for k, p in enumerate(used)}
            period = np.array([remap[p] for p in per_m])
            method = "MILP (HiGHS)" + (" — óptimo" if res.status == 0 else " — mejor solución en el tiempo límite")
        else:
            method += " (MILP sin solución factible en el tiempo límite)"
    out = h.copy()
    out["predecesor"] = [int(h.id.iat[j]) if j >= 0 else 0 for j in parent]
    out["dist_alivio"] = dist
    out["periodo"] = period
    out["t_ms"] = delay_times(period, system, cfg)
    if system.startswith("Carmex"):
        out["mecha_m"] = fuse_lengths(period, cfg, float(h.L.iat[0]))
    out = out.sort_values(["periodo", "dist_alivio"]).reset_index(drop=True)
    out["orden"] = np.arange(1, len(out) + 1)
    info.update(method=method, link=link, n_periods=int(out.periodo.max()),
                heur_periods=int(period_h.max()))
    return out, info
