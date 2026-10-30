"""Geometría de las secciones de galería para avance horizontal.

Convención: origen en el centro del piso, x horizontal (m), y vertical (m).
Todas las secciones se devuelven como polígonos de shapely orientados en
sentido antihorario.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

GALLERY_TYPES = [
    "Baúl (hastiales verticales + esquinas en arco)",
    "Medio punto (bóveda semicircular)",
    "Herradura",
    "Rectangular",
    "Trapezoidal",
]

# Secciones típicas de avance (ancho x alto, m)
PRESETS = {
    "Personalizada": None,
    "2.10 x 2.40": (2.10, 2.40),
    "2.40 x 2.70": (2.40, 2.70),
    "3.00 x 3.00": (3.00, 3.00),
    "3.50 x 3.50": (3.50, 3.50),
    "4.00 x 4.00": (4.00, 4.00),
    "4.50 x 4.00 (Ejemplo N°06 del libro)": (4.50, 4.00),
    "4.50 x 4.50": (4.50, 4.50),
    "5.00 x 4.50": (5.00, 4.50),
}


@dataclass
class Section:
    kind: str
    width: float
    height: float
    radius: float  # radio del baúl / bóveda (m); 0 si no aplica
    top_width: float  # sólo trapezoidal
    polygon: Polygon
    springline: float  # cota (m) sobre la cual el contorno es "corona"

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def perimeter(self) -> float:
        return float(self.polygon.length)

    @property
    def floor_width(self) -> float:
        xs = [x for x, y in self.polygon.exterior.coords if abs(y) < 1e-6]
        return float(max(xs) - min(xs)) if xs else self.width

    @property
    def wall_height(self) -> float:
        """Altura disponible para taladros de hastiales (H - r en el libro)."""
        return float(self.springline)

    def boundary_xy(self) -> np.ndarray:
        return np.asarray(self.polygon.exterior.coords)


def _arc(cx, cy, r, a0, a1, n=40):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))


def build_section(kind: str, A: float, H: float, r: float | None = None,
                  top_width: float | None = None) -> Section:
    if A <= 0 or H <= 0:
        raise ValueError("El ancho y la altura deben ser positivos.")
    a = A / 2.0

    if kind.startswith("Baúl") or kind.startswith("Medio punto"):
        if kind.startswith("Medio punto"):
            r = a
        r = float(np.clip(r if r else A / 3.0, 0.05, min(a, H - 0.05)))
        pts = [(-a, 0.0), (a, 0.0), (a, H - r)]
        pts += _arc(a - r, H - r, r, 0, 90)[1:]
        if a - r > 1e-6:
            pts += [(-(a - r), H)]
        pts += _arc(-(a - r), H - r, r, 90, 180)[1:]
        spring = H - r
        tw = A
    elif kind.startswith("Herradura"):
        R = a
        yc = float(np.clip(H - R, 0.2 * R, 1.9 * R))
        H = yc + R
        ys = np.linspace(0.0, yc, 25)
        right = [(-R + np.sqrt(4 * R**2 - (y - yc) ** 2), y) for y in ys]
        left = [(-x, y) for x, y in reversed(right)]
        top = _arc(0.0, yc, R, 0, 180)[1:-1]
        pts = [(left[-1][0], 0.0)] + right + top + left[:-1]
        spring = yc
        r = R
        tw = A
    elif kind.startswith("Rectangular"):
        pts = [(-a, 0.0), (a, 0.0), (a, H), (-a, H)]
        spring = H - 1e-3
        r = 0.0
        tw = A
    elif kind.startswith("Trapezoidal"):
        tw = float(np.clip(top_width if top_width else 0.8 * A, 0.3 * A, A))
        pts = [(-a, 0.0), (a, 0.0), (tw / 2, H), (-tw / 2, H)]
        spring = H - 1e-3
        r = 0.0
    else:
        raise ValueError(f"Tipo de sección desconocido: {kind}")

    poly = orient(Polygon(pts).buffer(0), sign=1.0)
    return Section(kind=kind, width=A, height=H, radius=float(r or 0.0),
                   top_width=float(tw), polygon=poly, springline=float(spring))
