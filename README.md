# Diseño de mallas de perforación y voladura — Método Holmberg

Web app en Streamlit para el avance horizontal subterráneo con cuele de taladros paralelos.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pestañas

| Pestaña | Qué hace |
|---|---|
| Flujo de cálculo | Memoria de cálculo por fases (Ec. 23–82 del libro de Carpio Chávez, 2025) y distribución por zona |
| Predicción de P80 | Ruta A: Kuz-Ram (Lilly → Kuznetsov → Cunningham → Rosin-Rammler) y búsqueda del burden óptimo. Ruta B: modelo ML propio (.joblib/.pkl) + Evolución Diferencial sobre B y S/B. El botón *Aplicar* regenera la malla con ese burden |
| Frente 2D | Sección, alivios, aberturas del cuele y taladros por zona; etiquetas de N°, retardo o carga |
| Vista 3D y carguío | Taladros con taco, columna de carga, cebo y carmex / tubo de choque / cable; ángulo de contorno γ; columna de carga por zona |
| Secuenciamiento | Árbol de alivio (Dijkstra) + asignación de retardos con restricciones; MILP exacto (HiGHS) o heurística; carga máxima por retardo directa o por PPV; animación de salida; longitudes de mecha para carmex |
| Resumen y exportación | Indicadores (factor de carga, perforación específica, costo/m) y descarga en CSV/JSON |

## Estructura

```
app.py                 interfaz Streamlit
core/sections.py       galerías: baúl, medio punto, herradura, rectangular, trapezoidal
core/holmberg.py       cálculo Holmberg, generación de la malla y carguío
core/fragmentation.py  Kuz-Ram y ruta ML
core/sequencing.py     secuenciamiento óptimo
core/viz.py            figuras Plotly (2D, 3D, perfiles, secuencia)
assets/                diagrama de flujo de referencia
```

## Criterios y supuestos

- El arranque y los cuadrantes siguen las Ec. 23–46. El número de cuadrantes es automático (se acepta el cuadrante N si A_hN ≤ √H', con 5 % de tolerancia) o manual.
- Los burdens de ayudas, hastiales y arrastre salen de B_max = 0.9·√(q₀·RWS/(Ĉ·f·S/B)); la corona usa voladura controlada (Ec. 71–77).
- Las reglas geométricas del libro (H_disp, L_segm) se muestran en la memoria. La malla distribuye los taladros sobre el contorno real de cada sección: fila de ayudas paralela al contorno y relleno simétrico hasta que todo punto de roca quede a ≤ ½·√(B² + S²) de un taladro.
- El cuele se ubica según H_disp(↓) = 0.64 y H_disp(↑) = 0.36 de (H − A_h), o a la altura que fije el usuario.
- Kuz-Ram usa el índice de uniformidad de Cunningham (1987) con el factor (Lc/H').
- Los valores de vibraciones (K, β), combustión de la mecha y tiempos nominales de detonadores son referenciales: reemplácelos por los del sitio y del fabricante.
- El "sustituto de demostración" de la ruta ML no es un modelo real; sólo sirve para probar el flujo.
