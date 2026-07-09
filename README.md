# Paquete de replicación
## "¿Aprender a competir o aprender a coludir? Pricing dinámico basado en atributos bajo competencia estratégica"

Reproduce íntegramente los Experimentos 1–4, la calibración con datos reales (Sección 7) y las Figuras 1–6 del manuscrito.

## Requisitos
Python 3.10+, `numpy`, `matplotlib`, `plotnine` (solo para el dataset `diamonds` de la calibración retail).

## Estructura
- `code/sim1.py` — Experimento 1: learners basados en modelo, convergencia al Nash (Figura 1).
- `code/sim2.py` — Experimento 2: Q-learning con/sin memoria, índice de colusión Δ (Figura 2). Variando `beta_eps` reproduce la evidencia de selección de equilibrio (Apéndice F.5) y variando `vol` la dosis-respuesta (Figura 4).
- `code/sim3.py` — Experimento 3: CEP-N (equivalencia cierta) vs. mejor respuesta rezagada; barrido de dimensión d ∈ {2,5,10,20} (Figura 3). Uso: `python sim3.py a|b|c`.
- `code/sim5.py` — Sección 7: calibración retail (hedónica sobre `diamonds`, R²=0,88) y eléctrica (costes SMARD). Uso: `python sim5.py r|e` (Figuras 5–6).
- `data/smard_prices.py` — Precios day-ahead horarios reales de Alemania (SMARD/Bundesnetzagentur, filtro 4169): semana 2022-08-15 (crisis) y semana 2024-03-18 (normal). Fuente: https://www.smard.de (API pública `chart_data/4169/DE/`).
- `results/*.json` — Salidas numéricas de todas las corridas reportadas (medias e IC 95%).
- `figures/` — Las seis figuras del manuscrito tal como se generaron.

## Correspondencia resultados ↔ manuscrito
| Archivo | Resultado del texto |
|---|---|
| res1.json | Gap 0,44→0,14 (Exp. 1) |
| res2.json, res2b.json | Δ = 0,91/0,52/0,48/0,74 (Exp. 2, Fig. 2) |
| res_vol.json | Dosis-respuesta volatilidad (Fig. 4) |
| res_sel.json | Selección de equilibrio vs η (Apéndice F.5) |
| res3.json, gs_*.npy | Barrido de dimensión CEP-N: 0,11/0,21/0,28/0,36 y gap 0,09 a T=30.000 (Fig. 3) |
| res_retail.json | Calibración diamonds: Δ 0,90→0,68 (Fig. 5) |
| res_elec.json | Calibración SMARD: Δ 0,92/0,84/0,80 (Fig. 6) |

## Semillas y reproducibilidad
Todas las corridas usan `numpy.random.default_rng` con las semillas fijadas en cada script (indicadas en los argumentos por defecto). Los IC al 95% usan 8–10 semillas por celda, como declara el manuscrito.

## Licencia
MIT (código); los datos SMARD son públicos (Bundesnetzagentur); `diamonds` se distribuye con ggplot2/plotnine.

