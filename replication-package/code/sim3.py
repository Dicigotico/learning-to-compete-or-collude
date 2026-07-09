# -*- coding: utf-8 -*-
# Experimento 3: CEP-N (equivalencia cierta de Nash + exploracion decreciente)
# frente a mejor respuesta rezagada; y barrido de dimension d.
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BETA, C = 2.0, 0.0

def demand_P(a_off, b_off):
    ea, eb = np.exp(a_off), np.exp(b_off)
    D = 1.0 + ea + eb
    return ea/D, eb/D

def best_response(p_rival, v_own, v_riv, grid):
    e_own = np.exp(BETA*(v_own-grid)); e_riv = np.exp(BETA*(v_riv-p_rival))
    prof = (grid-C)*e_own/(1.0+e_own+e_riv)
    return grid[int(np.argmax(prof))]

def nash_prices(v1, v2, grid, iters=30):
    p1, p2 = 1.0, 1.0
    for _ in range(iters):
        p1n = best_response(p2, v1, v2, grid)
        p2n = best_response(p1n, v2, v1, grid)
        if abs(p1n-p1) < 1e-9 and abs(p2n-p2) < 1e-9:
            return p1n, p2n
        p1, p2 = p1n, p2n
    return p1, p2

def run_ce(d, T, seed, mode="ce"):
    """mode 'ce': equivalencia cierta de Nash; 'lag': mejor respuesta rezagada."""
    rng = np.random.default_rng(seed)
    theta1 = rng.uniform(0.6, 1.4, d); theta1 = theta1/np.linalg.norm(theta1)*np.sqrt(d)*1.0
    theta2 = rng.uniform(0.6, 1.4, d); theta2 = theta2/np.linalg.norm(theta2)*np.sqrt(d)*1.0
    # normalizar para que v ~ O(2) como en d=2
    scale = 2.0 / (theta1 @ (np.ones(d)))
    theta1 *= scale; theta2 *= 2.0/(theta2 @ np.ones(d))
    grid = np.linspace(0.05, 3.0, 250)
    # cada firma estima el par completo (th_own, th_riv)
    est = [np.zeros((2, d)), np.zeros((2, d))]
    lr0 = 1.2
    p_prev = np.array([1.0, 1.0])
    gaps = []
    for t in range(1, T+1):
        x = rng.uniform(0.5, 1.5, d)
        v = np.array([theta1 @ x, theta2 @ x])
        sig = 0.6 / t**0.25
        p = np.zeros(2)
        for i in range(2):
            vi_h = est[i][0] @ x; vj_h = est[i][1] @ x
            if mode == "ce":
                pn = nash_prices(vi_h, vj_h, grid)
                p[i] = pn[0]
            else:
                p[i] = best_response(p_prev[1-i], vi_h, vj_h, grid)
        p_int = p.copy()
        p = np.clip(p + rng.normal(0, sig, 2), 0.05, 3.0)
        # demanda realizada
        a = BETA*(v[0]-p[0]); b = BETA*(v[1]-p[1])
        P1, P2 = demand_P(a, b)
        u = rng.uniform()
        y = np.array([1.0 if u < P1 else 0.0, 1.0 if (P1 <= u < P1+P2) else 0.0])
        # actualizacion SGD del softmax (cada firma con su y propio)
        lr = lr0/np.sqrt(t)
        for i in range(2):
            ah = BETA*((est[i][0] @ x) - p[i]); bh = BETA*((est[i][1] @ x) - p[1-i])
            Ph_own, Ph_riv = demand_P(ah, bh)
            resid = y[i] - Ph_own
            est[i][0] += lr * resid * BETA * x
            if Ph_own < 1.0 - 1e-9:
                est[i][1] += lr * (-resid * Ph_riv/(1.0-Ph_own)) * BETA * x
        pn_true = nash_prices(v[0], v[1], grid)
        gaps.append(abs(p_int[0]-pn_true[0]) + abs(p_int[1]-pn_true[1]))
        p_prev = p
    g = np.array(gaps)
    w = 200
    gs = np.convolve(g, np.ones(w)/w, mode="valid")
    return gs

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "a":  # d=2: CE vs lag
        gs_ce = run_ce(2, 5000, 7, "ce")
        gs_lag = run_ce(2, 5000, 7, "lag")
        np.save("gs_ce.npy", gs_ce); np.save("gs_lag.npy", gs_lag)
        print(json.dumps({"ce_final": float(gs_ce[-200:].mean()),
                          "lag_final": float(gs_lag[-200:].mean()),
                          "ce_ini": float(gs_ce[:50].mean()),
                          "lag_ini": float(gs_lag[:50].mean())}))
    elif which == "b":  # barrido de dimension con CE
        out = {}
        for d in [5, 10]:
            gs = run_ce(d, 4000, 11, "ce")
            out[str(d)] = {"final": float(gs[-200:].mean()), "ini": float(gs[:50].mean())}
            np.save(f"gs_d{d}.npy", gs)
        print(json.dumps(out))
    elif which == "c":
        gs = run_ce(20, 4000, 11, "ce")
        np.save("gs_d20.npy", gs)
        print(json.dumps({"20": {"final": float(gs[-200:].mean()), "ini": float(gs[:50].mean())}}))
