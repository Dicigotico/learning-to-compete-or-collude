# -*- coding: utf-8 -*-
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
BETA, C = 2.0, 0.0

def demand_probs(u1, u2):
    e1, e2 = np.exp(BETA*u1), np.exp(BETA*u2)
    den = 1.0 + e1 + e2
    return e1/den, e2/den

def best_response(p_rival, v_own, v_riv, grid):
    e_own = np.exp(BETA*(v_own-grid)); e_riv = np.exp(BETA*(v_riv-p_rival))
    prof = (grid-C)*e_own/(1.0+e_own+e_riv)
    return grid[int(np.argmax(prof))]

def nash_prices(v1, v2, grid, iters=30):
    p1, p2 = 1.0, 1.0
    for _ in range(iters):
        p1n = best_response(p2, v1, v2, grid)
        p2n = best_response(p1n, v2, v1, grid)
        if abs(p1n-p1)<1e-9 and abs(p2n-p2)<1e-9:
            p1, p2 = p1n, p2n; break
        p1, p2 = p1n, p2n
    return p1, p2

d = 2
theta1 = np.array([1.2, 0.8]); theta2 = np.array([0.9, 1.1])
T1 = 5000
gridF = np.linspace(0.05, 3.0, 250)

th1_hat = np.zeros(d); th2_hat = np.zeros(d)
lr0 = 0.8
gaps = []
p1_prev, p2_prev = 1.0, 1.0
for t in range(1, T1+1):
    x = rng.uniform(0.5, 1.5, d)
    v1, v2 = theta1@x, theta2@x
    v1h, v2h = th1_hat@x, th2_hat@x
    sig = 0.6 / t**0.35
    p1 = best_response(p2_prev, v1h, v2h, gridF) + rng.normal(0, sig)
    p2 = best_response(p1_prev, v2h, v1h, gridF) + rng.normal(0, sig)
    p1 = float(np.clip(p1, 0.05, 3.0)); p2 = float(np.clip(p2, 0.05, 3.0))
    q1p, q2p = demand_probs(v1-p1, v2-p2)
    u = rng.uniform()
    y1 = 1.0 if u < q1p else 0.0
    y2 = 1.0 if (u >= q1p and u < q1p+q2p) else 0.0
    lr = lr0 / np.sqrt(t)
    q1h, q2h = demand_probs(v1h-p1, v2h-p2)
    th1_hat += lr * (y1 - q1h) * BETA * x * q1h*(1-q1h) * 8
    th2_hat += lr * (y2 - q2h) * BETA * x * q2h*(1-q2h) * 8
    pn1, pn2 = nash_prices(v1, v2, gridF)
    gaps.append(abs(p1-pn1)+abs(p2-pn2))
    p1_prev, p2_prev = p1, p2

gaps = np.array(gaps); w = 200
gap_smooth = np.convolve(gaps, np.ones(w)/w, mode="valid")

out = {"gap_inicial": float(gap_smooth[:50].mean()),
       "gap_t1000": float(gap_smooth[1000-w]),
       "gap_final": float(gap_smooth[-200:].mean()),
       "theta1_err": float(np.linalg.norm(th1_hat-theta1)),
       "theta2_err": float(np.linalg.norm(th2_hat-theta2))}
json.dump(out, open("res1.json","w"), indent=1)
print(json.dumps(out, indent=1))

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(np.arange(len(gap_smooth))+w, gap_smooth, color="#1F4E79", lw=1.6)
ax.set_xlabel("Periodo t"); ax.set_ylabel("Distancia al Nash  |p1−p1*|+|p2−p2*|")
ax.set_title("Convergencia al Nash — learners sin memoria (media móvil, 200 periodos)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig1_convergencia.png"); plt.close(fig)
print("FIG1 OK")
