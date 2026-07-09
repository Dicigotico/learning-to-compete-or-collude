# -*- coding: utf-8 -*-
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BETA, C = 2.0, 0.0
theta1 = np.array([1.2, 0.8]); theta2 = np.array([0.9, 1.1])
d = 2

def demand_probs(u1, u2):
    e1, e2 = np.exp(BETA*u1), np.exp(BETA*u2)
    den = 1.0 + e1 + e2
    return e1/den, e2/den

def profits(p1, p2, v1, v2):
    q1, q2 = demand_probs(v1-p1, v2-p2)
    return (p1-C)*q1, (p2-C)*q2

def best_response(p_rival, v_own, v_riv, grid):
    e_own = np.exp(BETA*(v_own-grid)); e_riv = np.exp(BETA*(v_riv-p_rival))
    prof = (grid-C)*e_own/(1.0+e_own+e_riv)
    return grid[int(np.argmax(prof))]

def nash_prices(v1, v2, grid, iters=40):
    p1, p2 = 1.0, 1.0
    for _ in range(iters):
        p1n = best_response(p2, v1, v2, grid)
        p2n = best_response(p1n, v2, v1, grid)
        p1, p2 = p1n, p2n
    return p1, p2

v1c, v2c = float(theta1.sum()), float(theta2.sum())  # x=(1,1) -> v1=2.0, v2=2.0
K = 15
gridQ = np.linspace(0.3, 2.5, K)
PN = nash_prices(v1c, v2c, np.linspace(0.05, 3, 500))
pi_N = sum(profits(PN[0], PN[1], v1c, v2c))/2
# monopolio conjunto sobre la malla
best, PM = -1, None
for pa in gridQ:
    for pb in gridQ:
        pr = sum(profits(pa, pb, v1c, v2c))
        if pr > best: best, PM = pr, (pa, pb)
pi_M = best/2

def run_q(memory, T=40000, alpha=0.15, gamma=0.95, beta_eps=1.2e-4, seed=0, vol=0.0):
    r = np.random.default_rng(seed)
    nS = K if memory else 1
    Q1 = np.zeros((nS, K)); Q2 = np.zeros((nS, K))
    a1, a2 = int(r.integers(K)), int(r.integers(K))
    s1 = a2 if memory else 0   # estado de la firma 1: ultimo precio del rival
    s2 = a1 if memory else 0
    tail = []
    for t in range(T):
        if vol > 0:
            x = 1.0 + r.uniform(-vol, vol, d)
            v1, v2 = float(theta1@x), float(theta2@x)
        else:
            v1, v2 = v1c, v2c
        eps = np.exp(-beta_eps*t)
        na1 = int(r.integers(K)) if r.uniform() < eps else int(np.argmax(Q1[s1]))
        na2 = int(r.integers(K)) if r.uniform() < eps else int(np.argmax(Q2[s2]))
        r1, r2 = profits(gridQ[na1], gridQ[na2], v1, v2)
        ns1 = na2 if memory else 0
        ns2 = na1 if memory else 0
        Q1[s1, na1] += alpha*(r1 + gamma*np.max(Q1[ns1]) - Q1[s1, na1])
        Q2[s2, na2] += alpha*(r2 + gamma*np.max(Q2[ns2]) - Q2[s2, na2])
        s1, s2 = ns1, ns2
        if t > T-8000: tail.append((r1+r2)/2)
    return float(np.mean(tail))

def delta(pi): return (pi - pi_N) / (pi_M - pi_N)

SEEDS = 10
d_mem = [delta(run_q(True,  seed=s)) for s in range(SEEDS)]
d_nom = [delta(run_q(False, seed=s)) for s in range(SEEDS)]
d_vol = [delta(run_q(True,  seed=s, vol=0.45)) for s in range(SEEDS)]

def ci(a):
    a = np.array(a); return float(a.mean()), float(1.96*a.std(ddof=1)/np.sqrt(len(a)))

out = {"pN": [float(PN[0]), float(PN[1])], "pi_N": float(pi_N),
       "pM": [float(PM[0]), float(PM[1])], "pi_M": float(pi_M),
       "delta_mem": ci(d_mem), "delta_nom": ci(d_nom), "delta_vol": ci(d_vol),
       "d_mem_all": d_mem, "d_nom_all": d_nom, "d_vol_all": d_vol}
json.dump(out, open("res2.json","w"), indent=1)
print(json.dumps(out, indent=1))

plt.rcParams.update({"font.size": 10.5, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
labels = ["Q-learning con memoria\n(demanda estable)", "Q-learning sin memoria", "Q-learning con memoria\n(contexto volátil)"]
means = [ci(d_mem)[0], ci(d_nom)[0], ci(d_vol)[0]]
errs  = [ci(d_mem)[1], ci(d_nom)[1], ci(d_vol)[1]]
ax.bar(labels, means, yerr=errs, capsize=6, color=["#B23A48", "#1F4E79", "#5B8C5A"], width=0.55)
ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color="gray", ls="--", lw=0.8)
ax.set_ylabel("Índice de colusión Δ")
ax.set_title("Índice Δ por diseño algorítmico (10 semillas, IC 95%; Δ=0 Nash, Δ=1 monopolio)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("fig2_colusion.png"); plt.close(fig)
print("FIG2 OK")
