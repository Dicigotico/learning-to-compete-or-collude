# -*- coding: utf-8 -*-
# Calibracion con datos reales.
# Parte R (retail): hedonica sobre diamonds (53.940 transacciones reales) -> entorno calibrado.
# Parte E (electrico): coste mayorista real SMARD (2022 crisis vs 2024 normal) -> Q-learning.
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from smard_prices import P2022, P2024

BETA = 2.0

def dp(a, b):
    ea, eb = np.exp(a), np.exp(b); D = 1+ea+eb
    return ea/D, eb/D

def br(p_riv, v_own, v_riv, grid, c=0.0):
    e_o = np.exp(BETA*(v_own-grid)); e_r = np.exp(BETA*(v_riv-p_riv))
    return grid[int(np.argmax((grid-c)*e_o/(1+e_o+e_r)))]

def nash(v1, v2, grid, c=0.0, iters=25):
    p1, p2 = 1.0, 1.0
    for _ in range(iters):
        p1n = br(p2, v1, v2, grid, c); p2n = br(p1n, v2, v1, grid, c)
        if abs(p1n-p1) < 1e-9 and abs(p2n-p2) < 1e-9: return p1n, p2n
        p1, p2 = p1n, p2n
    return p1, p2

def prof(p1, p2, v1, v2, c=0.0):
    q1, q2 = dp(BETA*(v1-p1), BETA*(v2-p2))
    return (p1-c)*q1, (p2-c)*q2

# ---------------- Parte R: hedonica diamonds ----------------
def parte_retail():
    import plotnine.data as pdta
    df = pdta.diamonds.copy()
    cut_o = {"Fair":1, "Good":2, "Very Good":3, "Premium":4, "Ideal":5}
    col_o = {c: i for i, c in enumerate("JIHGFED", 1)}
    cla_o = {"I1":1, "SI2":2, "SI1":3, "VS2":4, "VS1":5, "VVS2":6, "VVS1":7, "IF":8}
    X = np.column_stack([df["carat"].values,
                         df["cut"].map(cut_o).values.astype(float),
                         df["color"].map(col_o).values.astype(float),
                         df["clarity"].map(cla_o).values.astype(float)])
    y = np.log(df["price"].values.astype(float))
    Xs = np.column_stack([np.ones(len(y)), X])
    beta_h, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    r2 = 1 - np.sum((y - Xs@beta_h)**2)/np.sum((y-y.mean())**2)
    # entorno calibrado: contexto = atributos reales estandarizados (d=4)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu)/sd
    theta = beta_h[1:]*sd            # efecto hedonico por unidad estandarizada
    v_raw = Z @ theta
    a, b = 2.0/ (v_raw.std()*4), 2.0  # centrar v en 2 con dispersion ~0.5
    vals = b + v_raw*a
    rng = np.random.default_rng(5)
    grid = np.linspace(0.05, 3.5, 250)
    # (R1) CEP-N sobre contextos reales (theta comun a ambas firmas => v1=v2)
    est = [np.zeros(4), np.zeros(4)]  # cada firma estima su theta (v propio)
    lr0 = 1.2; T = 5000
    gaps = []
    idx = rng.integers(0, len(vals), T)
    for t in range(1, T+1):
        z = Z[idx[t-1]]; v = vals[idx[t-1]]
        sig = 0.6/t**0.25
        p_int = np.zeros(2)
        for i in range(2):
            vh = 2.0 + (z @ est[i])   # intercepto conocido, pendientes aprendidas
            pn = nash(vh, vh, grid)
            p_int[i] = pn[0]
        p = np.clip(p_int + rng.normal(0, sig, 2), 0.05, 3.5)
        P1, P2 = dp(BETA*(v-p[0]), BETA*(v-p[1]))
        u = rng.uniform()
        yv = [1.0 if u < P1 else 0.0, 1.0 if P1 <= u < P1+P2 else 0.0]
        lr = lr0/np.sqrt(t)
        for i in range(2):
            vh = 2.0 + z @ est[i]
            Ph, _ = dp(BETA*(vh-p[i]), BETA*(vh-p[1-i]))
            est[i] += lr*(yv[i]-Ph)*BETA*z
        pn_t = nash(v, v, grid)
        gaps.append(abs(p_int[0]-pn_t[0]) + abs(p_int[1]-pn_t[1]))
    g = np.array(gaps); w = 200
    gs = np.convolve(g, np.ones(w)/w, mode="valid")
    np.save("gs_diam.npy", gs)
    # (R2) Q-learning sin memoria: contexto fijo vs rotacion real de atributos
    K = 15; gridQ = np.linspace(0.3, 2.8, K)
    v_fix = 2.0
    PNf = nash(v_fix, v_fix, np.linspace(0.05, 3.5, 500))
    piN = sum(prof(PNf[0], PNf[1], v_fix, v_fix))/2
    bestM, PM = -1, None
    for pa in gridQ:
        for pb in gridQ:
            pr = sum(prof(pa, pb, v_fix, v_fix))
            if pr > bestM: bestM, PM = pr, (pa, pb)
    piM = bestM/2
    def runq(real_ctx, seed, T=40000, alpha=0.15, gamma=0.95, be=1.2e-4):
        r = np.random.default_rng(seed)
        Q1 = np.zeros(K); Q2 = np.zeros(K)
        tail = []
        ids = r.integers(0, len(vals), T)
        for t in range(T):
            v = vals[ids[t]] if real_ctx else v_fix
            eps = np.exp(-be*t)
            a1 = int(r.integers(K)) if r.uniform() < eps else int(np.argmax(Q1))
            a2 = int(r.integers(K)) if r.uniform() < eps else int(np.argmax(Q2))
            r1, r2 = prof(gridQ[a1], gridQ[a2], v, v)
            Q1[a1] += alpha*(r1 + gamma*np.max(Q1) - Q1[a1])
            Q2[a2] += alpha*(r2 + gamma*np.max(Q2) - Q2[a2])
            if t > T-8000: tail.append((r1+r2)/2)
        return (np.mean(tail)-piN)/(piM-piN)
    d_fix  = [runq(False, s) for s in range(10)]
    d_real = [runq(True, s) for s in range(10)]
    def ci(a):
        a = np.array(a); return float(a.mean()), float(1.96*a.std(ddof=1)/np.sqrt(len(a)))
    out = {"r2_hedonica": float(r2), "theta_std": [float(x) for x in theta],
           "gap_ini": float(gs[:50].mean()), "gap_fin": float(gs[-200:].mean()),
           "delta_fijo": ci(d_fix), "delta_real": ci(d_real),
           "sd_v": float(vals.std())}
    json.dump(out, open("res_retail.json", "w")); print(json.dumps(out, indent=1))

# ---------------- Parte E: costes reales SMARD ----------------
def parte_elec():
    rng = np.random.default_rng(3)
    K = 15; gridQ = np.linspace(0.3, 2.8, K)
    v_fix = 2.0
    c22 = np.array(P2022); c24 = np.array(P2024)
    # normalizar cada serie a coste medio 0,5 (aisla la forma de la volatilidad)
    cost22 = 0.5*c22/np.mean(c22)
    cost24 = 0.5*c24/np.mean(c24)
    cv22 = float(np.std(c22)/np.mean(c22)); cv24 = float(np.std(c24)/np.mean(c24))
    gridN = np.linspace(0.05, 3.5, 500)
    def refs(c):
        PN = nash(v_fix, v_fix, gridN, c=c)
        piN = sum(prof(PN[0], PN[1], v_fix, v_fix, c))/2
        best = -1
        for pa in gridQ:
            for pb in gridQ:
                pr = sum(prof(pa, pb, v_fix, v_fix, c))
                if pr > best: best = pr
        return piN, best/2
    def runq(cost_series, seed, T=40000, alpha=0.15, gamma=0.95, be=1.2e-4):
        r = np.random.default_rng(seed)
        Q1 = np.zeros(K); Q2 = np.zeros(K)
        tail_p = []; tail_refs = []
        n = len(cost_series) if cost_series is not None else 0
        for t in range(T):
            c = cost_series[t % n] if cost_series is not None else 0.5
            eps = np.exp(-be*t)
            a1 = int(r.integers(K)) if r.uniform() < eps else int(np.argmax(Q1))
            a2 = int(r.integers(K)) if r.uniform() < eps else int(np.argmax(Q2))
            r1, r2 = prof(gridQ[a1], gridQ[a2], v_fix, v_fix, c)
            Q1[a1] += alpha*(r1 + gamma*np.max(Q1) - Q1[a1])
            Q2[a2] += alpha*(r2 + gamma*np.max(Q2) - Q2[a2])
            if t > T-8000:
                tail_p.append((r1+r2)/2); tail_refs.append(refs_cache[round(c,4)])
        pN = np.mean([a for a, _ in tail_refs]); pM = np.mean([b for _, b in tail_refs])
        return (np.mean(tail_p)-pN)/(pM-pN)
    # cache de referencias por coste
    global refs_cache
    refs_cache = {}
    for series in [None, cost24, cost22]:
        cs = [0.5] if series is None else sorted(set(np.round(series, 4)))
        for c in cs:
            if round(c, 4) not in refs_cache: refs_cache[round(c, 4)] = refs(c)
    d_cte = [runq(None, s) for s in range(10)]
    d_24  = [runq(cost24, s) for s in range(10)]
    d_22  = [runq(cost22, s) for s in range(10)]
    def ci(a):
        a = np.array(a); return float(a.mean()), float(1.96*a.std(ddof=1)/np.sqrt(len(a)))
    out = {"cv_2022": cv22, "cv_2024": cv24,
           "delta_cte": ci(d_cte), "delta_2024": ci(d_24), "delta_2022": ci(d_22)}
    json.dump(out, open("res_elec.json", "w")); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    {"r": parte_retail, "e": parte_elec}[sys.argv[1]]()
