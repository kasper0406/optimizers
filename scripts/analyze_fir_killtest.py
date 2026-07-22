#!/usr/bin/env python
"""Program #9 offline kill-tests (promoted from the session scratchpad,
2026-07-22): estimator-variance comparison of lag-constrained optimal
filters vs (Nesterov-)EMA kernels on measured per-direction ACFs, plus
the kernel-family-vs-rho-matching decomposition. See
reports/fir-phase-a.md section 1 and fir-phase-b.md for interpretation
(the headroom is real as measurement, void as intervention: the
anti-correlation proved optimizer-endogenous)."""
"""
import json, glob, sys
import numpy as np

BURNIN = 5
MAX_LAG = 8

def window_acf(s, refresh, burnin=BURNIN, max_lag=MAX_LAG):
    n = len(s); bounds = [r-1 for r in refresh if 0 <= r-1 < n] + [n]
    num = np.zeros(max_lag+1); den = 0.0; cnt = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        w = s[lo+burnin:hi]; w = w[np.isfinite(w)]
        if len(w) < 30: continue
        w = w - w.mean(); sd = w.std()
        if sd <= 0: continue
        w = w / sd
        for k in range(max_lag+1):
            num[k] += np.dot(w[:len(w)-k], w[k:]) / (len(w)-k)
        cnt += 1
    if cnt == 0: return None
    acf = num / cnt
    return acf / acf[0]

def gamma_from_acf(acf, L):
    g = np.zeros(L)
    g[:len(acf)] = acf
    return np.array([[g[abs(i-j)] for j in range(L)] for i in range(L)])

def ema_weights(tau, L):
    beta = tau/(1.0+tau)
    w = (1-beta) * beta**np.arange(L)
    return w / w.sum()

def opt_weights(G, tau, L):
    A = np.stack([np.ones(L), np.arange(L, dtype=float)], axis=1)
    b = np.array([1.0, tau])
    Gi = np.linalg.solve(G + 1e-9*np.eye(L), A)
    lam = np.linalg.solve(A.T @ Gi, b)
    return Gi @ lam

def main(files):
    rows = []
    acfs = {'top': [], 'bulk': []}
    for f in files:
        side = json.load(open(f))
        for mat in side['matrices'].values():
            refresh = mat['refresh_steps']
            for d in mat['directions']:
                s = np.asarray(d['s'], float)
                if len(s) < 300: continue
                acf = window_acf(s, refresh)
                if acf is None: continue
                acfs[d['kind']].append(acf)
                for tau, L in ((1.5, 12), (4.0, 24), (9.0, 40)):
                    G = gamma_from_acf(acf, L)
                    we = ema_weights(tau, L); wo = opt_weights(G, tau, L)
                    ve = we @ G @ we; vo = wo @ G @ wo
                    rows.append({'kind': d['kind'], 'tau': tau,
                                 'vr': ve/max(vo, 1e-12), 'rho1': acf[1]})
    for kind in ('top','bulk'):
        m = np.mean(np.stack(acfs[kind]), axis=0)
        print(f'mean ACF {kind}: ' + ' '.join(f'{v:+.3f}' for v in m))
    print()
    print('variance ratio EMA/optimal at matched mean lag (higher = filter wins):')
    print('kind  tau  n      median  q25    q75    frac>1.5x  frac>2x')
    import collections
    groups = collections.defaultdict(list)
    for r in rows: groups[(r['kind'], r['tau'])].append(r['vr'])
    for (kind, tau), v in sorted(groups.items()):
        v = np.array(v)
        print(f'{kind:5s} {tau:4.1f} {len(v):6d}  {np.median(v):6.3f} {np.quantile(v,.25):6.3f} {np.quantile(v,.75):6.3f}   {(v>1.5).mean():6.1%}   {(v>2).mean():6.1%}')

if __name__ == '__main__':
    main(sorted(glob.glob(sys.argv[1]))[:60])

def nesterov_weights(beta, L):
    # out = G_t + beta*buf_t, buf_t = sum beta^i G_{t-i}
    w = beta ** (np.arange(L) + 1.0)
    w[0] += 1.0
    return w / w.sum()

def mean_lag(w):
    return float(np.dot(w, np.arange(len(w))))

def rerun_vs_nesterov(files):
    import collections
    rows = collections.defaultdict(list)
    # match nesterov beta -> its mean lag, then compare optimal at SAME lag
    betas = (0.6, 0.85, 0.95)
    for f in files:
        side = json.load(open(f))
        for mat in side['matrices'].values():
            refresh = mat['refresh_steps']
            for d in mat['directions']:
                s = np.asarray(d['s'], float)
                if len(s) < 300: continue
                acf = window_acf(s, refresh)
                if acf is None: continue
                for beta in betas:
                    L = max(12, int(8*beta/(1-beta)))
                    wn = nesterov_weights(beta, L)
                    tau = mean_lag(wn)
                    G = gamma_from_acf(acf, L)
                    wo = opt_weights(G, tau, L)
                    vn = wn @ G @ wn; vo = wo @ G @ wo
                    rows[(d['kind'], beta)].append(vn/max(vo,1e-12))
    print('vs NESTEROV-EMA at matched mean lag:')
    print('kind  beta  tau_eff  n      median  q25    q75')
    for (kind, beta), v in sorted(rows.items()):
        v = np.array(v)
        L = max(12, int(8*beta/(1-beta)))
        tau = mean_lag(nesterov_weights(beta, L))
        print(f'{kind:5s} {beta:.2f} {tau:7.2f} {len(v):6d}  {np.median(v):6.3f} {np.quantile(v,.25):6.3f} {np.quantile(v,.75):6.3f}')

rerun_vs_nesterov(sorted(glob.glob('results/airbench_instrumented_seed14*.instrumentation.json'))[:60])

def decompose_gain(files):
    """How much of the vs-nesterov gain is kernel-family vs rho-matching?"""
    import collections
    rows = collections.defaultdict(list)
    for f in files:
        side = json.load(open(f))
        for mat in side['matrices'].values():
            refresh = mat['refresh_steps']
            for d in mat['directions']:
                s = np.asarray(d['s'], float)
                if len(s) < 300: continue
                acf = window_acf(s, refresh)
                if acf is None: continue
                for beta in (0.6, 0.95):
                    L = max(12, int(8*beta/(1-beta)))
                    wn = nesterov_weights(beta, L)
                    tau = mean_lag(wn)
                    G = gamma_from_acf(acf, L)
                    w_opt = opt_weights(G, tau, L)                       # matched to measured ACF
                    w_wht = opt_weights(np.eye(L), tau, L)               # white-noise kernel, same tau
                    vn, vo, vw = wn@G@wn, w_opt@G@w_opt, w_wht@G@w_wht
                    rows[(d['kind'], beta, 'nesterov/white')].append(vn/max(vw,1e-12))
                    rows[(d['kind'], beta, 'white/matched')].append(vw/max(vo,1e-12))
    print('gain decomposition (median variance ratios):')
    for k, v in sorted(rows.items()):
        v = np.array(v)
        print(f'  {k[0]:5s} beta={k[1]:.2f} {k[2]:15s}: {np.median(v):.3f}')

decompose_gain(sorted(glob.glob('results/airbench_instrumented_seed14*.instrumentation.json'))[:40])
