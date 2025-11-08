"""
Baselines clássicos para cada desafio.
"""
from __future__ import annotations
import numpy as np
from sklearn import svm, linear_model, metrics
import networkx as nx
import itertools

def baseline_terrain(Xtr, ytr, Xte):
    clf = svm.SVC(kernel="rbf", C=2.0, gamma="scale")
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)

def baseline_imu_reg(Xtr, ytr, Xte):
    # estatísticas simples da janela + ridge
    def feats(X):
        # X: (n, w, 6) -> média e desvio-padrão por canal
        mu = X.mean(axis=1); sd = X.std(axis=1)
        return np.concatenate([mu, sd], axis=1)
    Xtrf = feats(Xtr); Xtef = feats(Xte)
    reg = linear_model.Ridge(alpha=1.0)
    reg.fit(Xtrf, ytr)
    return reg.predict(Xtef)

def baseline_slip(Xtr, ytr, Xte):
    # SVM linear raso em características de potência da FFT (bem grosseiro)
    def feats(X):
        # (n, w, c) -> potência de banda dos 3 primeiros canais + inclinação do canal de corrente
        n, w, c = X.shape
        f = np.abs(np.fft.rfft(X[:, :, 0:3], axis=1)).mean(axis=1)  # (n, fft_bins, 3) -> (n,3)
        slope = (X[:, -1, 7] - X[:, 0, 7]).reshape(-1,1)
        return np.concatenate([f, slope], axis=1)
    Xtrf = feats(Xtr); Xtef = feats(Xte)
    clf = svm.LinearSVC()
    clf.fit(Xtrf, ytr)
    return clf.decision_function(Xtef)

def baseline_tsp(D):
    # greedy nearest-neighbor e depois melhoria 2-opt (instâncias pequenas)
    n = D.shape[0]
    unvisited = set(range(n))
    tour = [0]; unvisited.remove(0)
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt); unvisited.remove(nxt); cur = nxt
    # 2-opt
    def length(t):
        return sum(D[t[i], t[(i+1)%n]] for i in range(n))
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                new = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
                if length(new) + 1e-9 < length(tour):
                    tour = new; improved = True
    return tour

def baseline_mwis(G, w):
    # guloso por peso/(1+grau)
    order = sorted(G.nodes(), key=lambda u: w[u]/(1+G.degree[u]), reverse=True)
    chosen = []
    forbidden = set()
    for u in order:
        if u in forbidden: 
            continue
        chosen.append(u)
        forbidden.update([u] + list(G.neighbors(u)))
    return sorted(chosen)

def baseline_visual_servo(X1, X2):
    # Estimativa linearizada robusta com escore simples de inliers (proxy)
    # Aqui usamos apenas a mediana dos deltas par-a-par como estimativa simplificada.
    d = X2 - X1
    tx = np.median(d[:,:,0], axis=1)
    ty = np.median(d[:,:,1], axis=1)
    rot = np.zeros_like(tx)  # ignora rotação no baseline
    return np.stack([rot, tx, ty], axis=1)

def baseline_anomaly_threshold(X):
    # limiar por Z-score com tau=3
    z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    score = np.abs(z).max(axis=1)
    return score

def baseline_energy_aware_batch(targets, prev, steps=200, lr=0.03, lam=0.05):
    """Baseline de IK por descida de gradiente, mantido para benchmarking."""
    def fk(th):
        L = [1.0, 1.0, 1.0]
        x = sum(L[i] * np.cos(np.sum(th[: i + 1])) for i in range(3))
        y = sum(L[i] * np.sin(np.sum(th[: i + 1])) for i in range(3))
        return np.array([x, y])
    angles = []
    for target, start in zip(targets, prev):
        th = start.copy()
        for _ in range(steps):
            eps = 1e-3
            grad = np.zeros(3)
            for i in range(3):
                e = np.zeros(3); e[i] = eps
                f1 = np.linalg.norm(fk(th + e) - target) ** 2 + lam * np.linalg.norm((th + e) - start) ** 2
                f0 = np.linalg.norm(fk(th) - target) ** 2 + lam * np.linalg.norm(th - start) ** 2
                grad[i] = (f1 - f0) / eps
            th -= lr * grad
        angles.append(th)
    return np.stack(angles, axis=0)
