"""
Geradores sintéticos de dados para os desafios.
Todos os geradores aceitam um `seed` para determinismo.
"""
from __future__ import annotations
import numpy as np
import networkx as nx

RNG = np.random.default_rng

def terrain_dataset(n=300, d=12, seed=1337):
    """Pequenos vetores de características LiDAR para 3 classes: smooth, gravel, grass.
    Retorna X (n x d), y (n,)
    """
    rng = RNG(seed)
    n_por_classe = n // 3
    # Médias e variâncias base por classe
    mus = np.stack([
        np.linspace(0.1, 0.6, d),          # smooth
        np.linspace(0.4, 1.0, d),          # gravel
        np.linspace(0.2, 0.8, d),          # grass
    ])
    sigs = np.stack([
        np.linspace(0.02, 0.05, d),
        np.linspace(0.04, 0.10, d),
        np.linspace(0.03, 0.08, d),
    ])
    X = []
    y = []
    for c in range(3):
        Xi = mus[c] + sigs[c] * rng.standard_normal((n_por_classe, d))
        X.append(Xi)
        y.append(np.full(n_por_classe, c, int))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # Pequena permutação
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def imu_drift_dataset(n=400, w=32, seed=1337):
    """Janelas IMU (6 eixos); alvo é magnitude do bias do giroscópio em rad/s."""
    rng = RNG(seed)
    # Simula bias verdadeiro por amostra
    bias = 0.01 + 0.02 * rng.random(n)  # 0.01..0.03
    X = []
    for i in range(n):
        # 6 eixos: [ax, ay, az, gx, gy, gz] ao longo da janela w
        base = rng.standard_normal((w, 6)) * 0.1
        # Adiciona bias aos canais de giroscópio
        base[:, 3:] += bias[i]
        X.append(base)
    X = np.stack(X, axis=0)  # (n, w, 6)
    y = bias
    return X, y

def slip_detection_dataset(n=600, w=64, seed=42, pos_rate=0.3):
    """Classificação binária de slip a partir de trechos tácteis + corrente de motor (8 canais)."""
    rng = RNG(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.standard_normal((n, w, 8)) * 0.2
    # Injeta assinaturas de slip: pulsos nos canais 0..2 e corrente crescente (canal 7)
    for i in range(n):
        if y[i] == 1:
            t0 = rng.integers(5, w-10)
            X[i, t0:t0+8, 0:3] += 0.8 + 0.4 * rng.random((8,3))
            X[i, :, 7] += np.linspace(0, 0.5, w) + 0.1 * rng.random(w)
    return X, y

def micro_tsp_instance(n_nodes=8, grid=10, seed=7):
    """Pequeno TSP com obstáculos como células proibidas; retorna coords, máscara de obstáculos, matriz de distâncias."""
    rng = RNG(seed)
    coords = rng.integers(0, grid, size=(n_nodes, 2))
    # Grade de obstáculos ~10%
    obstacles = rng.random((grid, grid)) < 0.10
    # Distância euclidiana (ignora obstáculos para a métrica)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return coords, obstacles, D

def mwis_loop_closure_instance(n=50, p_edge=0.10, seed=11):
    """Grafo de conflitos aleatório para candidatos de loop-closure com pesos."""
    rng = RNG(seed)
    G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    w = rng.random(n) + 0.5  # 0.5..1.5
    # garante ausência de self-loops
    for u in range(n):
        if G.has_edge(u, u):
            G.remove_edge(u, u)
    return G, w

def energy_aware_ik_dataset(n=200, seed=5):
    """Alvos (x,y) e estado articular anterior para braço planar 3-DOF (links 1,1,1)."""
    rng = RNG(seed)
    targets = rng.uniform(low=[0.1, -2.5], high=[2.8, 2.5], size=(n,2))
    prev = rng.uniform(low=-np.pi, high=np.pi, size=(n,3))
    return targets, prev

def visual_servo_dataset(n=200, m=24, seed=77, outlier_rate=0.2):
    """Correspondências 2D-2D (coords normalizados), com outliers."""
    rng = RNG(seed)
    # Delta pose verdadeira: pequena rotação + translação
    true_rot = 0.05 * (rng.random(n) - 0.5)  # radianos em torno de z (proxy)
    true_tx = 0.02 * (rng.random(n) - 0.5)
    true_ty = 0.02 * (rng.random(n) - 0.5)
    X1 = rng.normal(0, 0.7, size=(n, m, 2))
    # Aplica delta para obter X2 (modelo linearizado simples)
    X2 = X1.copy()
    for i in range(n):
        theta = true_rot[i]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        X2[i] = (X1[i] @ R.T) + np.array([true_tx[i], true_ty[i]])
        # Adiciona ruído
        X2[i] += 0.01 * rng.standard_normal((m,2))
        # Injeta outliers
        k = int(outlier_rate * m)
        idx = rng.choice(m, size=k, replace=False)
        X2[i, idx] = rng.normal(0, 2.0, size=(k,2))
    y = np.stack([true_rot, true_tx, true_ty], axis=1)
    return X1, X2, y

def telemetry_anomaly_dataset(n=1000, d=8, seed=99, anomaly_rate=0.03):
    """Telemetria multivariada com anomalias raras."""
    rng = RNG(seed)
    X = rng.normal(0, 1.0, size=(n, d))
    y = np.zeros(n, dtype=int)
    k = int(anomaly_rate * n)
    idx = rng.choice(n, size=k, replace=False)
    X[idx] += rng.normal(3.0, 0.5, size=(k, d))
    y[idx] = 1
    return X, y
