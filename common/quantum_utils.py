"""
Infraestrutura quântica para experimentos com PennyLane.
"""
from __future__ import annotations
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

def default_device(n_qubits=8, shots=None):
    return qml.device("default.qubit", wires=n_qubits, shots=shots)

def feature_map(x, wires):
    """Mapa de características simples com reupload de dados (codificação por ângulo)."""
    for i, w in enumerate(wires[: len(x)]):
        qml.RX(x[i], w)
        qml.RY(0.5 * x[i], w)
    # entrelaçamento fraco
    for i in range(len(wires)-1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

def hardware_efficient(params, wires):
    """Bloco HEA pequeno."""
    n = len(wires)
    # rotações de um qubit
    for i, w in enumerate(wires):
        a, b, c = params[i]
        qml.RZ(a, w); qml.RX(b, w); qml.RZ(c, w)
    # entrelaçamento em anel
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i+1) % n]])

def make_classifier(n_qubits=6, n_layers=2, shots=None):
    dev = default_device(n_qubits=n_qubits, shots=shots)
    wires = list(range(n_qubits))
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    @qml.qnode(dev)
    def circuit(x, weights):
        feature_map(x, wires)
        for l in range(n_layers):
            hardware_efficient(weights[l], wires)
        return qml.expval(qml.PauliZ(0))
    return circuit, weight_shapes, dev
