# Quantum Robotics Mini‑QHack (4 dias, Individual)

Um hackathon **individual** focado em **métodos quânticos para robótica**.  
Resolva 8 desafios de código auto‑contidos; cada um traz testes públicos nos notebooks e será reavaliado pelos organizadores em seeds ocultas.

## Trilhas & Desafios (8 no total)
- **Percepção & Sensoriamento**
  - A1 — Tipo de Terreno via LiDAR Minúsculo (Fácil)
  - A2 — Regressão de Deriva de IMU (Fácil)
  - A3 — Detecção de Evento de Escorregamento (Médio)
- **Planejamento & Mapeamento**
  - B1 — Micro‑TSP para Inspeção de Waypoints (Médio)
  - B3 — Seleção de Fechamentos de Loop via MWIS (Difícil)
- **Controle & Dinâmica**
  - C1 — IK Sensível a Energia via Objetivo Variacional (Médio)
  - C3 — Atualização de Pose em Servo Visual (Difícil)
- **Segurança & Sistemas**
  - D1 — Detecção de Anomalias em Telemetria (Médio)

## Formato
- **Entregável:** Edite o notebook de cada desafio em `challenges/`, implementando a função `solve(...)`.
- **Avaliação:** Execute os testes públicos presentes no final de cada notebook. Os organizadores rodarão testes adicionais com seeds/tamanhos ocultos para compor a pontuação oficial.
- **Pontuação alvo:** Fácil 100, Médio 200, Difícil 350 (critérios detalhados permanecem internos).
- **Critério geral:** superar vias clássicas e demonstrar uso quântico (feature map, circuitos variacionais, QAOA, etc.).
- **Limites sugeridos:** qubits ≤ 12; passos ≤ 150 (Fácil/Médio) ou ≤ 250 (Difícil); mantenha execução ágil.

## Política de Baseline & Quantum
- Baselines em [`common.baselines`](common/baselines.py) são apenas referência.
- Os testes internos aplicam thresholds mais rígidos e detectam reuso literal de baselines; foque em soluções próprias.
- Use utilidades de [`common.quantum_utils`](common/quantum_utils.py) ou diretamente PennyLane (`qml`) para evidenciar o componente quântico.
- Resuma sua abordagem em uma célula Markdown ao final de cada notebook (opcional, mas recomendado).

## Observação
Os scripts completos do avaliador e suas seeds permanecem privados e não são distribuídos aos participantes.

## Setup (pip)
Crie e ative um ambiente virtual e instale dependências via pip:

```bash
python -m venv venv
# Unix / macOS:
source venv/bin/activate
# Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## Execução local
```bash
# Abrir os notebooks
jupyter lab
```
Rode as células “Testes públicos” em cada notebook para validar a implementação.

## Estrutura de pastas
```
quantum-robotics-mini-qhack/
  challenges/             # 8 notebooks + adaptadores para import
  common/                 # utilidades compartilhadas (dados, quantum, baselines)
  requirements.txt
  README.md
```

## Referências de arquivos
- Utilidades quânticas: [`common.quantum_utils`](common/quantum_utils.py)
- Baselines clássicos: [`common.baselines`](common/baselines.py)
- Geração de dados: [`common.data_utils`](common/data_utils.py)
- Requisitos: [requirements.txt](requirements.txt)
