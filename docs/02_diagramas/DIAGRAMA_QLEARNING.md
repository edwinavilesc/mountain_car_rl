# Diagrama de Arquitectura ~ Ciclo de Entrenamiento Q-Learning

> **Entregable:** Esquema propio del proceso de entrenamiento de Q-Learning tabular
> sobre MountainCar-v0, capturando el ciclo completo:
> estado → acción → recompensa → actualización de la Q-Table.

---

## Arquitectura General ~ Q-Learning Tabular en MountainCar-v0

```mermaid
flowchart LR
    subgraph ENV["🏔️ Entorno ~ MountainCar-v0"]
        direction TB
        E1["Estado Continuo\nposición ∈ [-1.2, 0.6]\nvelocidad ∈ [-0.07, 0.07]"]
        E2["Recompensa\nr = -1 por paso"]
        E3["Condición de término\nposición ≥ 0.5\nO 200 pasos"]
    end

    subgraph DISC["🔢 Discretización"]
        direction TB
        D1["np.digitize(posición, bordes)\n→ índice 0..19"]
        D2["np.digitize(velocidad, bordes)\n→ índice 0..19"]
        D3["Estado discreto\n= (idx_pos, idx_vel)\n400 estados totales"]
        D1 --> D3
        D2 --> D3
    end

    subgraph AGENT["🤖 Agente Q-Learning"]
        direction TB
        subgraph POL["Política ε-greedy"]
            P1{random < ε?}
            P2["Exploración\nacción aleatoria\n0, 1 o 2"]
            P3["Explotación\nargmax Q-Table-estado"]
            P1 -- SI --> P2
            P1 -- NO --> P3
        end

        subgraph QTBL["Q-Table ~ Memoria del Agente"]
            Q1["Estado (0,0) → [Q₀, Q₁, Q₂]"]
            Q2["Estado (7,14) → [-152, -148, -99]"]
            Q3["Estado (10,9) → [Q₀, Q₁, Q₂]"]
            Q4["... 400 estados posibles ..."]
        end

        subgraph UPD["Actualización TD ~ Temporal Difference"]
            U1["target = r + γ · max Q(s')"]
            U2["error = target - Q(s, a)"]
            U3["Q(s, a) += α · error"]
            U1 --> U2 --> U3
        end
    end

    ENV -- "obs: [pos, vel]" --> DISC
    DISC -- "state: (idx, idx)" --> POL
    P2 -- "acción a" --> ENV
    P3 -- "acción a" --> ENV
    ENV -- "r, s', terminated" --> UPD
    DISC -- "state, next_state" --> UPD
    UPD -- "actualiza tabla" --> QTBL
    QTBL -- "Q-values" --> POL
```

---

## Detalle ~ La Q-Table y el proceso de actualización

```mermaid
sequenceDiagram
    actor ENV as 🏔️ MountainCar-v0
    participant DISC as 🔢 Discretizer
    participant AGENT as 🤖 Q-Learning Agent
    participant QTBL as 📋 Q-Table
    participant POL as 🎲 Política ε-greedy

    Note over ENV,POL: Inicio del episodio

    ENV->>DISC: obs = [-0.48, 0.002]
    DISC->>AGENT: state = (7, 14)
    AGENT->>QTBL: ¿Qué sé sobre (7, 14)?
    QTBL->>POL: Q[(7,14)] = [-152.3, -148.1, -98.7]
    POL->>ENV: acción = 2 (→ derecha)

    ENV->>AGENT: r = -1, next_obs = [-0.47, 0.004], terminated = False
    AGENT->>DISC: discretizar next_obs
    DISC->>AGENT: next_state = (7, 15)
    AGENT->>QTBL: ¿Qué sé sobre (7, 15)?
    QTBL->>AGENT: Q[(7,15)] = [-120.5, -135.0, -88.7]

    Note over AGENT: Cálculo del target:
    Note over AGENT: best_next_q = max(-120.5, -135.0, -88.7) = -88.7
    Note over AGENT: target = -1 + 0.99 * (-88.7) = -88.813
    Note over AGENT: error  = -88.813 - (-98.7) = 9.887
    Note over AGENT: Q[(7,14)][2] += 0.1 * 9.887 → -98.7 + 0.989 = -97.711

    AGENT->>QTBL: Q[(7,14)][2] = -97.711
    Note over ENV,POL: Continúa hasta terminated or truncated
```

---

## El proceso de aprendizaje a lo largo del tiempo

```mermaid
xychart-beta
    title "Evolución de la Recompensa ~ Q-Learning en 20,000 episodios"
    x-axis ["0", "2k", "4k", "6k", "8k", "10k", "12k", "14k", "16k", "18k", "20k"]
    y-axis "Recompensa Promedio" -210 --> -80
    line [-200, -200, -199, -195, -180, -160, -145, -140, -137, -135, -133]
```

> **Nota:** La curva real se generará con matplotlib después del entrenamiento
> y se guardará en `docs/03_resultados/plots/`.

---

## Hiperparámetros de entrenamiento ~ Q-Learning

| Parámetro | Símbolo | Valor | Justificación |
|---|:---:|:---:|---|
| Número de bins por dimensión | n_bins | 20 | 400 estados totales: granularidad óptima |
| Tasa de aprendizaje | α | 0.10 | Actualización conservadora para estabilidad |
| Factor de descuento | γ | 0.99 | Recompensas futuras casi tan valiosas como inmediatas |
| Epsilon inicial | ε₀ | 1.00 | Exploración total al inicio (el agente no sabe nada) |
| Epsilon mínimo | ε_min | 0.01 | Siempre mantiene 1% de exploración residual |
| Decaimiento de epsilon | ε_decay | 0.9995 | Decaimiento lento: llega a ε_min en ~20,000 episodios |
| Episodios de entrenamiento | T | 20,000 | Convergencia típica según documentación del proyecto |
