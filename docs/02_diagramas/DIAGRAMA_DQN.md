# Diagrama de Arquitectura ~ Ciclo de Entrenamiento DQN

> **Entregable:** Esquema propio del proceso de entrenamiento de DQN ~ "Deep Q-Network ~
> Red Q Profunda" sobre MountainCar-v0, capturando el ciclo completo:
> replay buffer, target network y actualización de Bellman.

---

## Arquitectura General ~ DQN en MountainCar-v0

```mermaid
flowchart TB
    subgraph ENV["🏔️ Entorno ~ MountainCar-v0"]
        E1["Observación continua\nobs = [posición, velocidad]\n2 floats, sin discretizar"]
    end

    subgraph POLICY["🎲 Política ~ Sticky ε-greedy"]
        direction LR
        P1{random < ε?}
        P2{random < p_sticky?}
        P3["Repite last_action\nCorrelación temporal"]
        P4["Nueva acción aleatoria\nrandom 0-1 o 2"]
        P5["Explotación\nargmax Q_net(obs)"]
        P1 -- SI ~ Exploración --> P2
        P2 -- SI --> P3
        P2 -- NO --> P4
        P1 -- NO --> P5
    end

    subgraph BUFFER["💾 ReplayBuffer ~ Memoria de Experiencias"]
        direction TB
        B1["(s, a, r, s', terminated)\n(s, a, r, s', terminated)\n...\nCapacidad: 100,000 transiciones\nFIFO ~ elimina la más antigua"]
    end

    subgraph LEARN["🧠 Paso de Aprendizaje ~ cada paso"]
        direction TB
        L1["Sampleo 64 transiciones\naleatorias del buffer"]
        L2["current_q = Q_net(s).gather(a)\nshape: 64 x 1"]
        L3["next_q = Target_net(s').max()\nshape: 64 x 1 - SIN gradiente"]
        L4["target_q = r + γ·next_q·(1-done)\nEcuación de Bellman - shape 64 x 1"]
        L5["loss = MSE(current_q, target_q)"]
        L6["zero_grad → backward → step\nActualizo pesos de Q_net"]
        L1 --> L2 --> L3 --> L4 --> L5 --> L6
    end

    subgraph NETS["🔮 Redes Neuronales"]
        direction LR
        subgraph ONLINE["Q_net ~ Red Online - se entrena"]
            N1["Linear(2→128)\nReLU\nLinear(128→128)\nReLU\nLinear(128→3)"]
        end
        subgraph TARGET["Target_net ~ Red Congelada"]
            N2["Linear(2→128)\nReLU\nLinear(128→128)\nReLU\nLinear(128→3)"]
        end
        ONLINE -. "cada 10 episodios:\ntarget_net ← q_net" .-> TARGET
    end

    ENV -- "obs" --> POLICY
    POLICY -- "acción a" --> ENV
    ENV -- "obs, a, r, next_obs, terminated" --> BUFFER
    BUFFER -- "mini-batch 64" --> LEARN
    LEARN --> ONLINE
    ONLINE -- "Q-values para explotar" --> POLICY
    TARGET -- "Q-values para target" --> LEARN
```

---

## Detalle ~ La arquitectura de la QNetwork

```mermaid
flowchart LR
    subgraph INPUT["Entrada"]
        I1["posición\n-1.2 → 0.6"]
        I2["velocidad\n-0.07 → 0.07"]
    end

    subgraph H1["Capa Oculta 1"]
        direction TB
        H1L["Linear(2 → 128)\n128 neuronas"]
        H1R["ReLU\nf(x) = max(0, x)"]
        H1L --> H1R
    end

    subgraph H2["Capa Oculta 2"]
        direction TB
        H2L["Linear(128 → 128)\n128 neuronas"]
        H2R["ReLU\nf(x) = max(0, x)"]
        H2L --> H2R
    end

    subgraph OUTPUT["Salida ~ Q-values"]
        direction TB
        O1["Linear(128 → 3)\nSin activación final"]
        O2["Q(s, ←) acción 0"]
        O3["Q(s, ~) acción 1"]
        O4["Q(s, →) acción 2"]
        O1 --> O2
        O1 --> O3
        O1 --> O4
    end

    INPUT --> H1
    H1 --> H2
    H2 --> OUTPUT

    subgraph PARAMS["Parámetros totales"]
        PA["2×128 + 128 = 384\n128×128 + 128 = 16,512\n128×3 + 3 = 387\nTotal: 17,283 parámetros"]
    end
```

---

## El ReplayBuffer ~ Cómo funciona la memoria de experiencias

```mermaid
sequenceDiagram
    actor ENV as 🏔️ MountainCar-v0
    participant AGENT as 🤖 DQN Agent
    participant BUF as 💾 ReplayBuffer
    participant QNET as 🧠 Q_net (online)
    participant TNET as 🔮 Target_net (congelada)

    Note over ENV,TNET: Episodio en progreso (fase de exploración)

    ENV->>AGENT: obs = [-0.48, 0.002]
    AGENT->>AGENT: sticky ε-greedy → acción = 2
    AGENT->>ENV: env.step(2)
    ENV->>AGENT: r=-1, next_obs=[-0.47, 0.004], terminated=False, truncated=False
    AGENT->>BUF: push([-0.48,0.002], 2, -1.0, [-0.47,0.004], False)
    Note over BUF: buffer ahora tiene 1 transición (necesita 64 para aprender)

    Note over ENV,TNET: Después de 64+ transiciones en el buffer

    AGENT->>BUF: sample(64)
    BUF->>AGENT: 64 transiciones aleatorias

    AGENT->>QNET: Q_net(states_t) → shape (64, 3)
    QNET->>AGENT: .gather(1, actions_t) → current_q shape (64, 1)

    AGENT->>TNET: Target_net(next_states_t) → shape (64, 3)
    Note over TNET: with torch.no_grad() ~ SIN gradientes
    TNET->>AGENT: .max(dim=1, keepdim=True).values → next_q shape (64, 1)

    AGENT->>AGENT: target_q = r + 0.99 * next_q * (1 - terminated)
    AGENT->>AGENT: loss = MSE(current_q, target_q)
    AGENT->>QNET: backward() + step() → actualizo pesos

    Note over ENV,TNET: Cada 10 episodios

    QNET->>TNET: target_net.load_state_dict(q_net.state_dict())
    Note over TNET: Sincronizo la red congelada con la red online
```

---

## El Bug de Exploración y su Fix ~ Correlación Temporal

```mermaid
flowchart LR
    subgraph BAD["❌ Exploración Uniforme - NO funciona"]
        B1["Paso 1: acción aleatoria = 0 ←"]
        B2["Paso 2: acción aleatoria = 2 →"]
        B3["Paso 3: acción aleatoria = 0 ←"]
        B4["Paso 4: acción aleatoria = 2 →"]
        B5["Resultado: movimiento\ncaótico, no acumula\nmomentum"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph GOOD["✅ Sticky Exploration ~ p_sticky = 0.9"]
        G1["Paso 1: nueva acción = 2 →"]
        G2["Paso 2: repite 2 → (90%)"]
        G3["Paso 3: repite 2 → (90%)"]
        G4["Paso 4: repite 2 → (90%)"]
        G5["Resultado: carreras\nsostenidas, acumula\nmomentum para subir"]
        G1 --> G2 --> G3 --> G4 --> G5
    end

    subgraph PROB["📊 Probabilidad de éxito"]
        P1["Uniforme:\n(1/3)^20 ≈ 3×10^-10\n≈ imposible"]
        P2["Sticky p=0.9:\n0.9^19 ≈ 13.5%\n¡Alcanzable!"]
    end
```

---

## Hiperparámetros de entrenamiento ~ DQN

| Parámetro | Símbolo | Valor | Justificación |
|---|:---:|:---:|---|
| Tasa de aprendizaje | α | 0.001 | Adam optimizer ~ más pequeño que Q-Learning tabular |
| Factor de descuento | γ | 0.99 | Mismo que Q-Learning para comparación justa |
| Epsilon inicial | ε₀ | 1.00 | Exploración total al inicio |
| Epsilon mínimo | ε_min | 0.01 | Mantiene exploración residual |
| Decaimiento de epsilon | ε_decay | 0.995 | Más rápido que Q-Learning (menos episodios totales) |
| Tamaño del mini-batch | B | 64 | Balance estabilidad/velocidad estándar en DQN |
| Capacidad del buffer | N | 100,000 | Suficiente diversidad de experiencias |
| Frecuencia sync target | f | 10 episodios | Estabilidad sin desactualización excesiva |
| Neuronas por capa | h | 128 | Red pequeña suficiente para espacio 2D |
| Probabilidad sticky | p_sticky | 0.9 | Fix de exploración ~ carreras sostenidas de ~10 pasos |
| Episodios de entrenamiento | T | 2,500 | Convergencia esperada según documentación |
