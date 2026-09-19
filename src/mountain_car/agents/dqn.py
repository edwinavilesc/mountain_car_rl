"""
DQN ~ "Deep Q-Network ~ Red Q-Profunda" implementado en PyTorch.

Evito deliberadamente librerías de RL ~ "Reinforcement Learning" de alto nivel
para que cada pieza del algoritmo sea visible y comprensible para fines educativos.

Componentes principales:
  ~ QNetwork    : MLP ~ "Multi-Layer Perceptron" que mapea estado → Q(s, a)
  ~ ReplayBuffer: almacena transiciones (s, a, r, s', terminated) para replay
  ~ DQNAgent    : ciclo de entrenamiento, política ε-greedy con sticky action,
                  sincronización de target net ~ "red objetivo congelada"

Corrección del Ejercicio 3 ~ Sticky Action Exploration:
  La exploración uniforme (elegir acción random independiente en cada paso) nunca
  produce las carreras sostenidas que MountainCar necesita para subir la colina.
  P(20 empujes sostenidos con exploración uniforme) = (1/3)^20 ≈ 3 × 10^-10.
  Implementé exploración con correlación temporal ~ "temporal correlation" mediante
  sticky action ~ "acción pegajosa": con probabilidad p_sticky repito la acción
  anterior, permitiendo que P(20 empujes sostenidos) ≈ 0.9^19 ≈ 13.5%.
"""

import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim


# ── Red neuronal ~ QNetwork ────────────────────────────────────────────


class QNetwork(nn.Module):
    """Red neuronal que aproxima la función Q ~ "función de valor-acción".

    ~ Ejercicio 2a ~

    Arquitectura MLP ~ "Multi-Layer Perceptron ~ Perceptrón Multicapa":
        state_dim → Linear → ReLU → Linear → ReLU → action_dim

    Importante ~ sin activación en la capa de salida:
      Los Q-values son números reales negativos (de ~-200 a ~-100 en MountainCar).
      Si usara Softmax o Sigmoid rompería esa propiedad ~ los Q-values no son
      probabilidades ni están normalizados en ningún rango fijo.

    ReLU ~ "Rectified Linear Unit ~ Unidad Lineal Rectificada ~ f(x) = max(0, x)":
      Introduce no-linealidad para que la red aprenda patrones complejos del entorno.
      Sin ReLU, apilar capas lineales equivale a una sola capa linear (no sirve).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        # SIEMPRE llamar super().__init__() primero en cualquier nn.Module ~
        # sin esto PyTorch no puede registrar ni acceder a los parámetros de la red
        super().__init__()

        # ── Construyo la red como Sequential ~ "secuencial ~ capas aplicadas en orden"
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),  # Capa 1: state_dim(2) → hidden(128)
            nn.ReLU(),                     # Activación ~ introduce no-linealidad
            nn.Linear(hidden, hidden),     # Capa 2: hidden(128) → hidden(128)
            nn.ReLU(),                     # Segunda activación no-lineal
            nn.Linear(hidden, action_dim), # Capa de salida: hidden(128) → action_dim(3)
            # SIN activación final ~ los Q-values son reales sin restricción de rango
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante ~ "forward pass" de la red.

        Recibe un batch ~ "lote" de estados shape (B, state_dim) y retorna
        los Q-values de todas las acciones shape (B, action_dim).
        """
        return self.network(x)


# ── Buffer de repetición ~ ReplayBuffer ────────────────────────────────


class ReplayBuffer:
    """Buffer FIFO ~ "First In First Out ~ cola de capacidad fija".

    Almaceno transiciones (s, a, r, s', terminated) para experience replay ~
    "repetición de experiencias ~ aprender de experiencias pasadas en mini-batches
    aleatorios para romper la correlación temporal de los datos consecutivos".

    deque(maxlen) ~ "double-ended queue ~ cola de doble extremo": descarta
    automáticamente el elemento más antiguo cuando se llena la capacidad.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        # deque con maxlen ~ FIFO automático cuando capacity se alcanza
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Agrego una transición al buffer."""
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> list[tuple]:
        """Sampleo ~ "muestreo aleatorio sin reemplazo" de batch_size transiciones."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agente DQN ─────────────────────────────────────────────────────────


class DQNAgent:
    """
    Agente DQN ~ "Deep Q-Network ~ Red Q-Profunda" implementado desde cero en PyTorch.

    Implementé las siguientes características sobre el pseudocódigo base:
      ~ Experience replay   : mini-batches aleatorios del ReplayBuffer para romper
                              correlación temporal ~ "temporal correlation"
      ~ Target network      : red congelada para estabilizar el cálculo del target
                              de Bellman ~ "Bellman target ~ objetivo de aprendizaje"
      ~ Sticky exploration  : correlación temporal en exploración para MountainCar
                              (Ejercicio 3 ~ fix del bug de exploración uniforme)
      ~ Terminated vs done  : distinción crítica para el target de Bellman correcto
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
        p_sticky: float = 0.9,  # Ejercicio 3 ~ P de repetir acción anterior
    ) -> None:
        # ── Hiperparámetros ~ "hyperparameters ~ parámetros que controlan el aprendizaje"
        self.env_id = env_id
        self.lr = lr                           # α ~ "learning rate ~ tasa de aprendizaje"
        self.gamma = gamma                     # γ ~ "discount factor ~ factor de descuento"
        self.epsilon = epsilon_start           # ε ~ exploración inicial
        self.epsilon_end = epsilon_end         # ε mínimo ~ piso de exploración
        self.epsilon_decay = epsilon_decay     # Decaimiento multiplicativo por episodio
        self.batch_size = batch_size           # B ~ tamaño del mini-batch para aprender
        self.buffer_capacity = buffer_capacity # N ~ capacidad máxima del ReplayBuffer
        self.target_update_freq = target_update_freq  # Frecuencia de sync ~ "sincronización"
        self.hidden = hidden                   # h ~ neuronas por capa oculta
        self.p_sticky = p_sticky               # Ejercicio 3 ~ probabilidad de sticky action
        self.training_episodes = 0             # Contador de episodios entrenados
        self.rewards_history: list[float] = [] # Historial acumulado para graficar

        # ── Estado interno de sticky action ~ se resetea al inicio de cada episodio
        # Ejercicio 3 requiere un estado por-episodio para la correlación temporal
        self._last_action: int | None = None

        # ── Obtengo dimensiones del entorno
        env = gym.make(env_id)
        self.state_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        self.action_dim = int(env.action_space.n)             # type: ignore[attr-defined]
        env.close()

        # ── Dispositivo ~ "device ~ hardware donde viven los tensores y la red"
        # Detecta GPU NVIDIA automáticamente via CUDA ~ "Compute Unified Device
        # Architecture ~ plataforma de computación paralela de NVIDIA"
        # Para DQN en MountainCar la red es tiny ~ la GPU puede ser igual o más lenta
        # que CPU por el overhead de kernels ~ pero la detección automática es correcta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Red online ~ "q_net ~ se entrena en cada paso de gradiente"
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)

        # ── Red objetivo ~ "target_net ~ congelada, sincronizada cada N episodios"
        # Propósito: estabilizar el cálculo del target de Bellman ~ si usara q_net
        # para ambos (current y target), el blanco se movería en cada paso de gradiente
        # haciendo el entrenamiento inestable (como tratar de atrapar la propia sombra)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())  # Pesos idénticos al inicio

        # ── Optimizador Adam ~ "Adaptive Moment Estimation ~ tasa de aprendizaje adaptativa"
        # Solo optimizo los parámetros de q_net ~ target_net permanece congelada
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # ── Función de pérdida MSE ~ "Mean Squared Error ~ Error Cuadrático Medio"
        self.loss_fn = nn.MSELoss()

        # ── Buffer de experiencias para experience replay ~ "repetición de experiencias"
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── política ~ action selection ────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Selecciono una acción con exploración sticky ε-greedy.

        ~ Ejercicio 3 ~ Fix de exploración con correlación temporal ~

        Problema original: la exploración uniforme (acción random independiente en
        cada paso) nunca produce las carreras sostenidas de empujes que MountainCar
        necesita para acumular momentum ~ "impulso ~ energía cinética acumulada".

        Solución ~ sticky action ~ "acción pegajosa":
          ~ Con prob ε       : EXPLORACIÓN
              ~ Con prob p_sticky: repito self._last_action (correlación temporal)
              ~ Con prob 1-p_sticky: nueva acción aleatoria
          ~ Con prob (1-ε)   : EXPLOTACIÓN ~ argmax(q_net(state))

        P(20 empujes sostenidos con p_sticky=0.9) = 0.9^19 ≈ 13.5%
        P(20 empujes sostenidos sin sticky)       = (1/3)^20 ≈ 3×10^-10

        Cuando deterministic=True: siempre exploto ~ modo evaluación puro.
        La exploración pertenece SOLO al entrenamiento.
        """
        # ── Modo explotación o evaluación determinística
        if deterministic or random.random() >= self.epsilon:
            with torch.no_grad():
                # Convierto numpy array a tensor con batch dimension ~ shape (1, state_dim)
                t = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                # argmax ~ "índice del Q-value máximo ~ la mejor acción según la red"
                action = int(self.q_net(t).argmax(dim=1).item())
        else:
            # ── Modo exploración con sticky action ~ correlación temporal
            if self._last_action is not None and random.random() < self.p_sticky:
                # Repito la acción anterior ~ construyo carreras sostenidas de momentum
                action = self._last_action
            else:
                # Nueva acción aleatoria uniforme
                action = random.randrange(self.action_dim)

        # ── Actualizo la última acción tomada (solo en modo exploración)
        # En modo determinístico no contamino el estado de sticky action
        if not deterministic:
            self._last_action = action

        return action

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        """Interfaz de predicción compatible con la CLI ~ "Command Line Interface"."""
        return self.select_action(obs, deterministic=deterministic), None

    # ── paso de aprendizaje ~ learning step ────────────────────────────

    def _tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        """Convierto datos Python/numpy a tensor PyTorch en el dispositivo correcto."""
        return torch.as_tensor(np.array(x), dtype=dtype, device=self.device)

    def _learn(self) -> float:
        """Sampleo un mini-batch del buffer y ejecuto un paso de gradiente.

        ~ Ejercicio 2b ~

        Implemento el algoritmo DQN con cuatro pasos clave:
          1. current_q  ~ Q(s, a) de la red ONLINE para las acciones tomadas
          2. next_q     ~ max Q_target(s', a') de la red CONGELADA (sin gradiente)
          3. target_q   ~ objetivo de Bellman: r + γ·next_q·(1-terminated)
          4. Paso de gradiente: zero_grad → backward → step

        SHAPES CRÍTICOS ~ todos deben ser (B, 1) = (64, 1):
          Si algún tensor queda con shape (B,) en vez de (B, 1), PyTorch hace
          broadcast silencioso ~ "silent broadcast bug" que no lanza error pero
          entrena sobre resultados sin sentido. Verifico con .unsqueeze(1) y
          keepdim=True.

        Retorno el valor escalar de la pérdida para monitoreo.
        """
        # ── No aprendo hasta tener suficientes experiencias en el buffer
        if len(self.buffer) < self.batch_size:
            return 0.0

        # ── Sampleo mini-batch aleatorio ~ rompe correlación temporal de datos consecutivos
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        # zip(*batch) ~ "desempaqueta" las B tuplas en 5 listas separadas

        # ── Convierto a tensores con shapes correctos
        states_t      = self._tensor(states)                             # (B, state_dim) = (64, 2)
        actions_t     = self._tensor(actions, torch.int64).unsqueeze(1) # (B, 1) int64 para .gather
        rewards_t     = self._tensor(rewards).unsqueeze(1)              # (B, 1) float32
        next_states_t = self._tensor(next_states)                       # (B, state_dim) = (64, 2)
        terminateds_t = self._tensor(terminateds).unsqueeze(1)          # (B, 1) float32

        # ── 1. current_q ~ Q-values de la red ONLINE para las acciones realmente tomadas
        # q_net(states_t) → (B, action_dim) = (64, 3)
        # .gather(1, actions_t) → (B, 1): selecciona, por fila, el Q-value de la acción tomada
        current_q = self.q_net(states_t).gather(1, actions_t)

        # ── 2. next_q ~ máximos Q-values de la red CONGELADA sin propagar gradientes
        # torch.no_grad() ~ "no gradient ~ no calculo derivadas" ~ target_net es estática
        with torch.no_grad():
            # target_net(next_states_t) → (B, 3)
            # .max(dim=1, keepdim=True).values → (B, 1) ~ keepdim=True mantiene shape (B, 1)
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values

        # ── 3. target_q ~ objetivo de Bellman ~ "Bellman target"
        # terminated=0: target = r + γ * max_Q(s')   ~ el episodio continúa
        # terminated=1: target = r + 0               ~ llegué a la bandera, no hay futuro
        # (1.0 - terminateds_t) zeroes ~ "anula" el bootstrap branchlessly ~ "sin condicionales"
        target_q = rewards_t + self.gamma * next_q * (1.0 - terminateds_t)

        # ── 4. Paso de gradiente ~ "gradient step ~ actualización de pesos"
        loss = self.loss_fn(current_q, target_q)  # MSE entre (B, 1) y (B, 1) ~ shapes iguales
        self.optimizer.zero_grad()                 # Limpio gradientes acumulados del paso anterior
        loss.backward()                            # Backpropagation ~ computo ∂Loss/∂W para q_net
        self.optimizer.step()                      # Adam actualiza los pesos de q_net

        return loss.item()  # Retorno el valor escalar de la pérdida para monitoreo

    # ── ciclo de entrenamiento ~ training loop ──────────────────────────

    def train(self, total_episodes: int = 500, log_interval: int = 10) -> list[float]:
        """Ciclo completo de entrenamiento del agente DQN.

        En cada episodio:
          1. Exploro con sticky ε-greedy ~ acumulo experiencias en el buffer
          2. Aprendo de un mini-batch aleatorio del buffer en cada paso
          3. Sincronizo target_net ← q_net cada `target_update_freq` episodios
          4. Decaigo ε para reducir exploración gradualmente
        """
        env = gym.make(self.env_id)
        rewards_history: list[float] = []  # Recompensas de esta sesión

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            # ── Reset del estado de sticky action ~ EJERCICIO 3 lo requiere explícitamente
            # Cada episodio empieza sin acción previa ~ no contamino entre episodios
            self._last_action = None
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs)  # Sticky ε-greedy
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # ── Guardo `terminated`, NO `done` en el buffer
                # Truncated ~ "timeout ~ límite de 200 pasos" NO es un estado terminal real:
                # el coche sigue en el valle y el futuro existe ~ debo seguir bootstrapping.
                # Si guardara `done`, el 99% de los episodios tempranos tendrían
                # terminated=True → bootstrap desaparece → agente no aprende.
                self.buffer.push(obs, action, float(reward), next_obs, terminated)
                self._learn()  # Paso de gradiente ~ retorna 0.0 si buffer insuficiente

                obs = next_obs
                total_reward += reward

            # ── Decaimiento de ε después de cada episodio
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            # ── Sincronizo target_net con q_net cada N episodios
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episodio {episode:>6}/{total_episodes} | "
                    f"Avg Reward: {avg:>8.2f} | "
                    f"eps (Epsilon): {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer):>6}"
                )

        env.close()
        # ── Extiendo el historial acumulado de toda la vida del agente
        self.rewards_history.extend(rewards_history)
        return rewards_history

    # ── persistencia ~ persistence ──────────────────────────────────────

    _HPARAMS = (
        "env_id",
        "lr",
        "gamma",
        "epsilon_end",
        "epsilon_decay",
        "batch_size",
        "buffer_capacity",
        "target_update_freq",
        "hidden",
        "p_sticky",   # Ejercicio 3 ~ nuevo hiperparámetro de sticky action
    )

    def save(self, path: Path) -> None:
        """Serializo el agente con torch.save ~ optimizado para tensores PyTorch."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in self._HPARAMS}
        data["q_net_state"]     = self.q_net.state_dict()         # Pesos de la red online
        data["optimizer_state"] = self.optimizer.state_dict()     # Estado del optimizador Adam
        data["epsilon"]         = self.epsilon                     # ε actual para reanudar
        data["training_episodes"] = self.training_episodes
        data["rewards_history"] = self.rewards_history             # Historial para graficar
        torch.save(data, path)
        print(f"Agente DQN guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reconstruyo el agente DQN desde el archivo guardado."""
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            epsilon_start=data["epsilon"],  # Restauro ε exacto para reanudar correctamente
            **{k: data[k] for k in cls._HPARAMS if k != "env_id"},
        )
        # target_net empieza como copia de q_net ~ no necesito guardarla por separado
        # Se re-sincroniza durante el entrenamiento de todas formas
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["q_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        # .get con default ~ compatibilidad con saves anteriores sin historial
        agent.rewards_history = data.get("rewards_history", [])
        return agent

    def info(self) -> str:
        """Información resumida del agente ~ para el comando `load` de la CLI."""
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"Agente DQN para {self.env_id}\n"
            f"  Episodios entrenados  : {self.training_episodes}\n"
            f"  Parametros de red     : {params:,}\n"
            f"  Epsilon               : {self.epsilon:.4f}\n"
            f"  LR (alpha) / Gamma    : {self.lr} / {self.gamma}\n"
            f"  Batch size            : {self.batch_size}\n"
            f"  Target sync ~ cada    : {self.target_update_freq} episodios\n"
            f"  Sticky action (p)     : {self.p_sticky}\n"
            f"  Dispositivo ~ Device  : {self.device}\n"
            f"  Historial guardado    : {len(self.rewards_history)} episodios"
        )
