"""
Tabular Q-Learning para MountainCar-v0.
Q-Learning ~ "algoritmo de aprendizaje por refuerzo sin modelo que aprende
una Q-Table ~ tabla de calidad de acciones por estado"

MountainCar tiene una observación continua 2-D (posición, velocidad) con
límites publicados por el propio entorno, por lo que puedo discretizar todo
el espacio de estados en una cuadrícula n_bins x n_bins ~ sin necesidad de
ajustar límites manualmente ni casos especiales.

Implementé los tres métodos centrales del algoritmo:
  ~ discretize   : convierte observación continua en clave discreta de tabla
  ~ select_action: política ε-greedy ~ "epsilon-greedy ~ exploración vs explotación"
  ~ _update      : actualización TD ~ "Temporal Difference ~ Diferencia Temporal"
"""

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np


class QLearningAgent:
    """
    Agente de Q-Learning tabular para entornos Gymnasium con espacio de
    observación continuo y acción discreta.

    Discretizo el espacio de estados en una cuadrícula n_bins x n_bins y
    mantengo una Q-Table ~ "tabla Q ~ diccionario que mapea (estado discreto,
    acción) hacia el valor esperado de retorno acumulado".

    La Q-Table se implementa como defaultdict ~ "diccionario con valor por
    defecto ~ retorna ceros para estados nunca visitados sin necesidad de
    inicializarlos explícitamente".
    """

    def __init__(
        self,
        env_id: str,
        *,
        n_bins: int = 20,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
    ) -> None:
        # ── Hiperparámetros ~ "hyperparameters ~ parámetros que configuran el aprendizaje"
        self.env_id = env_id
        self.n_bins = n_bins              # Bins ~ cajitas de discretización por dimensión
        self.lr = lr                       # α ~ "learning rate ~ tasa de aprendizaje"
        self.gamma = gamma                 # γ ~ "discount factor ~ factor de descuento"
        self.epsilon = epsilon_start       # ε ~ exploración inicial (empieza explorando todo)
        self.epsilon_end = epsilon_end     # ε mínimo ~ nunca baja de este valor
        self.epsilon_decay = epsilon_decay # Decaimiento multiplicativo de ε por episodio
        self.training_episodes = 0         # Contador de episodios (permite reanudar)
        self.rewards_history: list[float] = []  # Historial acumulado para graficar

        # ── Obtengo los límites del espacio de observación directamente del entorno
        env = gym.make(env_id)
        low, high = env.observation_space.low, env.observation_space.high
        self.n_actions = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        # ── Construyo los bordes interiores de los bins ~ "bin edges ~ bordes de cajitas"
        # np.linspace genera n_bins+1 puntos equiespaciados entre lo y hi
        # [1:-1] descarta los extremos: np.digitize solo necesita los bordes INTERIORES
        # para producir exactamente n_bins casillas de salida por dimensión
        # Ejemplo para posición [-1.2, 0.6] con n_bins=20: 19 bordes interiores equiespaciados
        self._bins = [
            np.linspace(lo, hi, n_bins + 1)[1:-1]
            for lo, hi in zip(low, high)
        ]

        # ── Q-Table ~ "tabla Q ~ defaultdict que retorna ceros para estados no visitados"
        # El defaultdict evita la necesidad de verificar existencia antes de acceder
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

    # ── helpers ~ funciones auxiliares ────────────────────────────────

    def discretize(self, obs: np.ndarray) -> tuple:
        """Convierto una observación continua en una clave discreta para la Q-Table.

        ~ Ejercicio 1a ~

        Uso np.digitize ~ "función que retorna el índice del bin en que cae un valor"
        para cada dimensión de la observación. El resultado es una tupla hashable ~
        "que puede usarse como clave de diccionario de Python".

        Ejemplo concreto con MountainCar:
          obs = [-0.4823, 0.0021]  ~ posición y velocidad como floats
          → (7, 14)                ~ índice de cajita para cada dimensión

        self._bins ya está construido en __init__ ~ solo debo usarlo aquí.
        Los valores fuera del rango declarado por el entorno son manejados
        automáticamente por np.digitize ~ retorna 0 o n_bins (sin errores).
        """
        return tuple(
            # np.digitize(valor, bordes) ~ retorna en qué bin cae el valor
            int(np.digitize(obs[i], self._bins[i]))
            for i in range(len(obs))
        )

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        """Selecciono una acción usando la política ε-greedy ~ "epsilon-greedy".

        ~ Ejercicio 1b ~

        ε-greedy ~ "epsilon greedy ~ estrategia que balancea exploración y explotación":
          ~ Con probabilidad ε     : exploro ~ acción aleatoria uniforme
          ~ Con probabilidad (1-ε) : exploto ~ argmax de la Q-Table para este estado

        Cuando deterministic=True NUNCA exploro ~ modo evaluación y render puro.
        Ignorar este flag haría que los resultados de evaluación fueran ruidosos
        y el agente pareciera peor de lo que realmente es.
        """
        # ── Modo exploración ~ solo si no es determinístico y el dado cae bajo ε
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # Acción aleatoria uniforme en [0, n_actions)

        # ── Modo explotación ~ mejor acción según la Q-Table actual
        # defaultdict garantiza que self.q_table[state] existe (retorna zeros si es nuevo)
        # np.argmax retorna el índice del valor máximo ~ la mejor acción conocida
        return int(np.argmax(self.q_table[state]))

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        """Interfaz de predicción compatible con la CLI ~ "Command Line Interface".

        Discretizo la observación continua y delego en select_action.
        """
        return self.select_action(self.discretize(obs), deterministic=deterministic), None

    # ── core RL ~ algoritmo central ────────────────────────────────────

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        terminated: bool,
    ) -> None:
        """Actualización TD ~ "Temporal Difference ~ Diferencia Temporal" de Q-Learning.

        ~ Ejercicio 1c ~

        Muevo Q(state, action) hacia el objetivo TD ~ "TD target ~ blanco de aprendizaje":

            target   = reward + γ * max_a' Q(next_state, a')  [si NO terminó]
            target   = reward                                  [si terminó en la bandera]
            Q(s, a) += α * (target ~ Q(s, a))                 [actualización in-place]

        IMPORTANTE ~ distinción terminated vs done:
          ~ terminated=True: llegué a la bandera ~ estado terminal REAL ~ no hay futuro
          ~ terminated=False: el episodio continúa (o terminó por timeout/truncation)
          ~ Nunca uso `done` aquí ~ el timeout no es un estado terminal real

        Escribo el target primero, luego la actualización ~ ayuda a evitar bugs
        por colapsar todo en una sola línea antes de verificar que funciona.
        """
        # ── Valor Q actual para el par (state, action) que acabo de tomar
        current_q = self.q_table[state][action]

        if terminated:
            # ── Llegué a la bandera: no hay estado siguiente ~ no hago bootstrap
            # target = solo la recompensa inmediata (en MountainCar siempre -1)
            target = reward
        else:
            # ── El episodio continúa: bootstrap ~ "aprovecho mi mejor estimación futura"
            # max_a' Q(next_state, a') ~ el mejor Q-valor posible desde el siguiente estado
            best_next_q = float(np.max(self.q_table[next_state]))
            target = reward + self.gamma * best_next_q  # Ecuación de Bellman

        # ── Error TD ~ "cuánto me equivoqué en mi estimación anterior"
        td_error = target - current_q

        # ── Actualización in-place de la Q-Table
        # α (lr) controla el tamaño del paso ~ cuánto me corrijo hacia el target
        self.q_table[state][action] += self.lr * td_error

    def train(self, total_episodes: int = 10_000, log_interval: int = 100) -> list[float]:
        """Ciclo completo de entrenamiento del agente Q-Learning.

        Ejecuto `total_episodes` episodios en el entorno MountainCar-v0,
        actualizando la Q-Table después de cada transición.
        Retorno el historial de recompensas para graficar la curva de aprendizaje.
        """
        env = gym.make(self.env_id)
        rewards_history: list[float] = []  # Recompensas de esta sesión de entrenamiento

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()          # Reinicio el entorno ~ observación inicial continua
            state = self.discretize(obs)  # Discretizo para obtener la clave de tabla
            total_reward = 0.0
            done = False

            while not done:
                # ── Selecciono acción con política ε-greedy
                action = self.select_action(state)

                # ── Ejecuto la acción en el entorno
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # terminated ~ True si llegué a la bandera (éxito real)
                # truncated  ~ True si agotaron los 200 pasos (timeout ~ límite de tiempo)
                done = terminated or truncated

                # ── Discretizo el siguiente estado y actualizo la Q-Table
                next_state = self.discretize(next_obs)
                # Paso `terminated`, NO `done` ~ el timeout no es un estado terminal real
                # Ver CONCEPTOS.md §8 para la explicación completa de esta distinción
                self._update(state, action, float(reward), next_state, terminated)

                # ── Avanzo al siguiente estado
                state = next_state
                total_reward += reward

            # ── Decaimiento de ε después de cada episodio completo
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            # ── Log periódico ~ "periodic log ~ registro cada N episodios"
            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episodio {episode:>6}/{total_episodes} | "
                    f"Avg Reward: {avg:>8.2f} | "
                    f"eps (Epsilon): {self.epsilon:.4f} | "
                    f"Estados visitados: {len(self.q_table)}"
                )

        env.close()
        # ── Extiendo el historial acumulado de toda la vida del agente
        # Permite reanudar entrenamiento y conservar el historial completo
        self.rewards_history.extend(rewards_history)
        return rewards_history

    # ── persistence ~ persistencia del agente ─────────────────────────

    _HPARAMS = ("env_id", "n_bins", "lr", "gamma", "epsilon_end", "epsilon_decay")

    def save(self, path: Path) -> None:
        """Serializo el agente con pickle ~ "Python serialization ~ conversión a binario"."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in self._HPARAMS}
        data["q_table"] = dict(self.q_table)  # Convierto defaultdict a dict normal para serializar
        data["epsilon"] = self.epsilon         # ε actual ~ para reanudar entrenamiento
        data["training_episodes"] = self.training_episodes
        data["rewards_history"] = self.rewards_history  # Historial completo para graficar
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Agente Q-Learning guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reconstruyo el agente desde el archivo pickle guardado."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            data["env_id"],
            epsilon_start=data["epsilon"],  # Restauro ε exacto para reanudar correctamente
            **{k: data[k] for k in cls._HPARAMS if k != "env_id"},
        )
        # ── Reconstruyo el defaultdict ~ garantiza que estados no visitados retornen zeros
        agent.q_table = defaultdict(
            lambda: np.zeros(agent.n_actions), data["q_table"]
        )
        agent.training_episodes = data["training_episodes"]
        # .get con default ~ compatibilidad con saves anteriores sin historial
        agent.rewards_history = data.get("rewards_history", [])
        return agent

    def info(self) -> str:
        """Información resumida del agente ~ para el comando `load` de la CLI."""
        return (
            f"Agente Q-Learning para {self.env_id}\n"
            f"  Episodios entrenados : {self.training_episodes}\n"
            f"  Estados visitados    : {len(self.q_table)} / {self.n_bins ** 2}\n"
            f"  Epsilon              : {self.epsilon:.4f}\n"
            f"  LR (alpha) / Gamma   : {self.lr} / {self.gamma}\n"
            f"  Historial guardado   : {len(self.rewards_history)} episodios"
        )
