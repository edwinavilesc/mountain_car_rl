![CI](https://github.com/emiliomunozai/mountain_car/actions/workflows/ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2%2B-green)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

# MountainCar ~ Q-Learning vs DQN

> **Taller 1 ~ Aprendizaje por Refuerzo**
> Maestria en Inteligencia Artificial ~ Simulacion y Aprendizaje por Refuerzo
> Edwin Aviles

Implementacion y comparacion de dos enfoques de RL ~ "Reinforcement Learning ~
Aprendizaje por Refuerzo" sobre el entorno clasico **MountainCar-v0**:

| Agente | Tipo | Recompensa obtenida | Episodios de entrenamiento |
|---|---|:---:|:---:|
| **Q-Learning** | Tabular (metodo clasico) | -156.80 | 20,000 |
| **DQN** | Deep Reinforcement Learning | -117.90 | 2,500 |

Ambos agentes resuelven el problema de forma consistente ~ **10/10 episodios**
alcanzan la bandera en evaluacion. DQN obtiene mayor recompensa usando solo
el 12.5% de los episodios de entrenamiento de Q-Learning.

---

## El problema ~ MountainCar-v0

Un coche esta atrapado en un valle. Su motor es demasiado debil para subir
directo por la pendiente. Debe balancearse hacia atras y adelante para acumular
momentum ~ "impulso ~ energia cinetica acumulada" y llegar a la bandera.

```
                    Flag  (posicion >= 0.5)
                   /
                  /
    ____________ /
   /            X
  /    valle   / \
 /        Car /   \
/            /     \
  x: [-1.2 ─────── 0.6]
```

**Espacio de observacion:** 2 variables continuas ~ posicion `[-1.2, 0.6]` y
velocidad `[-0.07, 0.07]`.

**Acciones:** 3 discretas ~ empujar izquierda (0), sin aceleracion (1),
empujar derecha (2).

**Recompensa:** -1 por cada paso. El episodio termina al llegar a la bandera
o al pasar 200 pasos. **No hay recompensa positiva por acercarse a la meta**,
lo que hace este entorno especialmente dificil para algoritmos clasicos.

---

## Instalacion

Requiere [uv](https://docs.astral.sh/uv/) (gestor de paquetes Python moderno).

```bash
# Clonar el repositorio
git clone https://github.com/emiliomunozai/mountain_car.git
cd mountain_car

# Instalar dependencias (Python 3.11, PyTorch, Gymnasium, matplotlib)
uv sync
```

**Dependencias principales:**

| Paquete | Version | Proposito |
|---|---|---|
| `gymnasium[classic-control]` | >=1.2.3 | Entorno MountainCar-v0 |
| `torch` | >=2.10.0 | Red neuronal del agente DQN |
| `numpy` | >=2.4.2 | Operaciones numericas vectorizadas |
| `matplotlib` | >=3.9.0 | Graficas de curvas de aprendizaje |

---

## Uso ~ Comandos disponibles

Todos los comandos se acceden a traves de la CLI ~ "Command Line Interface":

```bash
uv run mountaincar <comando> [opciones]
```

### Referencia rapida de comandos

| Comando | Descripcion | Ejemplo |
|---|---|---|
| `inspect` | Ver el entorno y transiciones de ejemplo | `uv run mountaincar inspect` |
| `train <agent>` | Entrenar un agente | `uv run mountaincar train qlearning --episodes 20000` |
| `load <agent>` | Ver info del agente guardado | `uv run mountaincar load qlearning` |
| `load <agent> --eval` | Evaluar en 10 episodios | `uv run mountaincar load dqn --eval` |
| `plot <agent>` | Generar curvas de aprendizaje (.png) | `uv run mountaincar plot qlearning` |
| `sim <agent>` | Simular episodios en texto | `uv run mountaincar sim qlearning` |
| `render <agent>` | Ver el coche en accion (ventana) | `uv run mountaincar render dqn` |
| `delete <agent>` | Borrar agente guardado | `uv run mountaincar delete qlearning` |
| `list` | Listar agentes y su estado | `uv run mountaincar list` |
| `version` | Ver version del paquete | `uv run mountaincar version` |

`<agent>` es `qlearning` o `dqn`.

---

## Sesion completa de ejemplo

```bash
# 1. Ver el entorno antes de empezar
uv run mountaincar inspect --steps 5

# 2. Entrenar el agente Q-Learning (~4 minutos)
uv run mountaincar train qlearning --episodes 20000

# 3. Evaluar Q-Learning (10 episodios greedy)
uv run mountaincar load qlearning --eval

# 4. Generar curva de aprendizaje Q-Learning
uv run mountaincar plot qlearning --no-show

# 5. Entrenar el agente DQN (~5 minutos en CPU)
uv run mountaincar train dqn --episodes 2500

# 6. Evaluar DQN
uv run mountaincar load dqn --eval

# 7. Generar curvas de aprendizaje DQN + comparativa automatica
uv run mountaincar plot dqn --no-show

# 8. Ver el coche en accion (ventana grafica)
uv run mountaincar render dqn --episodes 3
```

---

## Agente 1 ~ Q-Learning Tabular

### Como funciona

Q-Learning tabular aprende una funcion Q(s, a) ~ "calidad de tomar accion `a`
en estado `s`" almacenada como una tabla indexada por estados discretos.

Como MountainCar tiene un espacio de observacion continuo, **discretizo** el
espacio de estados en una cuadricula de 20x20 = 400 casillas usando
`np.digitize()`. Cada casilla es una clave en un diccionario (la Q-Table).

**Ecuacion de actualizacion TD ~ Temporal Difference:**

```
Q(s, a) <- Q(s, a) + alfa * [ r + gamma * max Q(s', a')  -  Q(s, a) ]
                              |___________________________________|
                                     Error TD (cuanto me equivoque)
```

**Politica Epsilon-Greedy:**
```
Con probabilidad epsilon   -> Explorar (accion aleatoria)
Con probabilidad 1-epsilon -> Explotar (argmax Q-Table)
```

### Diagrama del ciclo de entrenamiento

Ver diagrama completo en:
[`docs/02_diagramas/DIAGRAMA_QLEARNING.md`](docs/02_diagramas/DIAGRAMA_QLEARNING.md)

```
Obs. continua                 Q-Table                  Entorno
[pos, vel]   ->  discretize  ->  Q[(7,14)] = [-152, -148, -99]
                  (7, 14)    ->  select_action (eps-greedy)
                               ->  accion = 2 (-> derecha)
                                 ->  env.step(2)
                               r=-1, s'=[-0.47, 0.004]
                  (7, 15)    <-  discretize(s')
                Q[(7,14)][2] += 0.1 * (target - current_q)
```

### Hiperparametros ~ Q-Learning

```
n_bins       = 20        # 400 estados discretos totales
lr (alfa)    = 0.10      # Tasa de aprendizaje
gamma        = 0.99      # Factor de descuento
epsilon_ini  = 1.00      # Exploracion inicial (100%)
epsilon_fin  = 0.01      # Exploracion minima (1%)
epsilon_dcy  = 0.9995    # Decaimiento multiplicativo por episodio
episodios    = 20,000
```

### Resultado obtenido

```
Mean reward: -156.80 +/- 24.67  |  Reached the flag: 10/10
Estados visitados: 294 / 400 (73.5% de la cuadricula)
```

---

## Agente 2 ~ DQN ~ Deep Q-Network

### Como funciona

DQN reemplaza la Q-Table por una **red neuronal** que predice los Q-values
directamente desde el estado continuo, sin necesidad de discretizar.

**Arquitectura de la red (QNetwork):**

```
Entrada        Capa 1          Capa 2         Salida
[pos, vel] -> Linear(2->128) -> Linear(128->128) -> Linear(128->3)
(2 floats)     + ReLU           + ReLU            (3 Q-values, sin activacion)
```

ReLU ~ "Rectified Linear Unit ~ f(x) = max(0, x)" introduce no-linealidad.
Sin activacion final porque los Q-values son reales negativos sin restriccion.

**Tres innovaciones sobre Q-Learning basico:**

1. **ReplayBuffer** ~ almacena hasta 100,000 transiciones y muestrea
   mini-batches ~ "lotes" aleatorios de 64 para romper correlacion temporal

2. **Target Network** ~ red congelada que provee targets de Bellman estables.
   Sin ella, el "blanco" cambia en cada paso -> entrenamiento inestable

3. **Sticky Exploration** ~ Ejercicio 3 ~ correccion critica:
   con `p_sticky=0.9`, el 90% de las veces repito la accion anterior durante
   la exploracion, produciendo carreras sostenidas de empujes que permiten
   al coche construir momentum. Sin esto, DQN nunca aprende:
   ```
   P(20 empujes sostenidos, exploracion uniforme) = (1/3)^20 ~= 0
   P(20 empujes sostenidos, p_sticky=0.9)         = 0.9^19  ~= 13.5%
   ```

**Ecuacion de Bellman ~ Bellman target:**

```python
current_q = q_net(s).gather(1, a)               # shape (B, 1)
next_q    = target_net(s').max(dim=1).values     # shape (B, 1), sin gradiente
target_q  = r + gamma * next_q * (1 - done)     # shape (B, 1)
loss      = MSE(current_q, target_q)
```

### Diagrama del ciclo de entrenamiento

Ver diagrama completo en:
[`docs/02_diagramas/DIAGRAMA_DQN.md`](docs/02_diagramas/DIAGRAMA_DQN.md)

```
obs continua --> select_action (sticky eps-greedy) --> env.step(a)
                      |                                    |
                      v                                    v
               ReplayBuffer.push(s, a, r, s', terminated)
                      |
                      v
               sample(64) --> _learn()
                                  |-- current_q = q_net(s).gather(a)
                                  |-- next_q    = target_net(s').max()
                                  |-- target_q  = r + gamma*next_q*(1-done)
                                  |-- loss      = MSE(current_q, target_q)
                                  `-- optimizer.step()
               Cada 10 eps: target_net <- q_net
```

### Hiperparametros ~ DQN

```
lr (alfa)          = 0.001    # Tasa de aprendizaje (Adam optimizer)
gamma              = 0.99     # Factor de descuento
epsilon_ini        = 1.00     # Exploracion inicial
epsilon_fin        = 0.01     # Exploracion minima
epsilon_dcy        = 0.995    # Decaimiento por episodio
batch_size (B)     = 64       # Tamano del mini-batch
buffer_capacity(N) = 100,000  # Capacidad del ReplayBuffer
target_update      = 10 eps   # Frecuencia de sync target net
hidden (h)         = 128      # Neuronas por capa oculta
p_sticky           = 0.9      # Probabilidad sticky action (Ejercicio 3)
episodios          = 2,500
```

### Resultado obtenido

```
Mean reward: -117.90 +/- 28.90  |  Reached the flag: 10/10
Parametros de la red: 17,283
Dispositivo: CPU (red demasiado pequena para aprovechar GPU)
```

---

## Resultados y Comparacion

### Curvas de aprendizaje

| Q-Learning (20,000 episodios) | DQN (2,500 episodios) |
|---|---|
| ![Q-Learning](docs/03_resultados/plots/qlearning_learning_curve.png) | ![DQN](docs/03_resultados/plots/dqn_learning_curve.png) |

### Comparativa directa

![Comparacion Q-Learning vs DQN](docs/03_resultados/plots/comparison_ql_vs_dqn.png)

### Tabla de resultados

| Metrica | Q-Learning | DQN | Ventaja |
|---|:---:|:---:|:---:|
| Recompensa promedio (eval) | -156.80 | **-117.90** | DQN (+38.9 pts) |
| Desviacion estandar | **+/-24.67** | +/-28.90 | Q-Learning |
| Banderas alcanzadas | 10/10 | 10/10 | Empate |
| Episodios de entrenamiento | 20,000 | **2,500** | DQN (8x menos) |
| Umbral "resuelto" (-110) | No (-156) | Cerca (-118) | DQN |
| Generalizacion entre estados | No | Si | DQN |
| Complejidad de implementacion | Baja | Alta | Q-Learning |

### Comparacion cualitativa

| Aspecto | Q-Learning | DQN |
|---|---|---|
| **Estabilidad entrenamiento** | Alta ~ curva monotonica | Media ~ ruidosa pero consistente |
| **Velocidad de aprendizaje** | Lenta ~ necesita explorar cada estado | Rapida ~ generaliza entre estados |
| **Desempeno final** | Funcional ~ 10/10 pero menos eficiente | Superior ~ 10/10 mas eficiente |
| **Requiere GPU** | No | No (red tiny, bottleneck es el entorno) |
| **Escalabilidad** | Limitada ~ maldicion de la dimensionalidad | Alta ~ funciona con imagenes, sensores |

Ver analisis detallado en:
[`docs/03_resultados/RESULTADOS.md`](docs/03_resultados/RESULTADOS.md)

---

## Estructura del repositorio

```
mountain_car_rl/
|
|-- src/mountain_car/
|   |-- cli.py                          <- CLI ~ todos los comandos del proyecto
|   `-- agents/
|       |-- qlearning.py                <- Agente Q-Learning tabular (Ejercicios 1a,1b,1c)
|       `-- dqn.py                      <- Agente DQN (Ejercicios 2a,2b,3)
|
|-- docs/
|   |-- 01_conceptos/
|   |   `-- CONCEPTOS.md               <- Guia conceptual completa con analisis de codigo
|   |-- 02_diagramas/
|   |   |-- DIAGRAMA_QLEARNING.md      <- Esquema de entrenamiento Q-Learning (entregable)
|   |   `-- DIAGRAMA_DQN.md            <- Esquema de entrenamiento DQN (entregable)
|   `-- 03_resultados/
|       |-- RESULTADOS.md              <- Analisis y comparacion de resultados (entregable)
|       `-- plots/                     <- Graficas .png generadas por `mountaincar plot`
|
|-- saves/                             <- Agentes entrenados (.pkl y .pt)
|-- .github/workflows/ci.yml           <- CI ~ GitHub Actions
|-- pyproject.toml                     <- Dependencias y configuracion del proyecto
|-- EXERCISES.md                       <- Guia de los ejercicios del profesor
`-- README.md                          <- Este archivo
```

---

## Conceptos clave

Para una explicacion completa de todos los conceptos desde cero, incluyendo
analisis linea a linea del codigo, diagramas de clases y glosario de acronimos,
ver:

**[docs/01_conceptos/CONCEPTOS.md](docs/01_conceptos/CONCEPTOS.md)**

Temas cubiertos:
- Que es RL ~ Reinforcement Learning ~ Aprendizaje por Refuerzo
- El entorno MountainCar-v0 y por que es dificil
- Q-Learning ~ como funciona la Q-Table y la ecuacion TD
- DQN ~ arquitectura de red, replay buffer, target network
- El bug de exploracion ~ por que la exploracion uniforme falla en MountainCar
- Distincion terminated vs truncated ~ critica para el target de Bellman
- Shapes de tensores en DQN ~ el bug silencioso de broadcast
- Flujo completo de la CLI y el sistema de guardado

---

## Reflexiones finales

Implementar estos dos agentes desde cero me dejo aprendizajes que no aparecen
directamente en los libros de texto.

El mas importante: **el bug de exploracion del Ejercicio 3**. Cuando DQN reporto
un score plano de -200 para siempre, mi primer instinto fue revisar el codigo
de aprendizaje. Pero el aprendizaje estaba correcto ~ el problema era la
exploracion. El calculo `(1/3)^20 ~= 3 en 10 mil millones` me hizo entender
que algunos entornos requieren explorar con estructura, no solo con ruido.
La solucion (sticky action con `p_sticky=0.9`) fue sencilla una vez entendi
el diagnostico, pero llegar al diagnostico requirio pensar como un experimentador,
no como un programador.

El segundo aprendizaje fue la distincion `terminated` vs `truncated`. Es un
detalle que parece trivial pero que rompe el entrenamiento silenciosamente si
se ignora. En MountainCar, el 99% de los episodios tempranos terminan por
timeout ~ si los marco como terminados, el agente aprende que "despues del
paso 200 no existe el futuro", lo cual es falso y corrompe los targets de
Bellman para casi todas las transiciones.

Finalmente, comparar ambos metodos empiricamente confirma algo que la teoria
predice pero que es mas convincente en los datos: DQN generaliza, Q-Learning
memoriza. Con el mismo entorno y los mismos hiperparametros base, DQN necesito
8 veces menos episodios para obtener una politica superior. El costo es la
complejidad de implementacion ~ pero en problemas reales ese costo vale la pena.

---

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
  (2nd ed., Caps. 4-6). MIT Press.
  [Link](http://incompleteideas.net/book/the-book-2nd.html)

- Lapan, M. (2020). *Deep Reinforcement Learning Hands-On* (2nd ed., Cap. 6 ~ DQN).
  Packt Publishing.

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning.
  *Nature*, 518, 529-533.
  [Link](https://www.nature.com/articles/nature14236)

- Gymnasium Documentation ~ MountainCar-v0.
  [Link](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

- PyTorch Documentation ~ nn.Module, optim.Adam, MSELoss.
  [Link](https://pytorch.org/docs/stable/index.html)

---

*Repositorio desarrollado como entregable del Taller 1 ~ Aprendizaje por Refuerzo.*
*Maestria en Inteligencia Artificial ~ 2026.*
