# Capitolo 4 – Sviluppo dell'Architettura Software e di Simulazione

## 4.1 Architettura generale del simulatore Python

Il simulatore sviluppato per questa tesi è implementato interamente in Python e si compone di cinque moduli principali, ciascuno con responsabilità ben definite secondo i principi della separazione delle competenze (separation of concerns). L'architettura è progettata per consentire il confronto diretto tra un controllore classico (LQR) e un agente di Deep Reinforcement Learning (PPO) sullo stesso modello dinamico.

### 4.1.1 Struttura del progetto

La struttura del progetto è organizzata come segue:

```
attitude-ai/
├── satellite_dynamics.py    # Modello dinamico del satellite (+ disturbi ambientali)
├── lqr_controller.py        # Controllore LQR (baseline classica)
├── satellite_env.py         # Ambiente Gymnasium per il training RL
├── train.py                 # Script di addestramento della rete neurale PPO
├── test_and_plot.py         # Campagna sperimentale e generazione grafici
├── requirements.txt         # Dipendenze del progetto
├── RUNBOOK.md               # Guida operativa completa
├── models/                  # Checkpoint e modelli addestrati
│   ├── latest_run.txt       # Puntatore all'ultimo run di training
│   └── run_YYYYMMDD_HHMMSS/ # Cartella per ciascun run
├── logs/                    # Log TensorBoard per il monitoraggio
└── tests/                   # Risultati dei test comparativi
    └── test_YYYYMMDD_HHMMSS/  # Grafici e metriche per ogni campagna
```

### 4.1.2 Stack tecnologico

Le librerie utilizzate e le rispettive responsabilità nel progetto sono dichiarate nel file `requirements.txt`:

```python
# Librerie scientifiche fondamentali
numpy>=1.24.0       # Algebra lineare e calcolo numerico
scipy>=1.10.0       # Integrazione numerica ODE e risoluzione equazione di Riccati

# Framework per Reinforcement Learning
gymnasium>=0.29.0           # Interfaccia standard per ambienti RL
stable-baselines3>=2.0.0    # Implementazione algoritmi RL (PPO)

# Libreria per il controllo ottimo classico
control>=0.9.4      # Strumenti per analisi di sistemi di controllo

# Visualizzazione e logging
matplotlib>=3.7.0   # Generazione grafici e visualizzazione risultati
tensorboard>=2.13.0 # Monitoraggio in tempo reale del training

# Backend e utilities
torch>=2.0.0        # Backend per le reti neurali di Stable-Baselines3
tqdm>=4.65.0        # Barre di progresso durante il training
rich>=13.0.0        # Output arricchito per il terminale
```

### 4.1.3 Flusso di dati tra i moduli

Il flusso di interazione tra i componenti segue il paradigma classico del Reinforcement Learning:

1. **`satellite_dynamics.py`** fornisce il motore fisico che propaga lo stato del satellite, includendo la modellazione di disturbi ambientali;
2. **`satellite_env.py`** incapsula la dinamica in un ambiente Gymnasium, gestendo osservazioni, azioni, reward, e funzionalità avanzate (domain randomization, sensor noise);
3. **`train.py`** utilizza l'ambiente per addestrare l'agente PPO tramite Stable-Baselines3;
4. **`lqr_controller.py`** fornisce la baseline di controllo classico per il confronto;
5. **`test_and_plot.py`** esegue la campagna sperimentale completa e genera i grafici per la tesi.

Il pattern architetturale adottato garantisce che sia il controllore LQR sia l'agente PPO operino sullo stesso identico modello dinamico, rendendo il confronto rigoroso e scientificamente valido. Tutte le funzionalità avanzate (disturbi, rumore, randomizzazione) sono configurabili tramite parametri opzionali, mantenendo la retrocompatibilità con la versione base.

---

## 4.2 Implementazione software del modello dinamico e del controllore LQR (Baseline)

### 4.2.1 Il modulo `satellite_dynamics.py`

Il modulo `satellite_dynamics.py` implementa le equazioni differenziali non lineari che governano il moto rotazionale di un corpo rigido nello spazio. La rappresentazione dell'orientamento utilizza i quaternioni unitari, una scelta fondamentale per evitare le singolarità intrinseche degli angoli di Eulero (gimbal lock).

#### Classe `SatelliteDynamics` — Inizializzazione

La classe principale del modulo incapsula l'intero modello fisico del microsatellite, includendo la configurazione opzionale dei disturbi ambientali:

```python
class SatelliteDynamics:
    def __init__(self, inertia_matrix=None, disturbances=None):
        """
        Args:
            inertia_matrix (np.ndarray, optional): Matrice di inerzia 3x3 [kg·m²]
            disturbances (dict, optional): Configurazione dei disturbi ambientali.
                Chiavi supportate:
                - 'gravity_gradient': bool - Coppia di gradiente gravitazionale
                - 'solar_pressure': bool - Coppia di pressione solare
                - 'noise_std': float - Deviazione standard del rumore [N·m]
                - 'orbital_rate': float - Velocità angolare orbitale [rad/s]
        """
        if inertia_matrix is None:
            self.J = np.diag([0.01, 0.01, 0.005])  # CubeSat 3U
        else:
            self.J = np.array(inertia_matrix)

        self.J_inv = np.linalg.inv(self.J)

        # Configurazione dei disturbi ambientali
        self.disturbances = disturbances or {}
        self.gravity_gradient = self.disturbances.get('gravity_gradient', False)
        self.solar_pressure = self.disturbances.get('solar_pressure', False)
        self.noise_std = self.disturbances.get('noise_std', 0.0)
        self.orbital_rate = self.disturbances.get('orbital_rate', 0.001)  # ~LEO 400km

        # Contatore temporale per disturbi dipendenti dal tempo
        self.elapsed_time = 0.0

        # Stato iniziale: quaternione identità + velocità angolari nulle
        self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
```

Il vettore di stato ha 7 componenti: le prime quattro rappresentano il quaternione di orientamento $q = [q_0, q_1, q_2, q_3]$ (dove $q_0$ è la parte scalare), mentre le ultime tre sono le componenti della velocità angolare $\omega = [\omega_x, \omega_y, \omega_z]$ espresse nel sistema di riferimento solidale al satellite.

I valori di inerzia adottati ($J_x = J_y = 0.01 \, \text{kg·m}^2$, $J_z = 0.005 \, \text{kg·m}^2$) sono rappresentativi di un microsatellite di tipo CubeSat 3U (dimensioni 10×10×30 cm, massa circa 4 kg) con simmetria assiale.

#### Matrice di cinematica quaternionica

La propagazione temporale del quaternione è gestita tramite la matrice $\Omega(\omega)$:

```python
def _omega_matrix(self, omega):
    """
    Costruisce la matrice Ω(ω) per la cinematica dei quaternioni.
    La derivata temporale del quaternione è: dq/dt = 0.5 * Ω(ω) * q
    """
    wx, wy, wz = omega
    return np.array([
        [0.0,  -wx,  -wy,  -wz],
        [wx,   0.0,   wz,  -wy],
        [wy,  -wz,   0.0,   wx],
        [wz,   wy,  -wx,   0.0]
    ])
```

Questa matrice antisimmetrica codifica la relazione differenziale tra la velocità angolare del satellite e la variazione del quaternione nel tempo. La formulazione matriciale è preferita rispetto al prodotto quaternionico diretto per ragioni di chiarezza implementativa e compatibilità con gli integratori numerici.

#### Modellazione dei disturbi ambientali

Una delle caratteristiche distintive dell'implementazione è la modellazione opzionale dei disturbi ambientali che agiscono su un satellite in orbita bassa (LEO). Sono implementati tre tipi di disturbo:

```python
def _compute_disturbance_torque(self, t, state_vec):
    """
    Calcola la coppia di disturbo ambientale totale:
    1. Gradiente gravitazionale: 3n² * (ẑ_body × J·ẑ_body)
    2. Pressione di radiazione solare: bias + componente sinusoidale
    3. Rumore stocastico: disturbi non modellati
    """
    disturbance = np.zeros(3)
    q = state_vec[:4]

    if self.gravity_gradient:
        # Direzione nadir nel frame body
        nadir_inertial = np.array([0.0, 0.0, 1.0])
        nadir_body = self._rotate_vector_by_quaternion(q, nadir_inertial)

        # Coppia di gradiente gravitazionale: τ_gg = 3n² * (ẑ_b × J·ẑ_b)
        n = self.orbital_rate
        J_nadir = self.J @ nadir_body
        tau_gg = 3.0 * n**2 * np.cross(nadir_body, J_nadir)
        disturbance += tau_gg

    if self.solar_pressure:
        # Pressione di radiazione solare: bias + componente periodica
        srp_magnitude = 1e-5  # Ordine di grandezza per CubeSat
        tau_srp = srp_magnitude * np.array([
            0.3 * np.sin(self.orbital_rate * (self.elapsed_time + t)),
            0.2 * np.cos(self.orbital_rate * (self.elapsed_time + t) * 0.7),
            0.1 * np.sin(self.orbital_rate * (self.elapsed_time + t) * 1.3)
        ])
        disturbance += tau_srp

    if self.noise_std > 0.0:
        # Rumore bianco gaussiano (disturbi non modellati)
        disturbance += np.random.normal(0.0, self.noise_std, size=3)

    return disturbance
```

Il **gradiente gravitazionale** è la coppia dominante per satelliti non sferici in orbite basse, proporzionale alla differenza tra i momenti di inerzia e al quadrato della velocità angolare orbitale $n$. La **pressione di radiazione solare** è modellata come una coppia periodica la cui ampiezza dipende dall'angolo di esposizione al Sole. Il **rumore stocastico** rappresenta l'insieme dei disturbi non esplicitamente modellati (drag aerodinamico residuo, campo magnetico terrestre, outgassing).

La rotazione dal frame inerziale al frame body è implementata tramite la Direction Cosine Matrix (DCM) derivata dal quaternione:

```python
def _rotate_vector_by_quaternion(self, q, v):
    """Ruota un vettore v dal frame inerziale al frame body."""
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 + q0*q3),     2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3),     1 - 2*(q1**2 + q3**2),     2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2),     2*(q2*q3 - q0*q1),     1 - 2*(q1**2 + q2**2)]
    ])
    return R @ v
```

#### Sistema di equazioni differenziali con disturbi

Il cuore del modello dinamico è la funzione `_dynamics`, che ora include i disturbi ambientali:

```python
def _dynamics(self, t, state_vec, torques):
    """
    Sistema ODE completo:
    1. Cinematica: dq/dt = 0.5 * Ω(ω) * q
    2. Dinamica: J * dω/dt = -ω × (J * ω) + τ_control + τ_disturbance
    """
    q = state_vec[:4]
    omega = state_vec[4:]

    # CINEMATICA DEI QUATERNIONI
    Omega = self._omega_matrix(omega)
    dq_dt = 0.5 * Omega @ q

    # DISTURBI AMBIENTALI
    disturbance_torque = self._compute_disturbance_torque(t, state_vec)
    total_torques = torques + disturbance_torque

    # DINAMICA DI EULERO
    h = self.J @ omega
    gyroscopic_term = np.cross(omega, h)
    domega_dt = self.J_inv @ (total_torques - gyroscopic_term)

    return np.concatenate([dq_dt, domega_dt])
```

Il termine giroscopico $\omega \times (J \cdot \omega)$ rappresenta le forze apparenti dovute alla rotazione del satellite. L'aggiunta dei disturbi ambientali rende il modello significativamente più realistico rispetto alla versione idealizzata.

#### Integrazione numerica e normalizzazione

```python
def step(self, torques, dt):
    """Avanza la simulazione di dt secondi con coppie di controllo specificate."""
    sol = solve_ivp(
        fun=lambda t, y: self._dynamics(t, y, torques),
        t_span=(0, dt),
        y0=self.state,
        method='RK45',
        vectorized=False
    )

    self.state = sol.y[:, -1]
    self.elapsed_time += dt  # Aggiornamento contatore per disturbi

    # Normalizzazione del quaternione (vincolo ||q|| = 1)
    q = self.state[:4]
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-10:
        self.state[:4] = q / q_norm

    return self.state.copy()
```

#### Calcolo dell'errore di assetto e operazioni quaternioniche

```python
def quaternion_error(self, q_current, q_desired):
    """Quaternione di errore: q_e = q_desired^(-1) ⊗ q_current"""
    q_d_conj = np.array([q_desired[0], -q_desired[1], -q_desired[2], -q_desired[3]])
    return self.quaternion_multiply(q_d_conj, q_current)

def quaternion_multiply(self, q1, q2):
    """Prodotto quaternionico: q1 ⊗ q2"""
    s1, x1, y1, z1 = q1
    s2, x2, y2, z2 = q2
    s = s1*s2 - x1*x2 - y1*y2 - z1*z2
    x = s1*x2 + s2*x1 + y1*z2 - z1*y2
    y = s1*y2 + s2*y1 + z1*x2 - x1*z2
    z = s1*z2 + s2*z1 + x1*y2 - y1*x2
    return np.array([s, x, y, z])

def quaternion_to_euler_angles(self, q):
    """Conversione quaternione → angoli di Eulero (convenzione ZYX)."""
    q0, q1, q2, q3 = q
    roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
    yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return np.array([roll, pitch, yaw])
```

---

### 4.2.2 Il controllore LQR (`lqr_controller.py`)

Il controllore LQR (Linear Quadratic Regulator) rappresenta la baseline di controllo classico. L'implementazione segue il procedimento standard: minimizzazione del funzionale di costo quadratico $J = \int_0^{\infty} (x^T Q x + u^T R u) \, dt$ soggetto alla dinamica linearizzata.

#### Inizializzazione e risoluzione dell'equazione di Riccati

```python
class LQRController:
    def __init__(self, inertia_matrix=None, Q=None, R=None, max_torque=0.05):
        if inertia_matrix is None:
            self.J = np.diag([0.01, 0.01, 0.005])
        else:
            self.J = np.array(inertia_matrix)

        self.J_inv = np.linalg.inv(self.J)
        self.max_torque = max_torque

        # Matrici di peso: rapporto 10:1 tra angoli e velocità
        if Q is None:
            Q = np.diag([100.0, 100.0, 100.0,   # Peso θx, θy, θz
                        10.0, 10.0, 10.0])       # Peso ωx, ωy, ωz
        if R is None:
            R = np.eye(3) * 1.0

        self.Q = Q
        self.R = R

        # Sistema linearizzato e calcolo guadagno ottimale
        A, B = self._get_linearized_system()
        P = solve_continuous_are(A, B, self.Q, self.R)  # Equazione di Riccati
        self.K = np.linalg.inv(self.R) @ B.T @ P        # K = R⁻¹ Bᵀ P
```

#### Linearizzazione del sistema

```python
def _get_linearized_system(self):
    """
    Sistema linearizzato per piccoli angoli:
        d/dt [θ]  = [0₃  I₃] [θ]  + [0₃  ] [u]
             [ω]    [0₃  0₃] [ω]    [J⁻¹ ]
    """
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)  # dθ/dt = ω

    B = np.zeros((6, 3))
    B[3:6, :] = self.J_inv   # dω/dt = J⁻¹ τ

    return A, B
```

Questa linearizzazione è valida solo per angoli inferiori a 10-15°. Per rotazioni più ampie, il modello LQR perde accuratezza — questa è una delle motivazioni chiave per l'approccio Deep RL.

#### Legge di controllo con saturazione

```python
def compute_control(self, state):
    """Calcola u = -K x con saturazione degli attuatori."""
    quaternion = state[:4]
    omega = state[4:]

    # Approssimazione piccoli angoli: θ ≈ 2 * q_vettoriale
    euler_angles = 2.0 * quaternion[1:]
    error_state = np.concatenate([euler_angles, omega])

    # Legge di controllo LQR
    control = -self.K @ error_state

    # Saturazione
    return np.clip(control, -self.max_torque, self.max_torque)
```

#### Simulazione di guasti agli attuatori

```python
def compute_control_with_fault(self, state, fault_axis=0, fault_severity=0.7):
    """
    fault_severity=0.7 → attuatore al 30% della capacità.
    """
    control = self.compute_control(state)
    control[fault_axis] *= (1.0 - fault_severity)
    return control
```

---

## 4.3 Sviluppo dell'ambiente di addestramento Gym (Spazio delle Osservazioni e Azioni)

### 4.3.1 Interfaccia Gymnasium e configurazione avanzata

Il modulo `satellite_env.py` implementa l'interfaccia standard `gymnasium.Env` con supporto per funzionalità avanzate:

```python
class SatelliteEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, max_torque=0.05, dt=0.1, max_steps=600, inertia_matrix=None,
                 domain_randomization=False, inertia_noise_pct=0.1,
                 sensor_noise_std=0.0, observation_augmented=False,
                 disturbances=None, initial_angle_max=None):
        super(SatelliteEnv, self).__init__()

        # Parametri base
        self.max_torque = max_torque
        self.dt = dt
        self.max_steps = max_steps
        self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Domain Randomization
        self.domain_randomization = domain_randomization
        self.inertia_noise_pct = inertia_noise_pct
        self.nominal_inertia = np.diag([0.01, 0.01, 0.005])

        # Sensor Noise
        self.sensor_noise_std = sensor_noise_std

        # Observation Augmentation
        self.observation_augmented = observation_augmented

        # Disturbi ambientali e Curriculum Learning
        self.disturbances = disturbances
        self.initial_angle_max = initial_angle_max

        # Inizializzazione modello dinamico
        self.dynamics = SatelliteDynamics(
            inertia_matrix=self.nominal_inertia,
            disturbances=disturbances
        )
```

Le funzionalità avanzate sono:
- **`domain_randomization`**: Varia la matrice di inerzia ad ogni episodio (±10%), per robustezza a incertezze parametriche;
- **`sensor_noise_std`**: Rumore gaussiano sulle osservazioni (simulazione sensori reali);
- **`observation_augmented`**: Estende l'osservazione con l'errore angolare esplicito;
- **`initial_angle_max`**: Limita l'angolo iniziale per curriculum learning.

### 4.3.2 Spazio delle osservazioni

```python
if self.observation_augmented:
    # 7 (stato) + 3 (errore angolare) = 10D
    self.observation_space = spaces.Box(
        low=np.array([-1.0]*4 + [-10.0]*3 + [-1.0]*3),
        high=np.array([1.0]*4 + [10.0]*3 + [1.0]*3),
        dtype=np.float32
    )
else:
    # 7D standard
    self.observation_space = spaces.Box(
        low=np.array([-1.0]*4 + [-10.0]*3),
        high=np.array([1.0]*4 + [10.0]*3),
        dtype=np.float32
    )
```

| Indice | Componente | Range | Descrizione |
|--------|-----------|-------|-------------|
| 0-3 | $q_0, q_1, q_2, q_3$ | [-1, 1] | Quaternione |
| 4-6 | $\omega_x, \omega_y, \omega_z$ | [-10, 10] rad/s | Velocità angolari |
| 7-9 | $e_1, e_2, e_3$ | [-1, 1] | Errore angolare (se augmented) |

### 4.3.3 Spazio delle azioni

```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

L'agente produce output in [-1, 1] scalati a $\tau = action \times max\_torque$.

### 4.3.4 Costruzione dell'osservazione con rumore e augmentation

```python
def _build_observation(self, state):
    obs = state.copy()

    # Rumore sui sensori
    if self.sensor_noise_std > 0.0:
        noise = np.random.normal(0.0, self.sensor_noise_std, size=obs.shape)
        obs += noise
        # Ri-normalizzazione quaternione
        q_norm = np.linalg.norm(obs[:4])
        if q_norm > 1e-10:
            obs[:4] /= q_norm

    # Augmentation: errore angolare esplicito
    if self.observation_augmented:
        q_error = self.dynamics.quaternion_error(state[:4], self.target_quaternion)
        obs = np.concatenate([obs, q_error[1:]])

    return obs.astype(np.float32)
```

### 4.3.5 Reset con Domain Randomization e Curriculum Learning

```python
def reset(self, seed=None, options=None):
    if seed is not None:
        np.random.seed(seed)
    self.current_step = 0

    # DOMAIN RANDOMIZATION
    if self.domain_randomization:
        noise = 1.0 + np.random.uniform(
            -self.inertia_noise_pct, self.inertia_noise_pct, size=3)
        randomized_inertia = np.diag(np.diag(self.nominal_inertia) * noise)
        self.dynamics = SatelliteDynamics(
            inertia_matrix=randomized_inertia, disturbances=self.disturbances)

    # CONDIZIONE INIZIALE
    if self.initial_angle_max is not None:
        # Curriculum Learning: angolo limitato
        angle = np.random.uniform(0, self.initial_angle_max)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        q0 = np.cos(angle / 2.0)
        qv = np.sin(angle / 2.0) * axis
        random_quaternion = np.array([q0, qv[0], qv[1], qv[2]])
    else:
        # Distribuzione uniforme completa
        random_quaternion = np.random.randn(4)
        random_quaternion /= np.linalg.norm(random_quaternion)

    initial_state = np.concatenate([random_quaternion, [0.0, 0.0, 0.0]])
    self.dynamics.reset(initial_state)
    return self._build_observation(initial_state), {}
```

La **Domain Randomization** espone l'agente a sistemi dinamici leggermente diversi ad ogni episodio, migliorando la robustezza della policy appresa (sim-to-real transfer).

---

## 4.4 Progettazione della Funzione di Reward

### 4.4.1 Struttura multi-obiettivo

La funzione di reward è composta da quattro termini complementari:

```python
# 1. REWARD POSITIVA PER RIDUZIONE ERRORE (lineare, densa)
normalized_error = np.clip(angle_error / np.pi, 0.0, 1.0)
attitude_reward = 10.0 * (1.0 - normalized_error)  # [0, 10]

# 2. PENALITÀ QUADRATICA PER GRANDI ERRORI
attitude_penalty = -5.0 * (angle_error ** 2)

# 3. PENALITÀ PER CONSUMO ENERGETICO
energy_penalty = -0.005 * np.sum(torques ** 2)

# 4. BONUS DI STABILITÀ
normalized_omega = np.clip(np.linalg.norm(omega) / 1.0, 0.0, 1.0)
stability_reward = 2.0 * (1.0 - normalized_omega)  # [0, 2]

# COMPOSIZIONE
reward = attitude_reward + attitude_penalty + energy_penalty + stability_reward
```

### 4.4.2 Analisi dei termini

| Termine | Formula | Range | Obiettivo |
|---------|---------|-------|-----------|
| Attitude reward | $10(1 - \theta/\pi)$ | [0, 10] | Incentivo positivo vicino al target |
| Attitude penalty | $-5\theta^2$ | [-49, 0] | Convergenza rapida per grandi errori |
| Energy penalty | $-0.005\|\tau\|^2$ | [-0.0004, 0] | Efficienza energetica |
| Stability reward | $2(1 - \|\omega\|)$ | [0, 2] | Bassa velocità angolare |

### 4.4.3 Bonus di successo

```python
if angle_error < np.deg2rad(2.0) and np.linalg.norm(omega) < 0.01:
    reward += 500.0
    terminated = True
```

Condizioni simultanee: errore < 2° E velocità < 0.01 rad/s.

### 4.4.4 Proprietà della reward

1. **Densità**: Segnale informativo ad ogni step (reward shaping);
2. **Multi-scala**: Gerarchia chiara degli obiettivi (10, 5, 0.005, 2);
3. **Boundedness**: Range limitato, stabilità numerica;
4. **Fisicità**: Metriche fisicamente significative.

---

## 4.5 Configurazione per l'addestramento della Rete Neurale

### 4.5.1 Pipeline di training (`train.py`)

Lo script supporta argomenti da riga di comando per flessibilità sperimentale:

```python
parser.add_argument("--timesteps", "-t", type=int, default=1_000_000)
parser.add_argument("--checkpoint-freq", "-c", type=int, default=50_000)
parser.add_argument("--learning-rate", "-lr", type=float, default=3e-4)
parser.add_argument("--domain-rand", action="store_true")
parser.add_argument("--sensor-noise", type=float, default=0.0)
parser.add_argument("--disturbances", action="store_true")
parser.add_argument("--obs-augmented", action="store_true")
```

Esempi di utilizzo:
```bash
python train.py                                          # Training base
python train.py --domain-rand --disturbances -t 2000000  # Training robusto
python train.py --sensor-noise 0.005 --domain-rand       # Con rumore sensori
```

### 4.5.2 Creazione dell'ambiente con funzionalità opzionali

```python
def make_env(domain_rand=False, sensor_noise=0.0, disturbances=False, obs_augmented=False):
    def _init():
        dist_config = None
        if disturbances:
            dist_config = {
                'gravity_gradient': True,
                'solar_pressure': True,
                'noise_std': 1e-5,
                'orbital_rate': 0.001
            }
        env = SatelliteEnv(
            max_torque=0.05, dt=0.1, max_steps=600,
            domain_randomization=domain_rand,
            sensor_noise_std=sensor_noise,
            disturbances=dist_config,
            observation_augmented=obs_augmented
        )
        env = Monitor(env)
        return env
    return _init
```

### 4.5.3 Normalizzazione adattiva

```python
env = DummyVecEnv([make_env(...)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
```

`VecNormalize` porta le osservazioni a media zero e varianza unitaria e riduce la varianza delle reward, stabilizzando i gradienti della policy.

### 4.5.4 Configurazione PPO

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cpu",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log=logs_dir
)
```

### 4.5.5 Iperparametri

| Parametro | Valore | Ruolo |
|-----------|--------|-------|
| `learning_rate` | 3×10⁻⁴ | Passo Adam |
| `n_steps` | 2048 | ~3-4 episodi per rollout |
| `batch_size` | 64 | Mini-batch SGD |
| `n_epochs` | 10 | Riutilizzo dati |
| `gamma` | 0.99 | Orizzonte effettivo: 100 step |
| `gae_lambda` | 0.95 | Bias-varianza tradeoff |
| `clip_range` | 0.2 | Trust region ±20% |
| `ent_coef` | 0.01 | Esplorazione |

**PPO** è scelto per stabilità (clipping), sample efficiency (riutilizzo dati), e robustezza al tuning degli iperparametri.

### 4.5.6 Architettura della rete (MlpPolicy)

- **Actor**: 2 hidden layers × 64 neuroni, tanh. Output: media + log-varianza gaussiana (3D).
- **Critic**: Stessa architettura, output scalare $V(s)$.

### 4.5.7 Checkpoint e tracking

```python
checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path=models_dir,
    name_prefix="ppo_satellite", save_vecnormalize=True)

model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

# Salvataggio finale
model.save(os.path.join(models_dir, "ppo_satellite_final"))
env.save(os.path.join(models_dir, "vec_normalize.pkl"))

# Puntatore all'ultimo run
with open("models/latest_run.txt", "w") as f:
    f.write(run_id)
```

### 4.5.8 Campagna sperimentale (`test_and_plot.py`)

Lo script esegue automaticamente i tre test per il Capitolo 5:

```python
def main():
    model, vec_normalize, _ = load_ppo_model()

    results["Test 1"] = run_test1(model, vec_normalize, test_dir)  # 15°, nominale
    results["Test 2"] = run_test2(model, vec_normalize, test_dir)  # 120°, non lineare
    results["Test 3"] = run_test3(model, vec_normalize, test_dir)  # Rumore + guasti

    save_summary_table(results, test_dir)  # CSV + testo
```

La simulazione unificata supporta tutti gli scenari:

```python
def run_simulation(
    controller_type,     # "LQR" o "PPO"
    initial_state,       # Condizione iniziale
    fault_time=None,     # Istante guasto [s]
    fault_loss=0.0,      # Severità [0, 1]
    sensor_noise_std=0.0,# Rumore sensori
    disturbances=None,   # Disturbi ambientali
    model=None,          # Modello PPO
    vec_normalize=None,  # Normalizzatore
) -> Dict[str, np.ndarray]:
```

Le metriche calcolate per ogni simulazione:

```python
def compute_metrics(data):
    return {
        "settling_time_s": ...,        # Tempo di assestamento
        "rms_error_deg": ...,          # Errore RMS
        "steady_state_error_deg": ..., # Errore a regime
        "energy_consumption": ...,     # Energia totale
        "peak_torque_Nm": ...,         # Coppia di picco
        "omega_max_rad_s": ...,        # ω massima
    }
```

Output generato:
- `test1_nominale.png` → §5.1 (scenario lineare)
- `test2_non_lineare.png` → §5.2 (grandi angoli)
- `test3_robustezza_sweep.png` → §5.3 (failure envelope)
- `test3_traiettoria_guasto.png` → §5.3 (traiettoria temporale)
- `metriche_comparative.csv` → §5.4 (tabella LaTeX)

### 4.5.9 Monitoraggio con TensorBoard

```bash
tensorboard --logdir=logs
```

Metriche tracciate: `ep_rew_mean`, `ep_len_mean`, `policy_loss`, `value_loss`, `entropy_loss`.
