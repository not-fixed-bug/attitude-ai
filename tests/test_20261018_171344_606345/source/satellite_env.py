"""
satellite_env.py

Ambiente Gymnasium personalizzato per il controllo d'assetto di un microsatellite.
Questo modulo definisce l'interfaccia standard di Reinforcement Learning secondo
le specifiche di Gymnasium, consentendo l'utilizzo di algoritmi di Deep RL come PPO.

L'ambiente simula il problema di riorientamento satellitare: l'agente deve portare
il satellite da un assetto iniziale perturbato all'assetto di riferimento (quaternione
identità), minimizzando sia l'errore di assetto che il consumo energetico.

Funzionalità avanzate (opzionali, backward-compatible):
- Domain Randomization: variazione della matrice di inerzia ad ogni episodio
- Sensor Noise: rumore gaussiano sulle osservazioni
- Observation Augmentation: errore angolare esplicito nell'osservazione
- Disturbi ambientali configurabili

Autore: Progetto attitude-ai
Tesi Magistrale in Ingegneria Informatica
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from satellite_dynamics import SatelliteDynamics, canonicalize_observation


class SatelliteEnv(gym.Env):
    """
    Ambiente Gymnasium per il controllo d'assetto satellitare con Deep RL.

    Spazio degli stati (observation_space):
        Modalità base (7D):
        - Quaternione di orientamento [q0, q1, q2, q3]
        - Velocità angolari [ωx, ωy, ωz] in rad/s

        Modalità augmented (10D):
        - Quaternione di orientamento [q0, q1, q2, q3]
        - Velocità angolari [ωx, ωy, ωz] in rad/s
        - Errore angolare vettoriale [e1, e2, e3] (parte vettoriale del quaternione errore)

    Spazio delle azioni (action_space):
        - Coppie di controllo normalizzate [-1, 1] per ciascun asse
        - Totale: 3 dimensioni continue
    """

    metadata = {'render_modes': []}

    def __init__(self, max_torque=0.05, dt=0.1, max_steps=600, inertia_matrix=None,
                 domain_randomization=False, inertia_noise_pct=0.1,
                 sensor_noise_std=0.0, observation_augmented=False,
                 disturbances=None, initial_angle_max=None):
        """
        Inizializza l'ambiente di simulazione satellitare.

        Args:
            max_torque (float): Coppia massima applicabile [N·m]
            dt (float): Passo temporale della simulazione [s]
            max_steps (int): Durata massima di un episodio in step
            inertia_matrix (np.ndarray, optional): Matrice di inerzia nominale
            domain_randomization (bool): Se True, randomizza l'inerzia ad ogni reset
            inertia_noise_pct (float): Percentuale di variazione dell'inerzia (0.1 = ±10%)
            sensor_noise_std (float): Deviazione standard del rumore sui sensori
            observation_augmented (bool): Se True, aggiunge l'errore angolare all'osservazione
            disturbances (dict, optional): Configurazione dei disturbi ambientali
            initial_angle_max (float, optional): Angolo iniziale massimo [rad].
                                                Se None, usa distribuzione uniforme completa.
        """
        super(SatelliteEnv, self).__init__()
        if not np.isfinite(dt) or dt <= 0 or max_steps <= 0:
            raise ValueError("dt e max_steps devono essere positivi")
        if not np.isfinite(max_torque) or max_torque <= 0:
            raise ValueError("max_torque deve essere finito e positivo")
        if not np.isfinite(sensor_noise_std) or sensor_noise_std < 0:
            raise ValueError("sensor_noise_std deve essere finito e non negativo")

        # Parametri di configurazione
        self.max_torque = max_torque
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        self.target_hold_steps = 0
        self.required_hold_steps = int(np.ceil(10.0 / dt)) + 1
        self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        # Domain Randomization
        self.domain_randomization = domain_randomization
        self.inertia_noise_pct = inertia_noise_pct
        self.nominal_inertia = np.diag([0.01, 0.01, 0.005]) if inertia_matrix is None else np.array(inertia_matrix)

        # Sensor Noise
        self.sensor_noise_std = sensor_noise_std

        # Observation Augmentation
        self.observation_augmented = observation_augmented

        # Disturbi ambientali
        self.disturbances = disturbances

        # Curriculum Learning: angolo iniziale massimo
        self.initial_angle_max = initial_angle_max

        # Inizializzazione del modello dinamico
        self.dynamics = SatelliteDynamics(
            inertia_matrix=self.nominal_inertia,
            disturbances=disturbances
        )

        # ===== DEFINIZIONE DEGLI SPAZI =====
        # Storico 2026-10-18: omega era limitata a [-10, 10] nel Box,
        # ma non nella dinamica (i run storici superano 200 rad/s).
        # Non si taglia lo stato fisico: si dichiara lo spazio effettivo.
        if self.observation_augmented:
            # 7 (stato) + 3 (errore angolare vettoriale) = 10 dimensioni
            self.observation_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, -np.inf,
                              -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf,
                               1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # 7 dimensioni (modalità standard)
            self.observation_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, -np.inf],
                             dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf],
                              dtype=np.float32),
                dtype=np.float32
            )

        # Spazio delle azioni: coppie normalizzate [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

    def _build_observation(self, state):
        """
        Costruisce l'osservazione a partire dallo stato, applicando augmentation e rumore.

        Args:
            state (np.ndarray): Stato fisico [q, ω]

        Returns:
            np.ndarray: Osservazione (con rumore e/o augmentation se configurati)
        """
        obs = state.copy()

        # Aggiunta del rumore sui sensori
        if self.sensor_noise_std > 0.0:
            # Storico 2026-10-18: il generatore globale accoppiava gli ambienti.
            # noise = np.random.normal(0.0, self.sensor_noise_std, size=obs.shape)
            noise = self.np_random.normal(0.0, self.sensor_noise_std, size=obs.shape)
            obs += noise
            # Ri-normalizzazione del quaternione dopo il rumore
            q_norm = np.linalg.norm(obs[:4])
            if q_norm > 1e-10:
                obs[:4] /= q_norm

        obs = canonicalize_observation(obs)
        # Observation augmentation: usa la stessa misura, anche se rumorosa.
        if self.observation_augmented:
            # Prima state[:4] esponeva lo stato vero anche con rumore sensori.
            q_error = self.dynamics.quaternion_error(obs[:4], self.target_quaternion)
            # Parte vettoriale del quaternione di errore (proporzionale all'errore angolare)
            angle_error_vec = q_error[1:]  # [e1, e2, e3]
            obs = np.concatenate([obs, angle_error_vec])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reimposta l'ambiente all'inizio di un nuovo episodio.

        Se domain_randomization è attivo, varia la matrice di inerzia.
        Se initial_angle_max è specificato, limita l'angolo iniziale (curriculum learning).
        """
        # Storico 2026-10-18:
        # if seed is not None:
        #     np.random.seed(seed)
        # Seed locale Gymnasium: ambienti distinti non si influenzano.
        super().reset(seed=seed)

        self.current_step = 0
        self.target_hold_steps = 0

        # === DOMAIN RANDOMIZATION ===
        if self.domain_randomization:
            # Perturbazione uniforme della matrice di inerzia nominale
            noise = 1.0 + self.np_random.uniform(
                -self.inertia_noise_pct, self.inertia_noise_pct, size=3
            )
            randomized_inertia = np.diag(np.diag(self.nominal_inertia) * noise)
            self.dynamics = SatelliteDynamics(
                inertia_matrix=randomized_inertia,
                disturbances=self.disturbances,
                rng=self.np_random
            )
        self.dynamics.rng = self.np_random

        # === GENERAZIONE CONDIZIONE INIZIALE ===
        if self.initial_angle_max is not None:
            # Curriculum Learning: angolo iniziale limitato
            angle = self.np_random.uniform(0, self.initial_angle_max)
            axis = self.np_random.normal(size=3)
            axis /= np.linalg.norm(axis)
            q0 = np.cos(angle / 2.0)
            qv = np.sin(angle / 2.0) * axis
            random_quaternion = np.array([q0, qv[0], qv[1], qv[2]])
        else:
            # Distribuzione uniforme completa sulla sfera unitaria
            random_quaternion = self.np_random.normal(size=4)
            random_quaternion /= np.linalg.norm(random_quaternion)

        initial_omega = np.array([0.0, 0.0, 0.0])
        initial_state = np.concatenate([random_quaternion, initial_omega])
        self.dynamics.reset(initial_state)

        observation = self._build_observation(initial_state)
        info = {}
        return observation, info

    def step(self, action):
        """
        Esegue un passo di simulazione applicando l'azione specificata dall'agente.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError("L'azione deve avere tre componenti finite in [-1, 1]")
        self.current_step += 1

        # Conversione azione normalizzata → coppia fisica
        torques = np.array(action) * self.max_torque

        # Propagazione della dinamica
        new_state = self.dynamics.step(torques, self.dt)

        # Calcolo dell'errore di assetto
        current_quaternion = new_state[:4]
        current_omega = new_state[4:]
        q_error = self.dynamics.quaternion_error(current_quaternion, self.target_quaternion)
        angle_error = 2.0 * np.arccos(np.clip(np.abs(q_error[0]), 0.0, 1.0))

        # Storico 2026-10-18: PPO RMS 101-105 gradi nei due esperimenti
        # test_20261018_131553 e test_20261018_161240.
        # max_angle_error = np.pi
        # normalized_error = np.clip(angle_error / max_angle_error, 0.0, 1.0)
        # attitude_reward = 10.0 * (1.0 - normalized_error)
        # attitude_penalty = -5.0 * (angle_error ** 2)
        # control_effort = np.sum(torques ** 2)
        # energy_penalty = -0.005 * control_effort
        # omega_magnitude = np.linalg.norm(current_omega)
        # max_omega = 1.0
        # normalized_omega = np.clip(omega_magnitude / max_omega, 0.0, 1.0)
        # stability_reward = 2.0 * (1.0 - normalized_omega)
        # reward = attitude_reward + attitude_penalty + energy_penalty + stability_reward
        # if angle_error < np.deg2rad(2.0) and np.linalg.norm(current_omega) < 0.01:
        #     reward += 500.0
        #     terminated = True
        # La velocita' non era penalizzata ulteriormente oltre 1 rad/s;
        # il successo interrompeva il mantenimento. Ora si usa un orizzonte
        # fisso, costo di velocita' non saturato e sforzo normalizzato.
        control_effort = np.sum(torques ** 2)
        omega_magnitude = np.linalg.norm(current_omega)
        attitude_reward = -4.0 * angle_error ** 2
        stability_reward = -0.5 * omega_magnitude ** 2
        energy_penalty = -0.01 * control_effort / self.max_torque ** 2
        success_threshold_angle = np.deg2rad(2.0)
        success_threshold_omega = 0.01
        in_target = bool(
            angle_error < success_threshold_angle
            and omega_magnitude < success_threshold_omega
        )
        self.target_hold_steps = self.target_hold_steps + 1 if in_target else 0
        reward = attitude_reward + stability_reward + energy_penalty + float(in_target)
        terminated = False
        truncated = (self.current_step >= self.max_steps)

        info = {
            'angle_error_deg': np.rad2deg(angle_error),
            'angle_error_rad': angle_error,
            'omega_magnitude': np.linalg.norm(current_omega),
            'control_effort': control_effort,
            'in_target': in_target,
            'success': self.target_hold_steps >= self.required_hold_steps,
            'reward_attitude': attitude_reward,
            'reward_stability': stability_reward,
            'reward_total': reward
        }

        observation = self._build_observation(new_state)
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Placeholder per la visualizzazione."""
        pass

    def close(self):
        """Pulizia delle risorse."""
        pass
