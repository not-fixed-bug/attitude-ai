"""
satellite_dynamics.py

Modulo per la simulazione della dinamica d'assetto di un microsatellite spaziale.
Implementa le equazioni differenziali non lineari che governano il moto rotazionale
di un corpo rigido nello spazio, utilizzando la rappresentazione quaternionica per
l'orientamento e le equazioni di Eulero per la dinamica angolare.

Include la modellazione opzionale di disturbi ambientali:
- Gradiente gravitazionale
- Pressione di radiazione solare
- Rumore stocastico (disturbi non modellati)

Autore: Progetto attitude-ai
Tesi Magistrale in Ingegneria Informatica
"""

import numpy as np
from scipy.integrate import solve_ivp


def canonicalize_observation(state):
    """Seleziona lo stesso emisfero per i quaternioni equivalenti q e -q."""
    obs = np.array(state, dtype=float, copy=True)
    nonzero = np.flatnonzero(obs[:4])
    if len(nonzero) and obs[nonzero[0]] < 0:
        obs[:4] *= -1
    return obs


class SatelliteDynamics:
    """
    Classe per la simulazione della dinamica d'assetto di un microsatellite.

    La dinamica del satellite è descritta da:
    1. Cinematica dei quaternioni: dq/dt = 0.5 * Ω(ω) * q
       dove Ω(ω) è la matrice di moltiplicazione quaternionica
    2. Equazioni di Eulero per il corpo rigido: J * dω/dt = -ω × (J * ω) + τ
       dove J è la matrice di inerzia, ω è la velocità angolare, τ è la coppia esterna

    Attributes:
        J (np.ndarray): Matrice di inerzia 3x3 del satellite [kg·m²]
        J_inv (np.ndarray): Inversa della matrice di inerzia
        state (np.ndarray): Vettore di stato [q0, q1, q2, q3, ωx, ωy, ωz]
                           dove q è il quaternione (q0 è la parte scalare)
                           e ω è il vettore di velocità angolare [rad/s]
    """

    def __init__(self, inertia_matrix=None, disturbances=None, rng=None):
        """
        Inizializza la dinamica del satellite con una matrice di inerzia specificata.

        Args:
            inertia_matrix (np.ndarray, optional): Matrice di inerzia 3x3 [kg·m²].
                                                   Se None, utilizza valori tipici
                                                   per un microsatellite cubico.
            disturbances (dict, optional): Configurazione dei disturbi ambientali.
                Chiavi supportate:
                - 'gravity_gradient': bool (default False) - Coppia di gradiente gravitazionale
                - 'solar_pressure': bool (default False) - Coppia di pressione solare
                - 'noise_std': float (default 0.0) - Deviazione standard del rumore [N·m]
                - 'orbital_rate': float (default 0.001) - Velocità angolare orbitale [rad/s]

        Note:
            Per un satellite a simmetria cubica, i momenti principali di inerzia sono
            approssimativamente uguali. Per un CubeSat 3U (10x10x30 cm, ~4 kg):
            Jx ≈ Jy ≈ 0.01 kg·m², Jz ≈ 0.005 kg·m²
        """
        if inertia_matrix is None:
            # Valori tipici per un microsatellite con simmetria assiale
            self.J = np.diag([0.01, 0.01, 0.005])
        else:
            self.J = np.array(inertia_matrix)

        # Precalcolo dell'inversa della matrice di inerzia per efficienza computazionale
        self.J_inv = np.linalg.inv(self.J)

        # Configurazione dei disturbi ambientali
        self.disturbances = disturbances or {}
        self.gravity_gradient = self.disturbances.get('gravity_gradient', False)
        self.solar_pressure = self.disturbances.get('solar_pressure', False)
        self.noise_std = self.disturbances.get('noise_std', 0.0)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._step_noise = np.zeros(3)
        self.orbital_rate = self.disturbances.get('orbital_rate', 0.001)  # ~LEO 400km

        # Contatore temporale per disturbi dipendenti dal tempo
        self.elapsed_time = 0.0

        # Inizializzazione dello stato: quaternione identità + velocità angolari nulle
        # Quaternione [1, 0, 0, 0] rappresenta l'assetto di riferimento (nessuna rotazione)
        self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _omega_matrix(self, omega):
        """
        Costruisce la matrice Ω(ω) per la cinematica dei quaternioni.

        La matrice Ω(ω) è definita in modo che la derivata temporale del quaternione
        sia data da: dq/dt = 0.5 * Ω(ω) * q

        Args:
            omega (np.ndarray): Vettore di velocità angolare [ωx, ωy, ωz] [rad/s]

        Returns:
            np.ndarray: Matrice 4x4 per la propagazione cinematica del quaternione

        Note matematica:
            La matrice Ω(ω) ha la forma:
            [[0,    -ωx,  -ωy,  -ωz],
             [ωx,    0,    ωz,  -ωy],
             [ωy,   -ωz,   0,    ωx],
             [ωz,    ωy,  -ωx,   0 ]]
        """
        wx, wy, wz = omega
        return np.array([
            [0.0,  -wx,  -wy,  -wz],
            [wx,   0.0,   wz,  -wy],
            [wy,  -wz,   0.0,   wx],
            [wz,   wy,  -wx,   0.0]
        ])

    def _compute_disturbance_torque(self, t, state_vec):
        """
        Calcola la coppia di disturbo ambientale totale agente sul satellite.

        Modella tre fonti principali di disturbo per un satellite in LEO:

        1. Gradiente gravitazionale: 3/2 * n² * (ẑ_orbit × J·ẑ_orbit)
           - Dominante per satelliti non sferici in orbite basse
           - Dipende dalla differenza tra i momenti di inerzia
           - Tipicamente dell'ordine di 10⁻⁶ N·m per microsatelliti

        2. Pressione di radiazione solare: coppia costante + componente sinusoidale
           - Presente solo quando il satellite è illuminato
           - Dipende dalla geometria e dalle proprietà ottiche delle superfici
           - Ordine di grandezza: 10⁻⁶ - 10⁻⁵ N·m

        3. Rumore stocastico: disturbi non modellati (aerodinamici, magnetici, ecc.)
           - Modellato come rumore bianco gaussiano
           - Rappresenta l'insieme dei disturbi trascurati

        Args:
            t (float): Tempo corrente [s]
            state_vec (np.ndarray): Vettore di stato corrente

        Returns:
            np.ndarray: Coppia di disturbo totale [τ_dx, τ_dy, τ_dz] [N·m]
        """
        disturbance = np.zeros(3)
        q = state_vec[:4]

        if self.gravity_gradient:
            # Direzione nadir nel frame body (approssimata tramite il quaternione)
            # Per un satellite in orbita, il nadir nel frame inerziale è circa [0, 0, 1]
            # Ruotiamo nel frame body usando il quaternione
            nadir_inertial = np.array([0.0, 0.0, 1.0])
            nadir_body = self._rotate_vector_by_quaternion(q, nadir_inertial)

            # Coppia di gradiente gravitazionale: τ_gg = 3n² * (ẑ_b × J·ẑ_b)
            n = self.orbital_rate
            J_nadir = self.J @ nadir_body
            tau_gg = 3.0 * n**2 * np.cross(nadir_body, J_nadir)
            disturbance += tau_gg

        if self.solar_pressure:
            # Pressione di radiazione solare: bias + componente periodica
            # La periodicità modella la variazione dell'angolo di incidenza solare
            srp_magnitude = 1e-5  # Ordine di grandezza tipico per CubeSat
            tau_srp = srp_magnitude * np.array([
                0.3 * np.sin(self.orbital_rate * (self.elapsed_time + t)),
                0.2 * np.cos(self.orbital_rate * (self.elapsed_time + t) * 0.7),
                0.1 * np.sin(self.orbital_rate * (self.elapsed_time + t) * 1.3)
            ])
            disturbance += tau_srp

        if self.noise_std > 0.0:
            # Rumore bianco gaussiano (disturbi non modellati)
            # Storico 2026-10-18:
            # disturbance += np.random.normal(0.0, self.noise_std, size=3)
            # Nella RHS il rumore dipendeva dalle chiamate adattive di RK45.
            # Ora la coppia casuale resta costante per un passo di controllo.
            disturbance += self._step_noise

        return disturbance

    def _rotate_vector_by_quaternion(self, q, v):
        """
        Ruota un vettore v dal frame inerziale al frame body usando il quaternione q.

        Implementa la rotazione: v_body = q* ⊗ v ⊗ q (usando la formula diretta)

        Args:
            q (np.ndarray): Quaternione [q0, q1, q2, q3]
            v (np.ndarray): Vettore 3D da ruotare

        Returns:
            np.ndarray: Vettore ruotato nel frame body
        """
        q0, q1, q2, q3 = q
        # Matrice di rotazione dal frame inerziale al frame body (DCM)
        R = np.array([
            [1 - 2*(q2**2 + q3**2),     2*(q1*q2 + q0*q3),     2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3),     1 - 2*(q1**2 + q3**2),     2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2),     2*(q2*q3 - q0*q1),     1 - 2*(q1**2 + q2**2)]
        ])
        return R @ v

    def _dynamics(self, t, state_vec, torques):
        """
        Calcola le derivate temporali dello stato secondo le equazioni del moto.

        Questa funzione implementa il sistema di equazioni differenziali ordinarie (ODE)
        che descrive la dinamica d'assetto del satellite:

        1. Cinematica quaternionica (propagazione dell'orientamento):
           dq/dt = 0.5 * Ω(ω) * q

        2. Equazioni di Eulero (dinamica della velocità angolare):
           J * dω/dt = -ω × (J * ω) + τ_control + τ_disturbance
           quindi: dω/dt = J⁻¹ * (-ω × (J * ω) + τ_total)

        Args:
            t (float): Tempo corrente [s]
            state_vec (np.ndarray): Vettore di stato [q0, q1, q2, q3, ωx, ωy, ωz]
            torques (np.ndarray): Coppie di controllo applicate [τx, τy, τz] [N·m]

        Returns:
            np.ndarray: Derivate temporali [dq/dt, dω/dt]

        Note fisiche:
            - Il termine -ω × (J * ω) rappresenta le forze giroscopiche (effetto Coriolis)
            - In assenza di coppie esterne, il momento angolare totale si conserva
            - Le equazioni sono espresse nel sistema di riferimento solidale al satellite
        """
        # Estrazione del quaternione e delle velocità angolari dallo stato
        q = state_vec[:4]  # Quaternione [q0, q1, q2, q3]
        omega = state_vec[4:]  # Velocità angolari [ωx, ωy, ωz]

        # ===== CINEMATICA DEI QUATERNIONI =====
        # Calcolo di dq/dt = 0.5 * Ω(ω) * q
        Omega = self._omega_matrix(omega)
        dq_dt = 0.5 * Omega @ q

        # ===== DISTURBI AMBIENTALI =====
        disturbance_torque = self._compute_disturbance_torque(t, state_vec)
        total_torques = torques + disturbance_torque

        # ===== DINAMICA DI EULERO =====
        # Calcolo del momento angolare h = J * ω
        h = self.J @ omega

        # Calcolo del termine giroscopico: ω × (J * ω) = ω × h
        # Questo termine rappresenta la coppia apparente dovuta alla rotazione del satellite
        gyroscopic_term = np.cross(omega, h)

        # Applicazione dell'equazione di Eulero: J * dω/dt = -ω × (J * ω) + τ_total
        # Risolviamo per dω/dt: dω/dt = J⁻¹ * (τ_total - ω × (J * ω))
        domega_dt = self.J_inv @ (total_torques - gyroscopic_term)

        # Concatenazione delle derivate per formare il vettore completo dstate/dt
        return np.concatenate([dq_dt, domega_dt])

    def step(self, torques, dt):
        """
        Avanza la simulazione di un intervallo di tempo dt applicando le coppie specificate.

        Utilizza un integratore numerico di alta precisione (Runge-Kutta di ordine 4-5)
        per risolvere le equazioni differenziali del moto. Dopo l'integrazione,
        normalizza il quaternione per evitare deriva numerica.

        Args:
            torques (np.ndarray): Coppie di controllo applicate [τx, τy, τz] [N·m]
            dt (float): Intervallo di integrazione temporale [s]

        Returns:
            np.ndarray: Nuovo stato del satellite [q, ω]

        Note implementative:
            - Si utilizza solve_ivp con metodo 'RK45' (Runge-Kutta adattivo)
            - Il quaternione viene normalizzato dopo ogni step per mantenere ||q|| = 1
            - La normalizzazione è fondamentale per prevenire l'accumulo di errori numerici
        """
        self._step_noise = (
            self.rng.normal(0.0, self.noise_std, size=3)
            if self.noise_std > 0.0 else np.zeros(3)
        )
        # Integrazione numerica delle equazioni del moto dal tempo t=0 a t=dt
        sol = solve_ivp(
            fun=lambda t, y: self._dynamics(t, y, torques),
            t_span=(0, dt),
            y0=self.state,
            method='RK45',
            vectorized=False
        )

        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise RuntimeError(f"Integrazione dinamica fallita: {sol.message}")
        # Estrazione dello stato finale dall'integratore
        self.state = sol.y[:, -1]

        # Aggiornamento del contatore temporale per i disturbi
        self.elapsed_time += dt

        # ===== NORMALIZZAZIONE DEL QUATERNIONE =====
        q = self.state[:4]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            self.state[:4] = q / q_norm

        return self.state.copy()

    def reset(self, initial_state=None):
        """
        Reimposta lo stato del satellite a una condizione iniziale specificata.

        Args:
            initial_state (np.ndarray, optional): Stato iniziale [q, ω].
                                                  Se None, usa quaternione identità e ω=0.

        Returns:
            np.ndarray: Lo stato reimpostato
        """
        if initial_state is None:
            self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            self.state = np.array(initial_state).copy()
            q_norm = np.linalg.norm(self.state[:4])
            if q_norm > 1e-10:
                self.state[:4] /= q_norm

        # Reset del contatore temporale per i disturbi
        self.elapsed_time = 0.0
        self._step_noise = np.zeros(3)

        return self.state.copy()

    def quaternion_error(self, q_current, q_desired):
        """
        Calcola il quaternione di errore tra l'orientamento corrente e quello desiderato.

        Il quaternione di errore q_e è definito come:
        q_e = q_desired^(-1) ⊗ q_current

        dove ⊗ indica il prodotto quaternionico e q^(-1) è il quaternione coniugato.

        Args:
            q_current (np.ndarray): Quaternione dell'orientamento corrente [q0, q1, q2, q3]
            q_desired (np.ndarray): Quaternione dell'orientamento desiderato

        Returns:
            np.ndarray: Quaternione di errore

        Note matematiche:
            - Per quaternioni unitari, l'inverso coincide con il coniugato: q^(-1) = q*
            - Il coniugato di q = [q0, qv] è q* = [q0, -qv]
            - L'angolo di errore θ può essere estratto da: θ = 2 * arccos(|q_e0|)
        """
        # Coniugato di q_desired (inversione della parte vettoriale)
        q_d_conj = np.array([q_desired[0], -q_desired[1], -q_desired[2], -q_desired[3]])

        # Prodotto quaternionico: q_error = q_d_conj ⊗ q_current
        return self.quaternion_multiply(q_d_conj, q_current)

    def quaternion_multiply(self, q1, q2):
        """
        Calcola il prodotto quaternionico q1 ⊗ q2.

        Il prodotto quaternionico non è commutativo e segue le regole:
        q1 ⊗ q2 = [s1*s2 - v1·v2, s1*v2 + s2*v1 + v1 × v2]

        dove q = [s, v] con s parte scalare e v parte vettoriale.

        Args:
            q1 (np.ndarray): Primo quaternione [s1, x1, y1, z1]
            q2 (np.ndarray): Secondo quaternione [s2, x2, y2, z2]

        Returns:
            np.ndarray: Prodotto quaternionico q1 ⊗ q2

        Note:
            Il prodotto di due quaternioni unitari è anch'esso un quaternione unitario.
            Questo prodotto rappresenta la composizione di due rotazioni.
        """
        s1, x1, y1, z1 = q1
        s2, x2, y2, z2 = q2

        # Calcolo della parte scalare: s = s1*s2 - (x1*x2 + y1*y2 + z1*z2)
        s = s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2

        # Calcolo della parte vettoriale utilizzando la formula del prodotto quaternionico
        x = s1 * x2 + s2 * x1 + y1 * z2 - z1 * y2
        y = s1 * y2 + s2 * y1 + z1 * x2 - x1 * z2
        z = s1 * z2 + s2 * z1 + x1 * y2 - y1 * x2

        return np.array([s, x, y, z])

    def quaternion_to_euler_angles(self, q):
        """
        Converte un quaternione in angoli di Eulero (roll, pitch, yaw).

        Utilizza la convenzione ZYX (yaw-pitch-roll), comunemente usata in aerospaziale.

        Args:
            q (np.ndarray): Quaternione [q0, q1, q2, q3]

        Returns:
            np.ndarray: Angoli di Eulero [roll, pitch, yaw] in radianti

        Note:
            - Roll (φ): rotazione attorno all'asse X
            - Pitch (θ): rotazione attorno all'asse Y
            - Yaw (ψ): rotazione attorno all'asse Z
            - Questa rappresentazione soffre di singolarità a pitch = ±90°
        """
        q0, q1, q2, q3 = q

        # Roll (φ): rotazione attorno all'asse X
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

        # Pitch (θ): rotazione attorno all'asse Y
        # Si utilizza np.clip per evitare errori numerici in arcsin (dominio [-1, 1])
        pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))

        # Yaw (ψ): rotazione attorno all'asse Z
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

        return np.array([roll, pitch, yaw])
