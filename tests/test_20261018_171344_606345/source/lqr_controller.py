"""
lqr_controller.py

Implementazione di un controllore Linear Quadratic Regulator (LQR) per il controllo
d'assetto satellitare. L'LQR rappresenta l'approccio classico di controllo ottimo,
basato sulla linearizzazione del sistema dinamico non lineare attorno al punto di
equilibrio e sulla risoluzione dell'equazione algebrica di Riccati.

Il controllore LQR serve come baseline di confronto per valutare le prestazioni
dell'agente di Deep Reinforcement Learning (PPO) nella tesi magistrale.

Autore: Progetto attitude-ai
Tesi Magistrale in Ingegneria Informatica
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from satellite_dynamics import canonicalize_observation


class LQRController:
    """
    Controllore Linear Quadratic Regulator per il controllo d'assetto satellitare.

    Teoria LQR:
    -----------
    Il problema LQR consiste nel minimizzare il funzionale di costo quadratico:

        J = ∫₀^∞ (x^T Q x + u^T R u) dt

    soggetto alla dinamica linearizzata:

        dx/dt = A x + B u

    dove:
    - x è il vettore di stato (errore di assetto e velocità angolare)
    - u è il vettore di controllo (coppie applicate)
    - Q è la matrice di peso per gli stati (penalizza l'errore di assetto)
    - R è la matrice di peso per i controlli (penalizza l'uso di energia)

    La soluzione ottima è data dalla legge di controllo lineare:

        u = -K x

    dove la matrice di guadagno K è calcolata come:

        K = R^(-1) B^T P

    e P è la soluzione dell'equazione algebrica di Riccati (ARE):

        A^T P + P A - P B R^(-1) B^T P + Q = 0

    Linearizzazione per il satellite:
    ---------------------------------
    Il sistema non lineare viene linearizzato attorno al punto di equilibrio:
    - Quaternione identità approssimato con piccoli angoli di Eulero
    - Velocità angolari nulle

    Per piccole rotazioni, il quaternione può essere approssimato come:
        q ≈ [1, θx/2, θy/2, θz/2]^T

    Questo porta a uno spazio di stato a 6 dimensioni:
        x = [θx, θy, θz, ωx, ωy, ωz]^T

    Attributes:
        J (np.ndarray): Matrice di inerzia del satellite [kg·m²]
        J_inv (np.ndarray): Inversa della matrice di inerzia
        Q (np.ndarray): Matrice di peso per gli stati (6x6)
        R (np.ndarray): Matrice di peso per i controlli (3x3)
        K (np.ndarray): Matrice di guadagno ottimale LQR (3x6)
        max_torque (float): Saturazione delle coppie di controllo [N·m]
        dt (float): Periodo di campionamento [s].

    Dal 2026-10-18 il progetto usa la versione discreta:
    J = sum_k (x_k.T Q x_k + u_k.T R u_k),
    x_{k+1} = A_d x_k + B_d u_k, con discretizzazione ZOH.
    Le equazioni continue sopra sono conservate come riferimento teorico.
    """

    def __init__(self, inertia_matrix=None, Q=None, R=None, max_torque=0.05, dt=0.1):
        """
        Inizializza il controllore LQR e calcola la matrice di guadagno ottimale.

        Args:
            inertia_matrix (np.ndarray, optional): Matrice di inerzia 3x3 del satellite.
                                                   Se None, usa valori di default CubeSat.
            Q (np.ndarray, optional): Matrice di peso per gli stati (6x6).
                                     Valori alti penalizzano maggiormente l'errore di assetto.
                                     Se None, usa una matrice diagonale con pesi bilanciati.
            R (np.ndarray, optional): Matrice di peso per i controlli (3x3).
                                     Valori alti penalizzano l'uso di energia.
                                     Se None, usa una matrice identità scalata.
            max_torque (float): Limite di saturazione per le coppie [N·m].
                               Le coppie calcolate vengono clippate in [-max_torque, +max_torque].

        Note di tuning:
            - Rapporto Q/R alto → controllo aggressivo, consumo energetico elevato
            - Rapporto Q/R basso → controllo conservativo, convergenza lenta
            - Per applicazioni spaziali reali, R deve riflettere i vincoli energetici
        """
        # Configurazione della matrice di inerzia
        if inertia_matrix is None:
            # Valori di default per un microsatellite tipo CubeSat 3U
            self.J = np.diag([0.01, 0.01, 0.005])  # [kg·m²]
        else:
            self.J = np.array(inertia_matrix)

        # Calcolo dell'inversa per l'uso nelle equazioni dinamiche
        self.J_inv = np.linalg.inv(self.J)

        # Limite di saturazione degli attuatori
        self.max_torque = max_torque
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt deve essere finito e positivo")
        self.dt = dt

        # ===== CONFIGURAZIONE DELLE MATRICI DI PESO =====

        if Q is None:
            # Matrice Q di default: penalizza maggiormente l'errore di assetto
            # rispetto alle velocità angolari
            # Struttura: Q = diag([q_θ, q_θ, q_θ, q_ω, q_ω, q_ω])
            Q = np.diag([100.0, 100.0, 100.0,  # Peso per gli angoli [θx, θy, θz]
                        10.0, 10.0, 10.0])     # Peso per le velocità [ωx, ωy, ωz]
            # Ratio 10:1 privilegia la precisione di puntamento sulla velocità di convergenza

        if R is None:
            # Matrice R di default: penalità uniforme sul consumo energetico
            R = np.eye(3) * 1.0  # Peso uguale per tutte e tre le coppie

        self.Q = Q
        self.R = R

        # ===== CALCOLO DELLA MATRICE DI GUADAGNO LQR =====
        # Costruzione delle matrici A e B del sistema linearizzato
        A, B = self._get_linearized_system()

        # Risoluzione dell'equazione algebrica di Riccati discreta (DARE).
        # Storico 2026-10-18: RMS nominale 3.17 gradi, residuo 2.644 gradi
        # e coppie continuamente saturate nei test storici.
        # P = solve_continuous_are(A, B, self.Q, self.R)
        # self.K = np.linalg.inv(self.R) @ B.T @ P
        # Il progetto continuo ignorava il mantenimento ZOH per dt=0.1 s.
        self.A_d, self.B_d, _, _, _ = cont2discrete(
            (A, B, np.eye(6), np.zeros((6, 3))), dt, method="zoh"
        )
        P = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        self.K = np.linalg.solve(
            self.R + self.B_d.T @ P @ self.B_d,
            self.B_d.T @ P @ self.A_d
        )

    def _get_linearized_system(self):
        """
        Calcola le matrici A e B del sistema linearizzato per piccole deviazioni.

        Linearizzazione della dinamica satellitare:
        -------------------------------------------
        Le equazioni non lineari del moto vengono linearizzate attorno al punto
        di equilibrio (assetto di riferimento, velocità angolari nulle).

        Per piccoli angoli θ, il sistema linearizzato assume la forma:

            d/dt [θ]  = [0₃ₓ₃  I₃ₓ₃] [θ]  + [0₃ₓ₃    ] [u]
                 [ω]    [0₃ₓ₃  0₃ₓ₃] [ω]    [J^(-1)  ]

        dove:
        - θ = [θx, θy, θz]^T sono i piccoli angoli di Eulero (approssimazione del quaternione)
        - ω = [ωx, ωy, ωz]^T sono le velocità angolari
        - I₃ₓ₃ è la matrice identità 3x3
        - 0₃ₓ₃ è la matrice nulla 3x3
        - J^(-1) è l'inversa della matrice di inerzia

        Derivazione:
        -----------
        1. Cinematica linearizzata: dθ/dt ≈ ω (per piccoli angoli)
        2. Dinamica linearizzata: J dω/dt ≈ τ (trascurando i termini giroscopici ω × (Jω))
           quindi: dω/dt ≈ J^(-1) τ

        Returns:
            A (np.ndarray): Matrice di stato 6x6
            B (np.ndarray): Matrice di controllo 6x3

        Note:
            - Questa linearizzazione è valida solo per piccoli angoli (< 10-15°)
            - Per grandi rotazioni, il modello LQR perde accuratezza
            - I termini giroscopici trascurati diventano significativi per alte velocità angolari
        """
        # Costruzione della matrice A (dinamica dello stato)
        # Struttura a blocchi: A = [[0, I], [0, 0]]
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)  # Blocco superiore destro: dθ/dt = ω
        # I blocchi [0, 0] e [3, 0] rimangono nulli (nessun accoppiamento diretto)

        # Costruzione della matrice B (influenza del controllo)
        # Struttura: B = [[0], [J^(-1)]]
        B = np.zeros((6, 3))
        B[3:6, :] = self.J_inv  # Blocco inferiore: dω/dt = J^(-1) τ

        return A, B

    def compute_control(self, state):
        """
        Calcola l'azione di controllo ottimale per lo stato corrente.

        Implementa la legge di controllo lineare u = -K x, dove:
        - x è il vettore di errore di stato (differenza rispetto al punto di equilibrio)
        - K è la matrice di guadagno ottimale LQR

        Args:
            state (np.ndarray): Stato corrente del satellite [q0, q1, q2, q3, ωx, ωy, ωz]

        Returns:
            np.ndarray: Coppie di controllo [τx, τy, τz] saturate in [-max_torque, +max_torque]

        Processo:
        ---------
        1. Estrazione del quaternione e delle velocità angolari dallo stato
        2. Conversione del quaternione in piccoli angoli di Eulero
        3. Costruzione del vettore di errore x = [θ_error, ω]
        4. Applicazione della legge di feedback: u = -K x
        5. Saturazione delle coppie entro i limiti fisici degli attuatori

        Note implementative:
            - La conversione quaternione → angoli è valida solo per piccole rotazioni
            - Per grandi rotazioni, l'approssimazione lineare introduce errori
            - La saturazione è essenziale per rispettare i vincoli hardware
        """
        # Estrazione delle componenti dello stato
        quaternion = state[:4]  # [q0, q1, q2, q3]
        omega = state[4:]       # [ωx, ωy, ωz]

        # ===== CONVERSIONE QUATERNIONE → ANGOLI DI EULERO =====
        # Per piccole rotazioni, il quaternione può essere approssimato come:
        # q = [cos(θ/2), sin(θ/2) * n̂] ≈ [1, θ/2 * n̂]
        # dove θ è l'angolo di rotazione e n̂ è l'asse di rotazione unitario
        #
        # Invertendo: θ * n̂ ≈ 2 * [q1, q2, q3] (valido per piccoli angoli)
        # Storico 2026-10-18:
        # euler_angles = 2.0 * quaternion[1:]
        # q e -q sono lo stesso assetto: non devono invertire il comando.
        quaternion = canonicalize_observation(state)[:4]
        euler_angles = 2.0 * quaternion[1:]

        # ===== COSTRUZIONE DEL VETTORE DI ERRORE =====
        # Il vettore di stato per l'LQR è: x = [θx, θy, θz, ωx, ωy, ωz]
        # Rappresenta la deviazione rispetto al punto di equilibrio (θ=0, ω=0)
        error_state = np.concatenate([euler_angles, omega])

        # ===== APPLICAZIONE DELLA LEGGE DI CONTROLLO LQR =====
        # Calcolo dell'azione ottimale: u = -K * x
        # Il segno negativo implementa il feedback negativo necessario per la stabilità
        control = -self.K @ error_state

        # ===== SATURAZIONE DELLE COPPIE =====
        # Gli attuatori reali hanno limiti fisici di coppia massima
        # Il clipping previene comandi irrealizzabili che degraderebbero le prestazioni
        control_saturated = np.clip(control, -self.max_torque, self.max_torque)

        return control_saturated

    def get_gain_matrix(self):
        """
        Restituisce la matrice di guadagno LQR calcolata.

        Utile per l'analisi teorica del controllore e la verifica dei margini di stabilità.

        Returns:
            np.ndarray: Matrice K di dimensioni 3x6

        Note:
            - I valori di K dipendono fortemente dal tuning di Q e R
            - Margini di stabilità possono essere analizzati tramite i margini di guadagno e fase
        """
        return self.K

    def compute_control_with_fault(self, state, fault_axis=0, fault_severity=0.7):
        """
        Calcola il controllo in presenza di un guasto agli attuatori.

        Simula un degrado parziale dell'efficacia di un attuatore, tipico di:
        - Degradazione dei motori di reazione
        - Perdita parziale di propellente
        - Malfunzionamento elettronico

        Args:
            state (np.ndarray): Stato corrente del satellite
            fault_axis (int): Indice dell'asse affetto dal guasto (0=X, 1=Y, 2=Z)
            fault_severity (float): Riduzione di efficacia in [0, 1].
                                   0.7 significa che l'attuatore opera al 30% della capacità.

        Returns:
            np.ndarray: Coppie di controllo con guasto applicato

        Note:
            - Un controllore robusto dovrebbe tollerare guasti parziali
            - Questo test è fondamentale per validare l'affidabilità del sistema
        """
        # Calcolo del controllo nominale
        control = self.compute_control(state)

        # Applicazione del guasto sull'asse specificato
        # Riduzione dell'efficacia: τ_effective = τ_nominal * (1 - severity)
        control[fault_axis] *= (1.0 - fault_severity)

        return control
