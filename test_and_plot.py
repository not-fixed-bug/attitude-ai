"""
test_and_plot.py

Campagna sperimentale completa per il confronto LQR vs PPO nel controllo
d'assetto di un microsatellite. Produce figure publication-quality per la tesi.

Test implementati (corrispondenza con Capitolo 5):
  - Test 1: Scenario nominale lineare (piccolo angolo ~15°)
  - Test 2: Regime non lineare (grande angolo ~120°)
  - Test 3: Robustezza (rumore sensori + guasti attuatori)

Ogni esecuzione crea una cartella dedicata in tests/ con tutti i risultati.

Autore: Progetto attitude-ai
Tesi Magistrale in Ingegneria Informatica
"""
from __future__ import annotations

import glob as glob_module
import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from lqr_controller import LQRController
from satellite_dynamics import SatelliteDynamics
from satellite_env import SatelliteEnv

# -----------------------------------------------------------------------------
# Configurazione grafica per pubblicazione
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "lines.antialiased": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.figsize": (11, 7),
})


# =============================================================================
# UTILITY: Caricamento modello e setup
# =============================================================================

def resolve_latest_training_run(models_root: str = "models") -> Tuple[str, str]:
    """Individua l'ultimo run di training disponibile."""
    pointer_path = os.path.join(models_root, "latest_run.txt")
    if os.path.exists(pointer_path):
        with open(pointer_path, "r", encoding="utf-8") as handle:
            run_id = handle.read().strip()
        run_dir = os.path.join(models_root, run_id)
        model_path = os.path.join(run_dir, "ppo_satellite_final.zip")
        vecnorm_path = os.path.join(run_dir, "vec_normalize.pkl")
        if os.path.exists(model_path):
            return model_path, vecnorm_path

    run_dirs = sorted(glob_module.glob(os.path.join(models_root, "run_*")), reverse=True)
    for run_dir in run_dirs:
        model_path = os.path.join(run_dir, "ppo_satellite_final.zip")
        vecnorm_path = os.path.join(run_dir, "vec_normalize.pkl")
        if os.path.exists(model_path):
            return model_path, vecnorm_path

    raise FileNotFoundError(
        f"Nessun modello PPO trovato in '{models_root}/'. Eseguire prima 'python train.py'."
    )


def load_ppo_model(models_root: str = "models") -> Tuple[PPO, object, str]:
    """Carica il modello PPO più recente e il normalizzatore."""
    model_path, vecnorm_path = resolve_latest_training_run(models_root)
    model = PPO.load(model_path)
    vec_normalize = None
    if os.path.exists(vecnorm_path):
        dummy_env = DummyVecEnv([lambda: SatelliteEnv()])
        vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    return model, vec_normalize, model_path


def normalize_obs_for_ppo(observation: np.ndarray, vec_normalize) -> np.ndarray:
    """Normalizza l'osservazione con le statistiche del training."""
    obs = observation.reshape(1, -1).astype(np.float32)
    if vec_normalize is not None and hasattr(vec_normalize, "normalize_obs"):
        return vec_normalize.normalize_obs(obs)
    return obs


def build_test_directory(prefix: str = "test") -> Tuple[str, str]:
    """Crea una cartella dedicata al singolo esperimento in tests/."""
    test_id = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    test_dir = os.path.join("tests", test_id)
    os.makedirs(test_dir, exist_ok=True)
    return test_id, test_dir


# =============================================================================
# GENERAZIONE CONDIZIONI INIZIALI
# =============================================================================

def generate_initial_state(angle_deg: float, seed: int = 42) -> np.ndarray:
    """Genera uno stato iniziale con errore angolare specificato."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    q0 = np.cos(angle_rad / 2.0)
    qv = np.sin(angle_rad / 2.0) * axis
    initial_quaternion = np.array([q0, qv[0], qv[1], qv[2]])
    initial_omega = np.array([0.0, 0.0, 0.0])
    return np.concatenate([initial_quaternion, initial_omega])


# =============================================================================
# SIMULAZIONE IN ANELLO CHIUSO
# =============================================================================

def run_simulation(
    controller_type: str,
    initial_state: np.ndarray,
    total_time: float = 60.0,
    dt: float = 0.1,
    fault_time: float | None = None,
    fault_loss: float = 0.0,
    fault_axis: int = 0,
    sensor_noise_std: float = 0.0,
    disturbances: dict | None = None,
    model: PPO | None = None,
    vec_normalize=None,
) -> Dict[str, np.ndarray]:
    """
    Esegue una simulazione in anello chiuso con configurazione flessibile.

    Args:
        controller_type: "LQR" o "PPO"
        initial_state: Stato iniziale [q, ω]
        total_time: Durata simulazione [s]
        dt: Passo temporale [s]
        fault_time: Istante di iniezione guasto [s] (None = nessun guasto)
        fault_loss: Severità del guasto [0, 1]
        fault_axis: Asse del guasto (0=X, 1=Y, 2=Z)
        sensor_noise_std: Rumore sui sensori
        disturbances: Configurazione disturbi ambientali
        model: Modello PPO (richiesto se controller_type="PPO")
        vec_normalize: Normalizzatore VecNormalize

    Returns:
        Dict con time, angle_error, omega, torques, states
    """
    num_steps = int(total_time / dt)
    fault_step = int(fault_time / dt) if fault_time is not None else num_steps + 1

    dynamics = SatelliteDynamics(disturbances=disturbances)
    dynamics.reset(initial_state)

    controller = LQRController(max_torque=0.05) if controller_type == "LQR" else None
    max_torque = 0.05
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Storage
    time_hist = np.zeros(num_steps)
    angle_error_hist = np.zeros(num_steps)
    omega_hist = np.zeros((num_steps, 3))
    torques_hist = np.zeros((num_steps, 3))
    states_hist = np.zeros((num_steps, 7))

    for step in range(num_steps):
        time_hist[step] = step * dt
        state = dynamics.state.copy()
        states_hist[step] = state

        # Calcolo errore
        q_error = dynamics.quaternion_error(state[:4], target_quat)
        angle_error_hist[step] = np.rad2deg(
            2.0 * np.arccos(np.clip(np.abs(q_error[0]), 0.0, 1.0))
        )
        omega_hist[step] = state[4:]

        # Osservazione (con eventuale rumore)
        obs = state.copy()
        if sensor_noise_std > 0.0:
            obs += np.random.normal(0.0, sensor_noise_std, size=obs.shape)
            q_norm = np.linalg.norm(obs[:4])
            if q_norm > 1e-10:
                obs[:4] /= q_norm

        # Calcolo controllo
        if controller_type == "LQR":
            commanded_torque = controller.compute_control(obs)
        else:
            normalized_obs = normalize_obs_for_ppo(obs.astype(np.float32), vec_normalize)
            action, _ = model.predict(normalized_obs, deterministic=True)
            commanded_torque = action[0] * max_torque

        # Applicazione guasto
        applied_torque = np.array(commanded_torque, dtype=float)
        if step >= fault_step:
            applied_torque[fault_axis] *= (1.0 - fault_loss)

        torques_hist[step] = applied_torque
        dynamics.step(applied_torque, dt)

    return {
        "time": time_hist,
        "angle_error": angle_error_hist,
        "omega": omega_hist,
        "torques": torques_hist,
        "states": states_hist,
    }


# =============================================================================
# METRICHE QUANTITATIVE
# =============================================================================

def compute_metrics(data: Dict[str, np.ndarray], settling_threshold_deg: float = 2.0) -> Dict[str, float]:
    """Calcola metriche di prestazione da una simulazione."""
    angle_error = data["angle_error"]
    torques = data["torques"]
    time = data["time"]

    # Settling time: primo istante dopo il quale l'errore rimane sotto soglia
    settling_time = time[-1]  # Default: non converge
    for i in range(len(angle_error)):
        if np.all(angle_error[i:] < settling_threshold_deg):
            settling_time = time[i]
            break

    # Errore RMS totale
    rms_error = float(np.sqrt(np.mean(np.square(angle_error))))

    # Errore a regime (ultimi 10% della simulazione)
    ss_start = int(0.9 * len(angle_error))
    steady_state_error = float(np.mean(angle_error[ss_start:]))

    # Consumo energetico totale (integrale delle coppie al quadrato)
    energy = float(np.sum(np.sum(torques**2, axis=1)) * (time[1] - time[0]))

    # Coppia di picco
    peak_torque = float(np.max(np.abs(torques)))

    # Velocità angolare massima
    omega_max = float(np.max(np.linalg.norm(data["omega"], axis=1)))

    return {
        "settling_time_s": settling_time,
        "rms_error_deg": rms_error,
        "steady_state_error_deg": steady_state_error,
        "energy_consumption": energy,
        "peak_torque_Nm": peak_torque,
        "omega_max_rad_s": omega_max,
    }


def compute_rms_post_fault(angle_error: np.ndarray, fault_step: int) -> float:
    """RMS dell'errore nella sola finestra post-guasto."""
    post = angle_error[fault_step:]
    return float(np.sqrt(np.mean(np.square(post))))


# =============================================================================
# TEST 1: SCENARIO NOMINALE LINEARE (PICCOLI ANGOLI)
# =============================================================================

def run_test1(model, vec_normalize, test_dir: str) -> Dict[str, Dict]:
    """Test 1: Scenario nominale con piccola perturbazione (~15°)."""
    print("[TEST 1] Scenario nominale lineare (15°)")

    initial_state = generate_initial_state(angle_deg=15.0, seed=42)

    lqr_data = run_simulation("LQR", initial_state, total_time=60.0)
    ppo_data = run_simulation("PPO", initial_state, total_time=60.0,
                              model=model, vec_normalize=vec_normalize)

    lqr_metrics = compute_metrics(lqr_data)
    ppo_metrics = compute_metrics(ppo_data)

    # --- FIGURA ---
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.35)

    # Panel 1: Errore di assetto
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(lqr_data["time"], lqr_data["angle_error"], label="LQR", color="tab:blue")
    ax1.plot(ppo_data["time"], ppo_data["angle_error"], label="PPO", color="tab:red")
    ax1.axhline(2.0, color="green", linestyle="--", alpha=0.7, label="Soglia 2°")
    ax1.set_xlabel("Tempo [s]")
    ax1.set_ylabel("Errore di assetto [°]")
    ax1.set_title("Test 1 — Scenario Nominale: Errore di Assetto (condizione iniziale 15°)")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(bottom=0)

    # Panel 2: Coppie di controllo
    ax2 = fig.add_subplot(gs[1])
    for i, axis_name in enumerate(["X", "Y", "Z"]):
        ax2.plot(lqr_data["time"], lqr_data["torques"][:, i] * 1000,
                 linestyle="-", alpha=0.7, label=f"LQR {axis_name}")
    for i, axis_name in enumerate(["X", "Y", "Z"]):
        ax2.plot(ppo_data["time"], ppo_data["torques"][:, i] * 1000,
                 linestyle="--", alpha=0.7, label=f"PPO {axis_name}")
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Coppia [mN·m]")
    ax2.set_title("Test 1 — Coppie di Controllo")
    ax2.legend(loc="upper right", ncol=2, fontsize=8)
    ax2.grid(True)
    ax2.set_xlim(0, 60)

    # Panel 3: Velocità angolare
    ax3 = fig.add_subplot(gs[2])
    lqr_omega_mag = np.linalg.norm(lqr_data["omega"], axis=1)
    ppo_omega_mag = np.linalg.norm(ppo_data["omega"], axis=1)
    ax3.plot(lqr_data["time"], np.rad2deg(lqr_omega_mag), label="LQR", color="tab:blue")
    ax3.plot(ppo_data["time"], np.rad2deg(ppo_omega_mag), label="PPO", color="tab:red")
    ax3.set_xlabel("Tempo [s]")
    ax3.set_ylabel("Velocità angolare [°/s]")
    ax3.set_title("Test 1 — Velocità Angolare Residua")
    ax3.legend(loc="upper right")
    ax3.grid(True)
    ax3.set_xlim(0, 60)
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(test_dir, "test1_nominale.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Figura salvata: {path}")

    return {"lqr": lqr_metrics, "ppo": ppo_metrics}


# =============================================================================
# TEST 2: REGIME NON LINEARE (GRANDI ANGOLI)
# =============================================================================

def run_test2(model, vec_normalize, test_dir: str) -> Dict[str, Dict]:
    """Test 2: Grande angolo di manovra (120°) – regime non lineare."""
    print("[TEST 2] Regime non lineare (120°)")

    initial_state = generate_initial_state(angle_deg=120.0, seed=42)

    lqr_data = run_simulation("LQR", initial_state, total_time=80.0)
    ppo_data = run_simulation("PPO", initial_state, total_time=80.0,
                              model=model, vec_normalize=vec_normalize)

    lqr_metrics = compute_metrics(lqr_data)
    ppo_metrics = compute_metrics(ppo_data)

    # --- FIGURA ---
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.35)

    # Panel 1: Errore di assetto
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(lqr_data["time"], lqr_data["angle_error"], label="LQR", color="tab:blue")
    ax1.plot(ppo_data["time"], ppo_data["angle_error"], label="PPO", color="tab:red")
    ax1.axhline(2.0, color="green", linestyle="--", alpha=0.7, label="Soglia 2°")
    # Evidenziazione zona non lineare (>15°)
    ax1.axhspan(15, 180, alpha=0.05, color="orange", label="Zona non lineare (>15°)")
    ax1.set_xlabel("Tempo [s]")
    ax1.set_ylabel("Errore di assetto [°]")
    ax1.set_title("Test 2 — Regime Non Lineare: Errore di Assetto (condizione iniziale 120°)")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(bottom=0)

    # Panel 2: Coppie
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(lqr_data["time"], np.linalg.norm(lqr_data["torques"], axis=1) * 1000,
             label="LQR ||τ||", color="tab:blue")
    ax2.plot(ppo_data["time"], np.linalg.norm(ppo_data["torques"], axis=1) * 1000,
             label="PPO ||τ||", color="tab:red")
    ax2.axhline(50.0, color="gray", linestyle=":", alpha=0.7, label="Saturazione (50 mN·m)")
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("||Coppia|| [mN·m]")
    ax2.set_title("Test 2 — Norma della Coppia di Controllo")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    ax2.set_xlim(0, 80)

    # Panel 3: Velocità angolare
    ax3 = fig.add_subplot(gs[2])
    lqr_omega_mag = np.linalg.norm(lqr_data["omega"], axis=1)
    ppo_omega_mag = np.linalg.norm(ppo_data["omega"], axis=1)
    ax3.plot(lqr_data["time"], np.rad2deg(lqr_omega_mag), label="LQR", color="tab:blue")
    ax3.plot(ppo_data["time"], np.rad2deg(ppo_omega_mag), label="PPO", color="tab:red")
    ax3.set_xlabel("Tempo [s]")
    ax3.set_ylabel("Velocità angolare [°/s]")
    ax3.set_title("Test 2 — Velocità Angolare")
    ax3.legend(loc="upper right")
    ax3.grid(True)
    ax3.set_xlim(0, 80)
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(test_dir, "test2_non_lineare.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Figura salvata: {path}")

    return {"lqr": lqr_metrics, "ppo": ppo_metrics}


# =============================================================================
# TEST 3: ROBUSTEZZA (RUMORE + GUASTI)
# =============================================================================

def run_test3(model, vec_normalize, test_dir: str) -> Dict[str, Dict]:
    """Test 3: Analisi di robustezza con rumore sui sensori e guasti attuatori."""
    print("[TEST 3] Analisi di robustezza")

    initial_state = generate_initial_state(angle_deg=45.0, seed=42)
    fault_time = 20.0
    dt = 0.1
    total_time = 60.0

    # --- PARTE A: Sweep guasti con rumore fisso ---
    print("  [3a] Sweep severità guasto...")
    fault_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
    noise_level = 0.005  # Rumore moderato sui sensori

    lqr_rms = []
    ppo_rms = []
    fault_step = int(fault_time / dt)

    for fl in fault_levels:
        lqr_data = run_simulation("LQR", initial_state, total_time=total_time,
                                  fault_time=fault_time, fault_loss=float(fl),
                                  sensor_noise_std=noise_level)
        ppo_data = run_simulation("PPO", initial_state, total_time=total_time,
                                  fault_time=fault_time, fault_loss=float(fl),
                                  sensor_noise_std=noise_level,
                                  model=model, vec_normalize=vec_normalize)
        lqr_rms.append(compute_rms_post_fault(lqr_data["angle_error"], fault_step))
        ppo_rms.append(compute_rms_post_fault(ppo_data["angle_error"], fault_step))

    lqr_rms = np.array(lqr_rms)
    ppo_rms = np.array(ppo_rms)

    # --- PARTE B: Sweep rumore con guasto fisso ---
    print("  [3b] Sweep livello rumore sensori...")
    noise_levels = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    fixed_fault = 0.5  # Guasto 50%

    lqr_rms_noise = []
    ppo_rms_noise = []

    for nl in noise_levels:
        lqr_data = run_simulation("LQR", initial_state, total_time=total_time,
                                  fault_time=fault_time, fault_loss=fixed_fault,
                                  sensor_noise_std=float(nl))
        ppo_data = run_simulation("PPO", initial_state, total_time=total_time,
                                  fault_time=fault_time, fault_loss=fixed_fault,
                                  sensor_noise_std=float(nl),
                                  model=model, vec_normalize=vec_normalize)
        lqr_rms_noise.append(compute_rms_post_fault(lqr_data["angle_error"], fault_step))
        ppo_rms_noise.append(compute_rms_post_fault(ppo_data["angle_error"], fault_step))

    lqr_rms_noise = np.array(lqr_rms_noise)
    ppo_rms_noise = np.array(ppo_rms_noise)

    # --- PARTE C: Simulazione singola con guasto 70% per visualizzazione traiettoria ---
    print("  [3c] Simulazione traiettoria con guasto 70%...")
    lqr_traj = run_simulation("LQR", initial_state, total_time=total_time,
                              fault_time=fault_time, fault_loss=0.7,
                              sensor_noise_std=noise_level)
    ppo_traj = run_simulation("PPO", initial_state, total_time=total_time,
                              fault_time=fault_time, fault_loss=0.7,
                              sensor_noise_std=noise_level,
                              model=model, vec_normalize=vec_normalize)

    # --- FIGURA 3A: Sweep guasti ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(fault_levels * 100, lqr_rms, "o-", color="tab:blue", label="LQR", markersize=7)
    ax1.plot(fault_levels * 100, ppo_rms, "s-", color="tab:red", label="PPO", markersize=7)
    ax1.set_xlabel("Severità del guasto [%]")
    ax1.set_ylabel("Errore RMS post-guasto [°]")
    ax1.set_title("Test 3a — Tolleranza a Guasti degli Attuatori\n(rumore sensori σ = 0.005)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(-2, 92)

    ax2.plot(noise_levels * 100, lqr_rms_noise, "o-", color="tab:blue", label="LQR", markersize=7)
    ax2.plot(noise_levels * 100, ppo_rms_noise, "s-", color="tab:red", label="PPO", markersize=7)
    ax2.set_xlabel("Rumore sensori σ [% dello stato]")
    ax2.set_ylabel("Errore RMS post-guasto [°]")
    ax2.set_title("Test 3b — Robustezza al Rumore sui Sensori\n(guasto attuatore 50%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path_sweep = os.path.join(test_dir, "test3_robustezza_sweep.png")
    plt.savefig(path_sweep, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Figura sweep salvata: {path_sweep}")

    # --- FIGURA 3B: Traiettoria con guasto ---
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 1, figure=fig, hspace=0.30)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(lqr_traj["time"], lqr_traj["angle_error"], label="LQR", color="tab:blue")
    ax1.plot(ppo_traj["time"], ppo_traj["angle_error"], label="PPO", color="tab:red")
    ax1.axvline(fault_time, color="black", linestyle="--", linewidth=1.5,
                label=f"Guasto 70% (t = {fault_time:.0f} s)")
    ax1.axhline(2.0, color="green", linestyle="--", alpha=0.7, label="Soglia 2°")
    ax1.set_xlabel("Tempo [s]")
    ax1.set_ylabel("Errore di assetto [°]")
    ax1.set_title("Test 3c — Traiettoria con Guasto Attuatore 70% + Rumore Sensori")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.set_xlim(0, total_time)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(lqr_traj["time"], lqr_traj["torques"][:, 0] * 1000,
             label="LQR (asse X)", color="tab:blue")
    ax2.plot(ppo_traj["time"], ppo_traj["torques"][:, 0] * 1000,
             label="PPO (asse X)", color="tab:red")
    ax2.axvline(fault_time, color="black", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel("Coppia applicata asse X [mN·m]")
    ax2.set_title("Test 3c — Coppia Effettivamente Applicata (asse guasto)")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    ax2.set_xlim(0, total_time)

    plt.tight_layout()
    path_traj = os.path.join(test_dir, "test3_traiettoria_guasto.png")
    plt.savefig(path_traj, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Figura traiettoria salvata: {path_traj}")

    # Metriche per la traiettoria con guasto
    lqr_metrics = compute_metrics(lqr_traj)
    ppo_metrics = compute_metrics(ppo_traj)

    return {"lqr": lqr_metrics, "ppo": ppo_metrics}


# =============================================================================
# TABELLA RIASSUNTIVA
# =============================================================================

def save_summary_table(all_results: Dict[str, Dict], test_dir: str) -> None:
    """Salva la tabella comparativa in formato CSV e testo formattato."""

    # CSV
    csv_path = os.path.join(test_dir, "metriche_comparative.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Controllore", "Settling Time [s]", "RMS Error [°]",
                         "Steady-State Error [°]", "Energia", "Coppia Picco [N·m]",
                         "ω max [rad/s]"])
        for test_name, results in all_results.items():
            for ctrl_name, metrics in results.items():
                writer.writerow([
                    test_name, ctrl_name.upper(),
                    f"{metrics['settling_time_s']:.2f}",
                    f"{metrics['rms_error_deg']:.2f}",
                    f"{metrics['steady_state_error_deg']:.3f}",
                    f"{metrics['energy_consumption']:.6f}",
                    f"{metrics['peak_torque_Nm']:.4f}",
                    f"{metrics['omega_max_rad_s']:.4f}",
                ])

    # Testo formattato per stampa
    txt_path = os.path.join(test_dir, "metriche_comparative.txt")
    with open(txt_path, "w") as f:
        header = f"{'Test':<20} {'Ctrl':<5} {'T_set [s]':<10} {'RMS [°]':<10} {'SS err [°]':<11} {'Energia':<12} {'τ_max [Nm]':<11} {'ω_max [r/s]':<11}"
        separator = "-" * len(header)
        f.write(header + "\n")
        f.write(separator + "\n")
        print(f"\n{'='*70}")
        print("TABELLA RIASSUNTIVA DEI RISULTATI")
        print(f"{'='*70}")
        print(header)
        print(separator)
        for test_name, results in all_results.items():
            for ctrl_name, metrics in results.items():
                line = (f"{test_name:<20} {ctrl_name.upper():<5} "
                        f"{metrics['settling_time_s']:<10.2f} "
                        f"{metrics['rms_error_deg']:<10.2f} "
                        f"{metrics['steady_state_error_deg']:<11.3f} "
                        f"{metrics['energy_consumption']:<12.6f} "
                        f"{metrics['peak_torque_Nm']:<11.4f} "
                        f"{metrics['omega_max_rad_s']:<11.4f}")
                f.write(line + "\n")
                print(line)
        print(f"{'='*70}")

    print(f"\n[OK] Metriche salvate in: {csv_path}")
    print(f"[OK] Metriche salvate in: {txt_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Esegue la campagna sperimentale completa (Cap. 5 della tesi)."""
    print("=" * 70)
    print("CAMPAGNA SPERIMENTALE — attitude-ai")
    print("Confronto LQR vs PPO | Capitolo 5 della Tesi")
    print("=" * 70)
    print()

    # Setup
    test_id, test_dir = build_test_directory()
    print(f"[INFO] Cartella risultati: {test_dir}/")

    try:
        model, vec_normalize, model_path = load_ppo_model()
    except FileNotFoundError as exc:
        print(f"[ERRORE] {exc}")
        return

    print(f"[INFO] Modello PPO caricato: {model_path}")
    print()

    # Esecuzione dei 3 test
    all_results = {}

    all_results["Test 1 (15°)"] = run_test1(model, vec_normalize, test_dir)
    print()
    all_results["Test 2 (120°)"] = run_test2(model, vec_normalize, test_dir)
    print()
    all_results["Test 3 (Robust.)"] = run_test3(model, vec_normalize, test_dir)
    print()

    # Tabella riassuntiva
    save_summary_table(all_results, test_dir)

    # Chiusura
    print()
    print("=" * 70)
    print("CAMPAGNA SPERIMENTALE COMPLETATA")
    print("=" * 70)
    print(f"\nCartella risultati: {test_dir}/")
    print("File generati:")
    for f in sorted(os.listdir(test_dir)):
        print(f"  • {f}")
    print()
    print("Mapping con Capitolo 5 della tesi:")
    print("  • test1_nominale.png          → §5.1 Scenario nominale lineare")
    print("  • test2_non_lineare.png       → §5.2 Regime non lineare")
    print("  • test3_robustezza_sweep.png  → §5.3 Analisi di robustezza (sweep)")
    print("  • test3_traiettoria_guasto.png→ §5.3 Traiettoria con guasto")
    print("  • metriche_comparative.csv    → §5.4 Discussione critica")
    print("  • metriche_comparative.txt    → §5.4 Tabella formattata")

    if vec_normalize is not None:
        vec_normalize.close()


if __name__ == "__main__":
    main()
