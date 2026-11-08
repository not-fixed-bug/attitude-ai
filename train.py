"""
train.py

Script principale per l'addestramento dell'agente di Deep Reinforcement Learning (PPO)
nel controllo d'assetto satellitare. Questo modulo implementa il pipeline completo di
training, includendo la configurazione dell'ambiente, l'inizializzazione dell'algoritmo
PPO, il logging su TensorBoard e il salvataggio periodico dei checkpoint del modello.

L'algoritmo PPO (Proximal Policy Optimization) è scelto per la sua stabilità e
efficienza sample, caratteristiche cruciali per problemi di controllo continuo come
quello dell'assetto satellitare.

Autore: Progetto attitude-ai
Tesi Magistrale in Ingegneria Informatica
"""

import os
import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from satellite_env import SatelliteEnv, PRECISION_REWARD_WEIGHT
from experiment_history import save_experiment_manifest


class ControlMetricsCallback(BaseCallback):
    """Affianca alla reward le grandezze fisiche e il successo a fine episodio."""

    def __init__(self):
        super().__init__()
        self.samples = defaultdict(list)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            for key in ("angle_error_deg", "omega_magnitude", "in_target", "reward_precision"):
                self.samples[key].append(float(info[key]))
            if done:
                self.samples["episode_success"].append(float(info["success"]))
        return True

    def _on_rollout_end(self):
        for key, values in self.samples.items():
            self.logger.record(f"control/{key}", float(np.mean(values)))
        self.samples.clear()


class PhysicalValidationCallback(BaseCallback):
    """Seleziona checkpoint su seed di validazione senza modificare VecNormalize."""

    def __init__(self, output_dir, frequency, episodes, seed, dt, episode_seconds):
        super().__init__()
        self.output_dir = output_dir
        self.frequency = frequency
        self.episodes = episodes
        self.seed = seed
        self.dt = dt
        self.episode_seconds = episode_seconds
        self.best_score = None

    @staticmethod
    def selection_score(success_rate, tail_error, control_effort):
        # Storico v1: si conservava solo il finale; nel pilota il checkpoint
        # 100k puntava meglio di quello 301k, ma nessuno manteneva l'assetto.
        # La reward di training non e' il criterio di selezione.
        return (success_rate, -tail_error, -control_effort)

    def _evaluate(self, phase):
        from test_and_plot import compute_metrics, generate_initial_state, run_simulation

        normalizer = self.model.get_vec_normalize_env()
        if normalizer is None:
            raise RuntimeError("La validazione richiede VecNormalize del training")
        rows = []
        for angle in (15.0, 120.0):
            duration = max(self.episode_seconds, 80.0 if angle == 120.0 else 60.0)
            for seed in range(self.seed, self.seed + self.episodes):
                data = run_simulation(
                    "PPO", generate_initial_state(angle, seed), total_time=duration,
                    dt=self.dt, seed=seed, model=self.model, vec_normalize=normalizer
                )
                metrics = compute_metrics(data)
                tail = data["time"] >= duration - 10.0
                tail_angle = float(np.mean(data["angle_error"][tail]))
                tail_omega = float(np.mean(np.linalg.norm(data["omega"][tail], axis=1)))
                rows.append({
                    "angle_deg": angle, "seed": seed, "duration_s": duration,
                    "success": metrics["success"], "rms_error_deg": metrics["rms_error_deg"],
                    "settling_time_s": (
                        metrics["settling_time_s"] if metrics["settled"] else None
                    ),
                    "tail_angle_deg": tail_angle, "tail_omega_rad_s": tail_omega,
                    "tail_normalized_error": tail_angle / 2.0 + tail_omega / 0.01,
                    "control_effort": metrics["energy_consumption"],
                })
        success_rate = float(np.mean([row["success"] for row in rows]))
        tail_error = float(np.mean([row["tail_normalized_error"] for row in rows]))
        effort = float(np.mean([row["control_effort"] for row in rows]))
        score = self.selection_score(success_rate, tail_error, effort)
        improved = self.best_score is None or score > self.best_score
        record = {
            "timesteps": self.num_timesteps, "phase": phase, "dt": self.dt,
            "success_rate": success_rate, "tail_normalized_error": tail_error,
            "control_effort": effort, "selection_score": score, "new_best": improved,
            "episodes": rows,
        }
        with open(os.path.join(self.output_dir, "validation_history.jsonl"), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        for key, value in (
            ("success_rate", success_rate), ("tail_normalized_error", tail_error),
            ("tail_angle_deg", float(np.mean([row["tail_angle_deg"] for row in rows]))),
            ("tail_omega_rad_s", float(np.mean([row["tail_omega_rad_s"] for row in rows]))),
        ):
            self.logger.record(f"validation/{key}", value)
        print(f"[VALIDAZIONE] step={self.num_timesteps} successo={success_rate:.1%} "
              f"errore normalizzato={tail_error:.3f} nuovo_best={improved}")
        if improved:
            self.best_score = score
            self.model.save(os.path.join(self.output_dir, "ppo_satellite_best"))
            normalizer.save(os.path.join(self.output_dir, "best_vec_normalize.pkl"))
            with open(os.path.join(self.output_dir, "best_validation.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, allow_nan=False)

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self._evaluate("periodic")
        return True

    def _on_training_end(self):
        # La valutazione nell'ultimo on_step precede l'aggiornamento PPO:
        # valutare ancora qui include anche i pesi dopo l'ultimo update.
        self._evaluate("final")
        self.logger.dump(self.num_timesteps)


def make_env(domain_rand=False, sensor_noise=0.0, disturbances=False, obs_augmented=False,
             dt=0.1, episode_seconds=60.0):
    """
    Factory function per la creazione dell'ambiente satellitare.

    Args:
        domain_rand (bool): Attiva la Domain Randomization
        sensor_noise (float): Deviazione standard del rumore sui sensori
        disturbances (bool): Attiva i disturbi ambientali
        obs_augmented (bool): Attiva l'observation augmentation
        dt (float): Periodo di controllo [s].
        episode_seconds (float): Durata fisica di ogni episodio [s].

    Returns:
        callable: Funzione che istanzia un ambiente wrappato con Monitor
    """
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
            max_torque=0.05,
            # Storico v1: dt=0.1, max_steps=600.
            # Il pilota a 300k oscillava a ogni campione; variare dt deve
            # mantenere la durata fisica dell'episodio, non i suoi 600 step.
            dt=dt,
            max_steps=episode_steps(dt, episode_seconds),
            domain_randomization=domain_rand,
            sensor_noise_std=sensor_noise,
            disturbances=dist_config,
            observation_augmented=obs_augmented
        )
        env = Monitor(
            env, info_keywords=("success", "angle_error_deg", "omega_magnitude")
        )
        return env
    return _init


def episode_steps(dt, episode_seconds):
    """Evita arrotondamenti silenziosi della durata fisica richiesta."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt deve essere finito e positivo")
    if not np.isfinite(episode_seconds) or episode_seconds < 10.0 + dt:
        raise ValueError("episode-seconds deve consentire almeno 10 s di mantenimento")
    steps = int(round(episode_seconds / dt))
    if not np.isclose(steps * dt, episode_seconds, rtol=0, atol=1e-9):
        raise ValueError("episode-seconds deve essere un multiplo di dt")
    return steps


def main():
    """
    Funzione principale per l'addestramento dell'agente PPO.

    Pipeline di training:
    --------------------
    1. Creazione dell'ambiente vettorializzato con normalizzazione
    2. Configurazione dell'algoritmo PPO con iperparametri ottimizzati
    3. Setup dei callback per checkpoint e valutazione
    4. Training per 1.000.000 di timesteps
    5. Salvataggio del modello finale

    Componenti chiave:
    -----------------
    - VecNormalize: Normalizza osservazioni e reward per stabilizzare il training
    - CheckpointCallback: Salva il modello ogni 50.000 timesteps
    - TensorBoard: Logging in tempo reale di metriche di performance

    Revisione 2026-10-18:
    --------------------
    I run precedenti da 1M e 2M non stabilizzavano il satellite (RMS > 100
    gradi). Il budget non garantisce convergenza: si parte con un pilota e
    si valutano errore, velocita' e mantenimento oltre alla reward.
    Reward corrente: -4*angle_error^2 -0.5*||omega||^2
    -0.01*||torque/max_torque||^2 + 1 quando entro entrambe le soglie.
    L'episodio dura episode-seconds anche se il bersaglio viene raggiunto.
    Revisione v2: dt configurabile, validazione periodica indipendente
    dalla reward e salvataggio della coppia modello/normalizzatore migliore.
    Revisione v3: aggiunge 1 / (1 + (angle/deg2rad(2))^2 + (||omega||/0.01)^2).
    Il best v2 arrestava il satellite ma a 2.072 gradi (0/5 successi).
    L'efficacia della nuova reward deve essere valutata con un nuovo pilota.
    La reward precedente resta commentata in SatelliteEnv.step.
    """

    # ===== PARSING DEGLI ARGOMENTI DA RIGA DI COMANDO =====
    # Permette di configurare i parametri principali senza modificare il codice.
    # Esempio:
    #   python train.py                         -> 1.000.000 steps (default)
    #   python train.py --timesteps 2000000     -> 2.000.000 steps
    #   python train.py --timesteps 500000 --checkpoint-freq 25000
    parser = argparse.ArgumentParser(
        description="Training agente PPO per controllo d'assetto satellitare",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1_000_000,
        help="Numero totale di timesteps per il training (es. 2000000)"
    )
    parser.add_argument(
        "--checkpoint-freq", "-c",
        type=int,
        default=50_000,
        help="Frequenza di salvataggio dei checkpoint in timesteps"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=3e-4,
        help="Tasso di apprendimento dell'ottimizzatore Adam"
    )
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        default=False,
        help="Attiva la Domain Randomization (variazione inerzia ±10%%)"
    )
    parser.add_argument(
        "--sensor-noise",
        type=float,
        default=0.0,
        help="Deviazione standard del rumore sui sensori (es. 0.01)"
    )
    parser.add_argument(
        "--disturbances",
        action="store_true",
        default=False,
        help="Attiva i disturbi ambientali (gradiente gravitazionale + SRP + rumore)"
    )
    parser.add_argument(
        "--obs-augmented",
        action="store_true",
        default=False,
        help="Attiva l'observation augmentation (aggiunge errore angolare)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed del training")
    parser.add_argument("--dt", type=float, default=0.1, help="Periodo di controllo [s]")
    parser.add_argument("--episode-seconds", type=float, default=60.0,
                        help="Durata fisica degli episodi [s]")
    parser.add_argument("--eval-freq", type=int, default=100000,
                        help="Valutazione fisica ogni N timestep (0 per disabilitare)")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Condizioni iniziali di validazione per ciascuno dei due scenari")
    parser.add_argument("--eval-seed", type=int, default=2000,
                        help="Seed di validazione, distinto dal test finale (1000)")
    args = parser.parse_args()
    try:
        max_steps = episode_steps(args.dt, args.episode_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.timesteps <= 0 or args.checkpoint_freq <= 0:
        parser.error("timesteps e checkpoint-freq devono essere positivi")
    if not np.isfinite(args.sensor_noise) or args.sensor_noise < 0:
        parser.error("sensor-noise deve essere finito e non negativo")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate deve essere finito e positivo")
    if args.seed < 0:
        parser.error("seed deve essere non negativo")
    if args.eval_freq < 0 or args.eval_episodes <= 0 or args.eval_seed < 0:
        parser.error("eval-freq ed eval-seed devono essere non negativi; eval-episodes positivo")
    if args.obs_augmented and args.eval_freq:
        parser.error("La validazione fisica e' 7D: usare --eval-freq 0 con --obs-augmented")
    if args.eval_freq:
        try:
            episode_steps(args.dt, max(args.episode_seconds, 60.0))
            episode_steps(args.dt, max(args.episode_seconds, 80.0))
        except ValueError as exc:
            parser.error(f"Durata di validazione non compatibile con dt: {exc}")

    print("="*70)
    print("TRAINING AGENTE PPO PER CONTROLLO D'ASSETTO SATELLITARE")
    print("Progetto: attitude-ai | Tesi Magistrale in Ingegneria Informatica")
    print("="*70)
    print()
    print("Parametri da riga di comando:")
    print(f"  • Timesteps totali:        {args.timesteps:,}")
    print(f"  • Passo controllo:         {args.dt} s")
    print(f"  • Durata episodio:         {args.episode_seconds} s ({max_steps} step)")
    print(f"  • Validazione fisica:      ogni {args.eval_freq} step (0 = OFF)")
    print(f"  • Frequenza checkpoint:    ogni {args.checkpoint_freq:,} steps")
    print(f"  • Learning rate:           {args.learning_rate:.0e}")
    print(f"  • Domain Randomization:    {'ON' if args.domain_rand else 'OFF'}")
    print(f"  • Sensor Noise (std):      {args.sensor_noise}")
    print(f"  • Disturbi ambientali:     {'ON' if args.disturbances else 'OFF'}")
    print(f"  • Obs. Augmented:          {'ON' if args.obs_augmented else 'OFF'}")
    print()

    # ===== CREAZIONE DELLE DIRECTORY PER IL SALVATAGGIO =====
    # Ogni esecuzione di training viene identificata da un run_id univoco basato
    # su timestamp (formato: run_AAAAMMGG_HHMMSS). Questo garantisce che lanci
    # multipli dello script NON si sovrascrivano mai a vicenda: ogni run produce
    # una propria sottocartella sia in models/ sia in logs/, mantenendo così
    # tracciabilità storica di tutti gli esperimenti condotti per la tesi.
    #
    # Struttura risultante:
    #   models/run_20260821_153000/   <- checkpoint + modello finale + vecnormalize
    #   logs/run_20260821_153000/     <- log TensorBoard di questo specifico run
    # Storico: timestamp al secondo poteva sovrascrivere due lanci ravvicinati.
    # run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    models_dir = os.path.join("models", run_id)
    logs_dir = os.path.join("logs", run_id)
    os.makedirs(models_dir, exist_ok=False)
    os.makedirs(logs_dir, exist_ok=False)
    save_experiment_manifest(models_dir, {
        **vars(args), "max_steps": max_steps, "max_torque": 0.05,
        "canonical_observations": True, "success_hold_s": 10.0,
        "validation_angles_deg": [15.0, 120.0],
        "best_selection": ["success_rate", "-tail_normalized_error", "-control_effort"],
        # Il peso e le scale identificano l'esperimento: non attribuire alla
        # v2 i risultati di questa reward e non confondere bonus con successo.
        "precision_reward": {
            "form": "weight / (1 + (angle_rad/angle_scale_rad)^2 + (omega/omega_scale_rad_s)^2)",
            "weight": PRECISION_REWARD_WEIGHT,
            "angle_scale_rad": float(np.deg2rad(2.0)),
            "omega_scale_rad_s": 0.01,
        },
    })

    print(f"[INFO] Identificatore univoco del run: {run_id}")
    print(f"[INFO] Directory di output create:")
    print(f"       - Modelli:          {models_dir}/")
    print(f"       - Logs TensorBoard: {logs_dir}/")
    print()

    # ===== CONFIGURAZIONE AMBIENTE VETTORIALIZZATO =====
    print("[INFO] Inizializzazione dell'ambiente di training...")

    # DummyVecEnv: Wrapper che trasforma un ambiente singolo in formato vettoriale
    env = DummyVecEnv([make_env(
        domain_rand=args.domain_rand,
        sensor_noise=args.sensor_noise,
        disturbances=args.disturbances,
        obs_augmented=args.obs_augmented,
        dt=args.dt,
        episode_seconds=args.episode_seconds
    )])

    # VecNormalize: Normalizzazione adattiva di osservazioni e reward
    # ----------------------------------------------------------------
    # Motivazione teorica:
    # - Le osservazioni hanno scale diverse (quaternioni [-1,1] vs velocità angolari [rad/s])
    # - Le reward non normalizzate possono avere varianze elevate che destabilizzano il training
    # - La normalizzazione adattiva migliora la convergenza degli algoritmi policy-gradient
    #
    # Parametri:
    # - norm_obs=True: Normalizza le osservazioni con media mobile e std
    # - norm_reward=True: Normalizza le reward con std mobile
    # - clip_obs=10.0: Limita le osservazioni normalizzate a [-10, 10] per stabilità
    # - clip_reward=10.0: Limita le reward normalizzate a [-10, 10]
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    print("[OK] Ambiente configurato con normalizzazione adattiva")
    print()

    # Verifica opzionale delle dipendenze per la progress bar di SB3.
    # Se 'rich' o 'tqdm' non sono disponibili, il training procede comunque
    # disabilitando la barra di avanzamento per evitare ImportError runtime.
    has_tqdm = importlib.util.find_spec("tqdm") is not None
    has_rich = importlib.util.find_spec("rich") is not None
    enable_progress_bar = has_tqdm and has_rich
    if not enable_progress_bar:
        print("[WARN] 'tqdm' e/o 'rich' non trovati: progress bar disattivata.")
        print("       Per abilitarla: pip install tqdm rich")
        print()

    # ===== CONFIGURAZIONE ALGORITMO PPO =====
    print("[INFO] Inizializzazione dell'algoritmo PPO...")
    print()
    print("Iperparametri del modello:")
    print("-" * 50)

    # Definizione degli iperparametri ottimizzati per il problema
    hyperparams = {
        "learning_rate": args.learning_rate, # Tasso di apprendimento di Adam
        "n_steps": 2048,             # Numero di step per rollout
        "batch_size": 64,            # Dimensione mini-batch per training
        "n_epochs": 10,              # Epoche di ottimizzazione per rollout
        "gamma": 0.99,               # Fattore di sconto per reward futuri
        "gae_lambda": 0.95,          # Parametro per Generalized Advantage Estimation
        "clip_range": 0.2,           # Range di clipping per l'obiettivo PPO
        "ent_coef": 0.01,            # Coefficiente di entropia per exploration
        "verbose": 1,                # Livello di verbosità (1 = progress bar)
        "tensorboard_log": logs_dir  # Directory per i log di TensorBoard
    }

    # Stampa dettagliata degli iperparametri con spiegazioni
    print(f"  • Learning Rate: {hyperparams['learning_rate']:.0e}")
    print(f"      Tasso di apprendimento dell'ottimizzatore Adam")
    print(f"  • Steps per Rollout: {hyperparams['n_steps']}")
    print(f"      Numero di transizioni raccolte prima dell'aggiornamento")
    print(f"  • Batch Size: {hyperparams['batch_size']}")
    print(f"      Dimensione dei mini-batch per l'ottimizzazione SGD")
    print(f"  • Epoche: {hyperparams['n_epochs']}")
    print(f"      Numero di passaggi sui dati per ogni aggiornamento")
    print(f"  • Gamma: {hyperparams['gamma']}")
    print(f"      Fattore di sconto per il calcolo del return")
    print(f"  • GAE Lambda: {hyperparams['gae_lambda']}")
    print(f"      Parametro per la stima del vantaggio (bias-variance tradeoff)")
    print(f"  • Clip Range: {hyperparams['clip_range']}")
    print(f"      Vincolo sulla variazione della policy (stabilità PPO)")
    print(f"  • Entropy Coefficient: {hyperparams['ent_coef']}")
    print(f"      Peso del termine di entropia per incentivare l'exploration")
    print()

    # ===== SCELTA DELL'ALGORITMO PPO =====
    # Motivazione:
    # ------------
    # PPO (Proximal Policy Optimization) è scelto per:
    # 1. Stabilità: Il meccanismo di clipping previene aggiornamenti troppo drastici
    # 2. Sample efficiency: Riutilizza i dati raccolti per multiple epoche
    # 3. Robustezza: Meno sensibile al tuning degli iperparametri rispetto a TRPO o A3C
    # 4. Stato dell'arte: Eccellenti prestazioni su task di controllo continuo
    #
    # Alternative considerate:
    # - SAC (Soft Actor-Critic): Ottimo per task off-policy, ma più complesso
    # - TD3 (Twin Delayed DDPG): Richiede più tuning per problemi sparsi come questo
    # - DDPG: Meno stabile, sensibile all'exploration noise

    model = PPO(
        policy="MlpPolicy",  # Policy basata su Multi-Layer Perceptron
        env=env,
        device="cpu",        # Per policy MLP, SB3 è generalmente più efficiente su CPU
        seed=args.seed,
        **hyperparams
    )
    model.control_dt = args.dt
    model.canonical_observations = True

    print("[OK] Modello PPO inizializzato con successo")
    print()

    # ===== CONFIGURAZIONE DEI CALLBACK =====
    print("[INFO] Configurazione dei callback per il training...")

    # CheckpointCallback: Salvataggio periodico del modello
    # -----------------------------------------------------
    # Salva lo stato completo del modello (pesi della rete, statistiche di normalizzazione,
    # stato dell'ottimizzatore) ogni 50.000 timesteps.
    # Essenziale per:
    # - Recupero in caso di interruzioni del training
    # - Analisi dell'evoluzione della policy nel tempo
    # - Selezione del miglior checkpoint in base a metriche di validazione
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,         # Frequenza di salvataggio in timesteps
        save_path=models_dir,           # Cartella di destinazione
        name_prefix="ppo_satellite",    # Prefisso per i file dei checkpoint
        save_replay_buffer=False,       # Non applicabile a PPO (on-policy)
        save_vecnormalize=True          # Salva anche le statistiche di normalizzazione
    )

    print(f"  • Checkpoint salvati ogni {args.checkpoint_freq:,} timesteps in '{models_dir}/'")
    print(f"  • Salvataggio delle statistiche di normalizzazione: attivo")
    print()

    # ===== AVVIO DEL TRAINING =====
    total_timesteps = args.timesteps   # assegnato qui, usato da learn() e dai print sotto
    print("="*70)
    print("INIZIO TRAINING")
    print("="*70)
    print()
    print("Parametri di training:")
    print(f"  • Timesteps addestrati: {total_timesteps:,}")
    print(f"  • Durata stimata: {total_timesteps / 1_000_000 * 2:.1f}-{total_timesteps / 1_000_000 * 4:.1f} ore (su CPU)")
    print(f"  • Monitoraggio in tempo reale via TensorBoard:")
    print(f"      $ tensorboard --logdir={logs_dir}")
    print()
    print("Metriche tracciate su TensorBoard:")
    print("  - rollout/ep_rew_mean: Reward media per episodio")
    print("  - rollout/ep_len_mean: Lunghezza media degli episodi")
    print("  - train/learning_rate: Learning rate corrente")
    print("  - train/entropy_loss: Loss di entropia (exploration)")
    print("  - train/policy_gradient_loss: Loss della policy")
    print("  - train/value_loss: Loss della value function")
    print("  - control/angle_error_deg: Errore medio di assetto")
    print("  - control/omega_magnitude: Velocita' angolare media [rad/s]")
    print("  - control/in_target: Frazione di campioni entro entrambe le soglie")
    print("  - control/episode_success: Successo negli ultimi 10 s dell'episodio")
    print("  - control/reward_precision: Nuovo termine locale (non e' tasso di successo)")
    print("  - validation/success_rate: Successo deterministico sui seed di validazione")
    print("  - validation/tail_omega_rad_s: Velocita' negli ultimi 10 s")
    print("  - validation/tail_angle_deg: Errore negli ultimi 10 s")
    print()
    print("-"*70)
    print()

    # Esecuzione del training
    # Il progresso viene visualizzato automaticamente grazie a verbose=1
    callbacks = [checkpoint_callback, ControlMetricsCallback()]
    if args.eval_freq:
        callbacks.append(PhysicalValidationCallback(
            models_dir, args.eval_freq, args.eval_episodes, args.eval_seed,
            args.dt, args.episode_seconds
        ))
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=enable_progress_bar  # Abilitata solo se dipendenze presenti
    )

    print()
    print("="*70)
    print("TRAINING COMPLETATO CON SUCCESSO!")
    print("="*70)
    print()

    # ===== SALVATAGGIO DEL MODELLO FINALE =====
    final_model_path = os.path.join(models_dir, "ppo_satellite_final")
    model.save(final_model_path)
    print(f"[OK] Modello finale salvato in: {final_model_path}.zip")

    # Salvataggio delle statistiche di normalizzazione
    # (necessarie per utilizzare il modello in fase di test)
    vecnorm_path = os.path.join(models_dir, "vec_normalize.pkl")
    env.save(vecnorm_path)
    with open(os.path.join(models_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "actual_timesteps": model.num_timesteps, "seed": args.seed,
            "dt": args.dt, "episode_seconds": args.episode_seconds,
            "simulated_seconds": model.num_timesteps * args.dt,
        }, handle, indent=2)
    print(f"[OK] Statistiche di normalizzazione salvate in: {vecnorm_path}")
    print()

    # ===== TRACCIAMENTO DELL'ULTIMO RUN =====
    # Scrive un piccolo file "puntatore" (models/latest_run.txt) contenente
    # l'identificativo del run appena concluso. Questo permette a script
    # successivi (es. test_and_plot.py) di individuare automaticamente
    # l'ultimo modello addestrato senza dover specificare manualmente il path,
    # pur mantenendo intatta la cronologia di tutti i run precedenti.
    latest_pointer_path = os.path.join("models", "latest_run.txt")
    with open(latest_pointer_path, "w") as f:
        f.write(run_id)
    print(f"[OK] Puntatore all'ultimo run aggiornato: {latest_pointer_path} -> {run_id}")
    print()

    # ===== RIEPILOGO FINALE =====
    print("="*70)
    print("RIEPILOGO")
    print("="*70)
    print(f"• Run ID: {run_id}")
    print(f"• Timesteps addestrati: {total_timesteps:,}")
    print(f"• Modello salvato: {final_model_path}.zip")
    print(f"• Checkpoint intermedi: {models_dir}/ppo_satellite_*.zip")
    print(f"• Log TensorBoard: {logs_dir}/")
    print()
    print("PROSSIMI PASSI:")
    print("1. Analizzare le curve di training su TensorBoard")
    print("2. Eseguire test_and_plot.py per valutare le prestazioni")
    print("   (caricherà automaticamente questo run tramite models/latest_run.txt)")
    print("3. Confrontare PPO vs LQR in presenza di guasti")
    print()
    print("Per visualizzare i log di training:")
    print(f"  $ tensorboard --logdir=logs")
    print("  Poi aprire: http://localhost:6006")
    print()
    print("="*70)

    # Chiusura dell'ambiente
    env.close()


if __name__ == "__main__":
    """
    Entry point dello script.
    
    Eseguire con:
        python train.py
    
    Requisiti:
        - Installazione corretta di tutte le dipendenze (vedi requirements.txt)
        - Spazio su disco sufficiente per checkpoint (~50-100 MB totali)
        - RAM consigliata: almeno 4 GB
        - GPU opzionale (ma consigliata per accelerare il training)
    
    Note:
        - Il training può essere interrotto (Ctrl+C) e ripreso caricando l'ultimo checkpoint
        - Per riprendere il training: modificare model.learn() per caricare un checkpoint esistente
    """
    main()
