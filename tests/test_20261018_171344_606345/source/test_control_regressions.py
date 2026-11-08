"""Regressioni della revisione 2026-10-18, senza dipendenze di test aggiuntive."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.env_checker import check_env

from experiment_history import EXPERIMENT_REVISION, save_experiment_manifest
from lqr_controller import LQRController
from satellite_dynamics import SatelliteDynamics, canonicalize_observation
from satellite_env import SatelliteEnv
from train import PhysicalValidationCallback, episode_steps, make_env
from test_and_plot import (
    compute_metrics,
    format_settling_time,
    generate_initial_state,
    load_ppo_model,
    run_repeated_evaluation,
    run_simulation,
)


class ZeroPolicy:
    canonical_observations = True

    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic=True):
        self.observations.append(observation.copy())
        return np.zeros((1, 3)), None


class ControlRegressionTests(unittest.TestCase):
    def test_physical_episode_duration_is_preserved(self):
        self.assertEqual(episode_steps(0.1, 60), 600)
        self.assertEqual(episode_steps(0.05, 60), 1200)
        env = make_env(dt=0.05)()
        self.assertEqual(env.unwrapped.max_steps, 1200)
        self.assertEqual(env.unwrapped.dt, 0.05)
        env.close()
        for dt in (0, -0.1, float("nan"), 0.07):
            with self.assertRaises(ValueError):
                episode_steps(dt, 60)

    def test_validation_selection_prioritizes_success_then_physical_error(self):
        score = PhysicalValidationCallback.selection_score
        self.assertGreater(score(1, 10, 10), score(0, 0, 0))
        self.assertGreater(score(0, 2, 10), score(0, 3, 0))
        self.assertGreater(score(0, 2, 1), score(0, 2, 10))

    def test_model_dt_is_shared_by_both_controllers(self):
        model = ZeroPolicy()
        model.control_dt = 0.05
        for controller in ("LQR", "PPO"):
            data = run_simulation(
                controller, generate_initial_state(15), model=model, total_time=10
            )
            self.assertEqual(len(data["time"]), 201)
            self.assertEqual(data["time"][-1], 10)
            self.assertAlmostEqual(data["time"][1], 0.05)
        diagnostic = run_simulation(
            "PPO", generate_initial_state(15), model=model, total_time=10, dt=0.1
        )
        self.assertEqual(len(diagnostic["time"]), 101)

    def test_physical_validation_preserves_normalizer_and_saves_best_pair(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from experiment_history import save_experiment_manifest

        with tempfile.TemporaryDirectory() as directory:
            env = VecNormalize(DummyVecEnv([lambda: SatelliteEnv(dt=0.05)]))
            model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, seed=42, device="cpu")
            model.set_logger(configure(format_strings=[]))
            model.control_dt = 0.05
            model.canonical_observations = True
            env.reset()
            obs_count = env.obs_rms.count
            reward_count = env.ret_rms.count
            callback = PhysicalValidationCallback(directory, 100, 1, 2000, 0.05, 60)
            callback.init_callback(model)
            callback.num_timesteps = 100

            def trajectory(*args, **kwargs):
                duration = kwargs["total_time"]
                dt = kwargs["dt"]
                n = int(round(duration / dt))
                return {
                    "time": np.arange(n + 1) * dt, "angle_error": np.zeros(n + 1),
                    "omega": np.zeros((n + 1, 3)), "torques": np.zeros((n, 3)),
                    "states": np.zeros((n + 1, 7)),
                }

            with patch("test_and_plot.run_simulation", side_effect=trajectory):
                callback._evaluate("periodic")
                callback._on_training_end()
            self.assertEqual(env.obs_rms.count, obs_count)
            self.assertEqual(env.ret_rms.count, reward_count)
            self.assertTrue(env.training)
            best = json.loads((Path(directory) / "best_validation.json").read_text())
            self.assertEqual(best["success_rate"], 1)
            self.assertEqual(best["phase"], "periodic")
            lines = (Path(directory) / "validation_history.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])["phase"], "final")
            self.assertFalse(json.loads(lines[-1])["new_best"])
            save_experiment_manifest(directory, {
                "dt": 0.05, "canonical_observations": True
            })
            loaded, norm, path = load_ppo_model(run_dir=directory, model_kind="best")
            self.assertTrue(path.endswith("ppo_satellite_best.zip"))
            self.assertEqual(loaded.control_dt, 0.05)
            self.assertFalse(norm.training)
            self.assertEqual(norm.obs_rms.count, obs_count)
            norm.close()
            env.close()

    def test_missing_best_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Modello best assente"):
                load_ppo_model(run_dir=directory, model_kind="best")

    def test_sampled_closed_loop_is_stable(self):
        for dt in (0.05, 0.1, 0.2):
            controller = LQRController(dt=dt)
            poles = np.linalg.eigvals(controller.A_d - controller.B_d @ controller.K)
            self.assertLess(np.max(np.abs(poles)), 1)

    def test_lqr_stabilizes_both_scenarios(self):
        for angle in (15, 120):
            data = run_simulation("LQR", generate_initial_state(angle), total_time=20)
            metrics = compute_metrics(data)
            self.assertTrue(metrics["success"], (angle, metrics))
            self.assertLess(metrics["steady_state_error_deg"], 0.01)
            self.assertLessEqual(metrics["peak_torque_Nm"], 0.05)

    def test_equivalent_quaternions_have_identical_commands(self):
        for state in (generate_initial_state(120), np.array([0., -1, 0, 0, 0, 0, 0])):
            equivalent = state.copy()
            equivalent[:4] *= -1
            np.testing.assert_array_equal(
                canonicalize_observation(state), canonicalize_observation(equivalent)
            )
            controller = LQRController()
            np.testing.assert_allclose(
                controller.compute_control(state), controller.compute_control(equivalent)
            )

    def test_gymnasium_contract(self):
        check_env(SatelliteEnv(), warn=True)
        check_env(SatelliteEnv(observation_augmented=True), warn=True)

    def test_success_requires_holding_not_early_termination(self):
        env = SatelliteEnv(max_steps=120)
        env.reset(seed=42)
        env.dynamics.reset()
        for step in range(120):
            _, reward, terminated, truncated, info = env.step(np.zeros(3))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 119)
            self.assertEqual(info["success"], step >= 100)
            self.assertAlmostEqual(reward, 1)
        env.dynamics.reset(generate_initial_state(15))
        _, _, _, _, info = env.step(np.zeros(3))
        self.assertFalse(info["success"])

    def test_high_angular_velocity_remains_penalized(self):
        env = SatelliteEnv()
        rewards = []
        for omega in (2., 4.):
            env.reset(seed=42)
            env.dynamics.reset(np.array([1., 0, 0, 0, omega, 0, 0]))
            _, _, _, _, info = env.step(np.zeros(3))
            rewards.append(info["reward_stability"])
        self.assertLess(rewards[1], rewards[0])
        self.assertTrue(env.observation_space.contains(
            np.array([1, 0, 0, 0, 200, 0, 0], dtype=np.float32)
        ))

    def test_simulation_records_terminal_state_and_interval_torques(self):
        data = run_simulation("LQR", generate_initial_state(15), total_time=10)
        self.assertEqual(len(data["time"]), 101)
        self.assertEqual(data["time"][-1], 10)
        self.assertEqual(data["states"].shape, (101, 7))
        self.assertEqual(data["torques"].shape, (100, 3))

    def test_invalid_action_does_not_reach_dynamics(self):
        env = SatelliteEnv()
        env.reset(seed=42)
        state = env.dynamics.state.copy()
        for action in ([2, 0, 0], [0, float("nan"), 0], [0, 0]):
            with self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(state, env.dynamics.state)

    def test_seed_is_local_and_reproducible(self):
        env = SatelliteEnv(
            domain_randomization=True, sensor_noise_std=0.005,
            disturbances={"noise_std": 1e-5}
        )
        other = SatelliteEnv(sensor_noise_std=0.1)
        np.random.seed(987)
        global_state = np.random.get_state()
        initial, _ = env.reset(seed=42)
        step = env.step(np.zeros(3))
        other.reset(seed=123)
        other.step(np.ones(3))
        repeated, _ = env.reset(seed=42)
        repeated_step = env.step(np.zeros(3))
        np.testing.assert_array_equal(initial, repeated)
        np.testing.assert_array_equal(step[0], repeated_step[0])
        self.assertEqual(step[1], repeated_step[1])
        np.testing.assert_array_equal(global_state[1], np.random.get_state()[1])

    def test_disturbance_is_constant_inside_integration_step(self):
        dynamics = SatelliteDynamics(
            disturbances={"noise_std": 1e-5}, rng=np.random.default_rng(42)
        )
        dynamics.step(np.zeros(3), 0.1)
        first = dynamics._compute_disturbance_torque(0, dynamics.state)
        second = dynamics._compute_disturbance_torque(0.05, dynamics.state)
        np.testing.assert_array_equal(first, second)
        dynamics.step(np.zeros(3), 0.1)
        self.assertFalse(np.array_equal(first, dynamics._step_noise))

    def test_controller_comparison_uses_identical_sensor_noise(self):
        observed = []
        model = ZeroPolicy()
        with patch("test_and_plot.LQRController") as controller:
            def capture(obs):
                observed.append(canonicalize_observation(obs))
                return np.zeros(3)
            controller.return_value.compute_control.side_effect = capture
            options = dict(
                total_time=1, sensor_noise_std=0.005, seed=1000,
                disturbances={"noise_std": 1e-5}
            )
            lqr = run_simulation("LQR", generate_initial_state(15), **options)
            ppo = run_simulation("PPO", generate_initial_state(15), model=model, **options)
        np.testing.assert_array_equal(lqr["states"], ppo["states"])
        np.testing.assert_allclose(np.array(observed), np.concatenate(model.observations), atol=1e-7)

    def test_no_convergence_is_explicit(self):
        data = {
            "time": np.arange(200) * 0.1, "angle_error": np.full(200, 3.),
            "omega": np.zeros((200, 3)), "torques": np.zeros((200, 3))
        }
        metrics = compute_metrics(data)
        self.assertTrue(np.isnan(metrics["settling_time_s"]))
        self.assertEqual(format_settling_time(metrics["settling_time_s"]), "N/C")
        self.assertFalse(metrics["success"])
        data["angle_error"][-1] = 0
        self.assertFalse(compute_metrics(data)["settled"])
        data["angle_error"][:] = 0
        data["omega"][:, 0] = 0.1
        self.assertTrue(compute_metrics(data)["settled"])
        self.assertFalse(compute_metrics(data)["success"])

    def test_manifest_preserves_sources_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_experiment_manifest(directory, {"seed": 42})
            self.assertEqual(manifest["revision"], EXPERIMENT_REVISION)
            saved = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(saved["configuration"]["seed"], 42)
            self.assertTrue((Path(directory) / "source" / "satellite_env.py").exists())
            self.assertEqual(len(saved["source_sha256"]["satellite_env.py"]), 64)
            with self.assertRaises(FileExistsError):
                save_experiment_manifest(directory, {})

    def test_legacy_model_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "allow-legacy-model"):
                load_ppo_model(run_dir=directory)

    def test_missing_normalizer_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("test_and_plot.PPO.load") as load:
                load.return_value.observation_space.shape = (7,)
                with self.assertWarns(UserWarning):
                    with self.assertRaisesRegex(FileNotFoundError, "Normalizzatore"):
                        load_ppo_model(run_dir=directory, allow_legacy=True)

    def test_repeated_evaluation_exports_counts_and_raw_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("test_and_plot.run_simulation") as simulation:
                simulation.return_value = {
                    "time": np.arange(601) * 0.1, "angle_error": np.zeros(601),
                    "omega": np.zeros((601, 3)), "torques": np.zeros((600, 3)),
                    "states": np.zeros((601, 7)),
                }
                run_repeated_evaluation(ZeroPolicy(), None, directory, 1000, 2)
            with (Path(directory) / "episodi.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["seed"] for row in rows}, {"1000", "1001"})
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 12)


if __name__ == "__main__":
    unittest.main()
