"""Rigenera tabelle e figure del capitolo 5 dagli esperimenti archiviati."""

import csv
import hashlib
import json
from pathlib import Path
from shutil import copy2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tesi" / "risultati_capitolo5"
CAMPAIGNS = {
    42: "test_20261018_180423_678353",
    43: "test_20261018_183007_799072",
    44: "test_20261018_190447_807096",
    45: "test_20261018_191804_938788",
    46: "test_20261018_193100_975275",
}
SCENARIOS = {
    "nominale": "Nominale (15 gradi)",
    "non_lineare": "Grande rotazione (120 gradi)",
    "guasto": "Guasto X 70% + rumore",
}
METRICS = (
    "rms_error_deg", "steady_state_error_deg", "settling_time_s",
    "energy_consumption", "post_fault_rms_deg",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_campaigns():
    campaigns = {}
    provenance = []
    reference_lqr = None
    expected_keys = {
        (scenario, controller, str(seed))
        for scenario in SCENARIOS for controller in ("LQR", "PPO")
        for seed in range(3000, 3030)
    }
    for training_seed, test_id in CAMPAIGNS.items():
        directory = ROOT / "tests" / test_id
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["configuration"]
        training = config["training_manifest"]["configuration"]
        if (
            manifest["revision"] != "2026-10-18-local-precision-v3"
            or config["model"] != "best" or config["dt"] != 0.05
            or config["episodes"] != 30
            or config["evaluation_seeds"] != list(range(3000, 3030))
            or training["seed"] != training_seed
            or any(training[key] for key in (
                "domain_rand", "disturbances", "obs_augmented", "sensor_noise"
            ))
        ):
            raise ValueError(f"Protocollo inatteso: {directory}")
        rows = read_csv(directory / "episodi.csv")
        keys = {(row["scenario"], row["controller"], row["seed"]) for row in rows}
        if keys != expected_keys or len(rows) != len(expected_keys):
            raise ValueError(f"Episodi mancanti o duplicati: {directory}")
        lqr = sorted(
            (row for row in rows if row["controller"] == "LQR"),
            key=lambda row: (row["scenario"], row["seed"]),
        )
        # Gli stessi 30 episodi LQR sono ripetuti nei cinque report:
        # conservarli una sola volta, non fingere 150 repliche indipendenti.
        if reference_lqr is not None and lqr != reference_lqr:
            raise ValueError(f"La baseline LQR non coincide: {directory}")
        reference_lqr = lqr
        model_dir = ROOT / "models" / Path(config["model_path"]).parent.name
        history_path = model_dir / "validation_history.jsonl"
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
        ]
        best = config["best_validation"]
        # Il seed 42 aveva budget 1.2M: verificare che limitare le opportunita'
        # di selezione al comune budget ~1M conservi lo stesso checkpoint.
        within_budget = [record for record in history if record["timesteps"] <= 1_001_472]
        best_common = max(within_budget, key=lambda record: tuple(record["selection_score"]))
        if best_common["timesteps"] != best["timesteps"]:
            raise ValueError(f"Best non confrontabile entro il budget comune: {training_seed}")
        campaigns[training_seed] = {
            "directory": directory, "rows": rows, "history": history, "config": config,
        }
        provenance.append({
            "training_seed": training_seed,
            "test_id": test_id,
            "training_run": model_dir.name,
            "requested_timesteps": training["timesteps"],
            "selected_timesteps": best["timesteps"],
            "best_within_common_budget": True,
            "model_sha256": config["model_sha256"],
            "normalizer_sha256": config["normalizer_sha256"],
            "inputs": [
                fingerprint(manifest_path), fingerprint(directory / "episodi.csv"),
                fingerprint(history_path),
            ],
        })
    return campaigns, provenance


def per_run_summary(campaigns):
    summaries = []
    for controller in ("LQR", "PPO"):
        seeds = (42,) if controller == "LQR" else tuple(CAMPAIGNS)
        for training_seed in seeds:
            campaign = campaigns[training_seed]
            for scenario in SCENARIOS:
                group = [
                    row for row in campaign["rows"]
                    if row["controller"] == controller and row["scenario"] == scenario
                ]
                successes = sum(row["success"] == "True" for row in group)
                record = {
                    "controller": controller,
                    "training_seed": training_seed if controller == "PPO" else "",
                    "scenario": scenario,
                    "n_episodes": len(group),
                    "n_success": successes,
                    "success_rate": successes / len(group),
                    "n_settled_angle": sum(row["settled"] == "True" for row in group),
                }
                for metric in METRICS:
                    values = [
                        float(row[metric]) for row in group
                        if row[metric] not in ("", "N/C", "N/A")
                    ]
                    record[f"{metric}_mean"] = float(np.mean(values)) if values else ""
                    record[f"{metric}_n_valid"] = len(values)
                summaries.append(record)
    return summaries


def across_run_summary(summaries):
    rows = []
    for scenario in SCENARIOS:
        for controller in ("LQR", "PPO"):
            group = [
                row for row in summaries
                if row["scenario"] == scenario and row["controller"] == controller
            ]
            for metric in ("success_rate",) + tuple(f"{name}_mean" for name in METRICS):
                values = [float(row[metric]) for row in group if row[metric] != ""]
                rows.append({
                    "scenario": scenario, "controller": controller, "metric": metric,
                    "n_training_runs": len(group) if controller == "PPO" else 0,
                    "n_values": len(values),
                    "mean_of_run_means": float(np.mean(values)) if values else "",
                    "std_between_run_means": (
                        float(np.std(values, ddof=1))
                        if controller == "PPO" and len(values) > 1 else ""
                    ),
                })
    return rows


def save_figure(fig, name):
    fig.savefig(OUTPUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_success(summaries):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (scenario, label) in zip(axes, SCENARIOS.items()):
        ppo = [row for row in summaries if row["controller"] == "PPO"
               and row["scenario"] == scenario]
        lqr = next(row for row in summaries if row["controller"] == "LQR"
                   and row["scenario"] == scenario)
        values = [100 * row["success_rate"] for row in ppo]
        ax.bar(range(5), values, color=["tab:blue"] * 4 + ["tab:orange"])
        ax.axhline(100 * lqr["success_rate"], color="black", ls="--", label="LQR")
        for x, row in enumerate(ppo):
            ax.text(x, values[x] + 2, f'{row["n_success"]}/30', ha="center", fontsize=9)
        ax.set(xticks=range(5), xticklabels=list(CAMPAIGNS),
               xlabel="Seed di training PPO", ylabel="Successo [%]",
               title=label, ylim=(0, 115))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    save_figure(fig, "fig5_1_successi_per_seed.png")


def plot_tradeoff(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, scenario in zip(axes, ("nominale", "non_lineare")):
        for row in (r for r in summaries if r["scenario"] == scenario):
            lqr = row["controller"] == "LQR"
            x, y = row["energy_consumption_mean"], row["rms_error_deg_mean"]
            label = "LQR" if lqr else f'Seed {row["training_seed"]}'
            ax.scatter(x, y, marker="*" if lqr else "o", s=140 if lqr else 60,
                       color="black" if lqr else (
                           "tab:orange" if row["training_seed"] == 46 else "tab:blue"
                       ))
            ax.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set(xscale="log", xlabel="Integrale coppia applicata al quadrato [N^2 m^2 s]",
               ylabel="Errore RMS totale medio [gradi]", title=SCENARIOS[scenario])
        ax.grid(alpha=0.25)
    save_figure(fig, "fig5_2_compromesso_sforzo_errore.png")


def plot_validation(campaigns):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (seed, campaign) in enumerate(campaigns.items()):
        history = campaign["history"]
        steps = [record["timesteps"] for record in history]
        axes[0].plot(steps, [100 * record["success_rate"] for record in history],
                     ".-", color=colors(index), label=f"Seed {seed}")
        axes[1].plot(
            steps, [np.mean([episode["tail_angle_deg"] for episode in record["episodes"]])
                    for record in history],
            ".-", color=colors(index),
        )
        best = campaign["config"]["best_validation"]
        axes[0].scatter(best["timesteps"], 100 * best["success_rate"], marker="*",
                        s=150, color=colors(index), edgecolors="black", zorder=4)
    axes[0].set(ylabel="Successo in validazione [%]", ylim=(-5, 108))
    axes[0].legend(ncol=3)
    axes[1].axhline(2, color="black", ls="--", label="Soglia angolare 2 gradi")
    axes[1].set(yscale="log", xlabel="Timestep di training",
                ylabel="Errore medio negli ultimi 10 s [gradi]")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure(fig, "fig5_3_evoluzione_validazione.png")


def plot_failure(campaigns):
    seed = 3013
    names = {
        "LQR": campaigns[42]["directory"] / f"non_lineare_LQR_seed{seed}.npz",
        "PPO training 42": campaigns[42]["directory"] / f"non_lineare_PPO_seed{seed}.npz",
        "PPO training 46": campaigns[46]["directory"] / f"non_lineare_PPO_seed{seed}.npz",
    }
    inputs = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for name, path in names.items():
        inputs.append(fingerprint(path))
        with np.load(path) as data:
            axes[0].plot(data["time"], data["angle_error"], label=name)
            axes[1].plot(data["time"], np.linalg.norm(data["omega"], axis=1), label=name)
    axes[0].axhline(2, color="black", ls="--", label="Soglia 2 gradi")
    axes[1].axhline(0.01, color="black", ls="--", label="Soglia 0.01 rad/s")
    axes[0].set(ylabel="Errore di assetto [gradi]",
                title="Stessa condizione iniziale: 120 gradi, seed di valutazione 3013")
    axes[1].set(xlabel="Tempo [s]", ylabel="Norma velocita' angolare [rad/s]")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    save_figure(fig, "fig5_4_caso_critico_seed46.png")
    return inputs


def export_test_sections(campaigns, summaries):
    sections = {
        "nominale": ("test1_nominale", ("test1_nominale.png",)),
        "non_lineare": ("test2_non_lineare", ("test2_non_lineare.png",)),
        "guasto": (
            "test3_robustezza",
            ("test3_traiettoria_guasto.png", "test3_robustezza_sweep.png"),
        ),
    }
    for scenario, (folder, figures) in sections.items():
        directory = OUTPUT / folder
        directory.mkdir(parents=True, exist_ok=True)
        episodes = []
        for training_seed, campaign in campaigns.items():
            for row in campaign["rows"]:
                if row["scenario"] != scenario:
                    continue
                if row["controller"] == "LQR" and training_seed != 42:
                    continue
                episodes.append({
                    "training_seed": training_seed if row["controller"] == "PPO" else "",
                    **row,
                })
        write_csv(directory / "episodi.csv", episodes)
        write_csv(directory / "risultati_per_seed.csv", [
            row for row in summaries if row["scenario"] == scenario
        ])
        for figure in figures:
            copy2(campaigns[42]["directory"] / figure, directory / figure)
        if scenario == "guasto":
            sweep = [
                {"training_seed": seed, **row}
                for seed, campaign in campaigns.items()
                for row in read_csv(campaign["directory"] / "robustezza_sweep.csv")
            ]
            write_csv(directory / "robustezza_sweep_per_seed.csv", sweep)


def main():
    campaigns, provenance = load_campaigns()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = per_run_summary(campaigns)
    write_csv(OUTPUT / "risultati_per_seed.csv", summaries)
    write_csv(OUTPUT / "sintesi_tra_seed.csv", across_run_summary(summaries))
    plot_success(summaries)
    plot_tradeoff(summaries)
    plot_validation(campaigns)
    failure_inputs = plot_failure(campaigns)
    export_test_sections(campaigns, summaries)
    with (OUTPUT / "provenienza.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "generator": fingerprint(Path(__file__).resolve()),
            "campaigns": provenance, "failure_figure_inputs": failure_inputs,
            "aggregation": (
                "PPO: media e SD campionaria delle medie per training seed; "
                "settling: soli episodi convergenti; LQR: 30 episodi unici per scenario"
            ),
        }, handle, indent=2, allow_nan=False)
    print(f"Generati CSV, figure, cartelle dei tre test e provenienza.json in {OUTPUT}")


if __name__ == "__main__":
    main()
