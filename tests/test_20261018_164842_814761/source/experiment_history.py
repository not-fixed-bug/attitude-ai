"""Metadati e snapshot locali per ricostruire ogni esperimento."""

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


EXPERIMENT_REVISION = "2026-10-18-discrete-hold-v1"


def save_experiment_manifest(output_dir, configuration):
    """Salva configurazione, dipendenze, commit di partenza e sorgenti esatti."""
    root = Path(__file__).resolve().parent
    destination = Path(output_dir)
    snapshot = destination / "source"
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in sorted(root.glob("*.py")) + [root / "requirements.txt"]:
        shutil.copy2(source, snapshot / source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, text=True
    )
    if diff.returncode == 0:
        (snapshot / "working_tree.patch").write_text(diff.stdout, encoding="utf-8")
    manifest = {
        "revision": EXPERIMENT_REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "gymnasium", "stable-baselines3", "torch")
        },
        "source_sha256": hashes,
        "configuration": configuration,
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return manifest
