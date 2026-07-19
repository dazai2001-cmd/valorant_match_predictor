from __future__ import annotations

import json
from pathlib import Path

from .config import MODEL_SELECTION_PATH


MODEL_CHOICES = {
    "player": {
        "auto": "Auto",
        "hgb_absolute": "HGB Absolute",
        "hgb_squared": "HGB Squared",
        "extra_trees": "Extra Trees",
        "last_10_baseline": "Hierarchical Form Baseline",
    },
    "team": {
        "auto": "Auto",
        "logistic_classifier": "Logistic Classifier",
        "hgb_classifier": "HGB Classifier",
        "hgb_residual": "HGB Elo-Residual",
        "elo_baseline": "Elo / Form Baseline",
    },
    "map": {
        "auto": "Auto",
        "logistic_classifier": "Logistic Classifier",
        "hgb_classifier": "HGB Classifier",
        "hgb_residual": "HGB Map-Residual",
        "map_form_baseline": "Team-Anchored Map Baseline",
    },
}

DEFAULT_MODEL_SELECTION = {task: "auto" for task in MODEL_CHOICES}


def normalize_model_selection(selection: dict | None) -> dict[str, str]:
    selection = selection or {}
    output = {}
    for task, choices in MODEL_CHOICES.items():
        value = str(selection.get(task, "auto")).strip()
        output[task] = value if value in choices else "auto"
    return output


def load_model_selection(path: str | Path = MODEL_SELECTION_PATH) -> dict[str, str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = {}
    return normalize_model_selection(payload)


def save_model_selection(
    selection: dict,
    path: str | Path = MODEL_SELECTION_PATH,
) -> dict[str, str]:
    normalized = normalize_model_selection(selection)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def selected_candidate(
    task: str,
    requested: str,
    recommended: str,
) -> tuple[str, str]:
    normalized = normalize_model_selection({task: requested})[task]
    if normalized == "auto":
        return recommended, "auto"
    return normalized, "manual"
