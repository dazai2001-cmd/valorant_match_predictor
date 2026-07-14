import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.form_calculations import calculate_team_form, clean_match_data
from ..config import MAP_MODEL_PATH
from ..features.team_context import (
    current_team_context,
    normalize_lineup_identity,
    team_context_from_state,
)
from .train_models import (
    PLAYER_FEATURES,
    PLAYER_MODEL_PATH,
    TEAM_FEATURES,
    TEAM_MODEL_PATH,
    apply_probability_blend_temperature,
    apply_symmetric_calibration,
    map_feature_frame,
    team_feature_row,
    team_rows_from_cleaned,
)


def load_model_payload(path: str) -> dict | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def cached_team_context(
    team1: str,
    team2: str,
    lineup1: set[str],
    lineup2: set[str],
    dataset_fingerprint: str,
    as_of_date=None,
    current_patch: str = "",
    model_path: str = TEAM_MODEL_PATH,
) -> dict | None:
    payload = load_model_payload(model_path)
    if payload is None or payload.get("sequential_state") is None:
        return None
    if payload.get("metadata", {}).get("dataset_fingerprint") != dataset_fingerprint:
        return None
    return team_context_from_state(
        payload["sequential_state"],
        team1,
        team2,
        lineup_a={normalize_lineup_identity(player) for player in lineup1},
        lineup_b={normalize_lineup_identity(player) for player in lineup2},
        as_of_date=as_of_date,
        current_patch=current_patch,
    )


def apply_player_model(
    profiles: pd.DataFrame,
    matches: pd.DataFrame,
    team1: str,
    team2: str,
    model_path: str = PLAYER_MODEL_PATH,
) -> pd.DataFrame:
    payload = load_model_payload(model_path)
    if payload is None or profiles.empty:
        return profiles

    cleaned = clean_match_data(matches)
    output = profiles.copy()
    team_forms = {
        team: calculate_team_form(cleaned, team)
        for team in {team1, team2}
    }
    cleaned_player_names = cleaned["player"].astype(str).str.lower()
    feature_rows = []
    for _, row in output.iterrows():
        team_form = team_forms.get(row["team"])
        if team_form is None:
            team_form = calculate_team_form(cleaned, row["team"])
        opponent = team2 if row["team"] == team1 else team1
        opponent_form = team_forms.get(opponent)
        if opponent_form is None:
            opponent_form = calculate_team_form(cleaned, opponent)
        player_vs_opponent = cleaned[
            (cleaned_player_names == str(row["player"]).lower())
            & (cleaned["opponent"] == opponent)
        ]

        feature_rows.append(
            {
                "player_last_3_rating": row.get("last_3_rating", row.get("base_rating", 1.0)),
                "player_last_5_rating": row.get("last_5_rating", row.get("base_rating", 1.0)),
                "player_last_10_rating": row.get("last_10_rating", row.get("base_rating", 1.0)),
                "player_60d_rating": row.get("recent_60d_rating", row.get("raw_form_rating", row.get("base_rating", 1.0))),
                "player_overall_rating": row.get("overall_rating", row.get("base_rating", 1.0)),
                "player_rating_trend": row.get("rating_trend", 0.0),
                "player_recent_acs": row.get("avg_acs", 200.0),
                "player_recent_kd": row.get("kd_ratio", 1.0),
                "player_recent_assists": row.get("avg_assists", 5.0),
                "player_maps": row.get("total_maps", row.get("recent_maps", 0)),
                "player_60d_maps": row.get("maps_60d", row.get("recent_maps", 0)),
                "player_effective_maps": row.get("effective_maps", row.get("recent_maps", 0)),
                "player_days_since_last_match": row.get("days_since_last_match", 365.0),
                "player_freshness": row.get("freshness", 0.0),
                "player_rating_std": row.get("rating_std", 0.18),
                "player_agent_pool_size": row.get("agent_pool_size", 0),
                "team_recent_rating": team_form["team_recent_rating"],
                "team_recent_win_rate": team_form["recent_win_rate"],
                "opponent_recent_rating": opponent_form["team_recent_rating"],
                "opponent_recent_win_rate": opponent_form["recent_win_rate"],
                "player_vs_opponent_rating": player_vs_opponent["rating_for_model"].mean()
                if not player_vs_opponent.empty
                else row.get("overall_rating", row.get("base_rating", 1.0)),
                "player_vs_opponent_maps": len(player_vs_opponent),
            }
        )

    x = (
        pd.DataFrame(feature_rows)[payload["features"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    model_reliability = float(
        payload.get("metadata", {}).get("player_metrics", {}).get("model_reliability", 0.0)
    )
    correction_weight = max(
        0.0,
        min(1.0, float(payload.get("correction_weight", model_reliability))),
    )
    baseline = pd.to_numeric(
        output.get("last_10_rating", output.get("base_rating", 1.0)),
        errors="coerce",
    ).fillna(1.0)
    residual = payload["model"].predict(x)
    output["trained_rating_correction"] = correction_weight * residual
    output["trained_base_rating"] = (
        baseline + output["trained_rating_correction"]
    ).clip(0.45, 1.70)
    lower_model = payload.get("lower_model")
    upper_model = payload.get("upper_model")
    if lower_model is not None and upper_model is not None:
        adjustment = float(payload.get("interval_adjustment", 0.0))
        lower_raw = baseline.to_numpy(dtype=float) + lower_model.predict(x)
        upper_raw = baseline.to_numpy(dtype=float) + upper_model.predict(x)
        output["trained_rating_low"] = (
            pd.Series(lower_raw, index=output.index).where(lower_raw <= upper_raw, upper_raw)
            - adjustment
        ).clip(0.35, 1.90)
        output["trained_rating_high"] = (
            pd.Series(upper_raw, index=output.index).where(upper_raw >= lower_raw, lower_raw)
            + adjustment
        ).clip(0.35, 1.90)
    output["base_rating"] = output["trained_base_rating"]
    output["trained_model_reliability"] = model_reliability
    output["trained_residual_std"] = float(payload.get("residual_std", 0.18))
    return output


def predict_team_win_probability(
    matches: pd.DataFrame,
    team1: str,
    team2: str,
    lineup1: set[str] | None = None,
    lineup2: set[str] | None = None,
    team_maps: pd.DataFrame | None = None,
    sequential_context: dict | None = None,
    target_date=None,
    current_patch: str = "",
    model_path: str = TEAM_MODEL_PATH,
) -> tuple[float | None, dict | None]:
    payload = load_model_payload(model_path)
    if payload is None:
        return None, None

    cleaned = clean_match_data(matches)
    if team_maps is None:
        team_maps = team_rows_from_cleaned(cleaned)
    if team_maps.empty:
        return None, None

    target = pd.Series(
        {
            "match_date_sort": pd.to_datetime(target_date, utc=True, errors="coerce")
            if target_date is not None
            else (
                team_maps["match_date_sort"].max() + pd.Timedelta(days=1)
                if team_maps["match_date_sort"].notna().any()
                else pd.NaT
            ),
            "match_id": team_maps["match_id"].max() + 1
            if team_maps["match_id"].notna().any()
            else None,
        }
    )
    features = team_feature_row(team_maps, target, team1, team2)
    if features is None:
        return None, None

    normalized_lineup1 = {normalize_lineup_identity(player) for player in lineup1} if lineup1 else None
    normalized_lineup2 = {normalize_lineup_identity(player) for player in lineup2} if lineup2 else None
    sequential = dict(sequential_context) if sequential_context is not None else current_team_context(
        cleaned,
        team_maps,
        team1,
        team2,
        lineup_a=normalized_lineup1,
        lineup_b=normalized_lineup2,
        as_of_date=target_date,
        current_patch=current_patch,
    )
    map_probabilities = sequential.pop("map_probabilities", {})
    features.update(sequential)

    x = pd.DataFrame([features])[payload["features"]].fillna(0.0)
    elo_probability = float(features.get("elo_probability", 0.5))
    model_kind = payload.get("model_kind", "residual")
    if model_kind == "baseline":
        raw_probability = elo_probability
        raw_residual = 0.0
    elif model_kind == "classifier":
        raw_probability = float(payload["model"].predict_proba(x)[0, 1])
        raw_residual = raw_probability - elo_probability
    else:
        raw_residual = float(payload["model"].predict(x)[0])
        raw_probability = min(0.97, max(0.03, elo_probability + raw_residual))
    calibrator = payload.get("calibrator")
    if model_kind == "baseline":
        probability = raw_probability
    elif "model_blend_weight" in payload:
        probability = float(
            apply_probability_blend_temperature(
                [raw_probability],
                [elo_probability],
                blend_weight=payload.get("model_blend_weight", 1.0),
                temperature=payload.get("probability_temperature", 1.0),
                max_model_delta=payload.get("max_model_delta", 0.20),
            )[0]
        )
    else:
        probability = float(apply_symmetric_calibration(calibrator, [raw_probability])[0])
    metrics = payload.get("metadata", {}).get("team_metrics", {})
    features["raw_model_probability"] = raw_probability
    features["raw_model_residual"] = raw_residual
    features["model_reliability"] = metrics.get("model_reliability", 0.0)
    features["active_model_candidate"] = metrics.get("selected_candidate", "")
    features["manual_model_override"] = metrics.get("selection_mode") == "manual"
    features["active_model_is_baseline"] = model_kind == "baseline"
    features["map_probabilities"] = map_probabilities
    return probability, features


def predict_map_win_probabilities(
    team_features: dict,
    map_order: list[str],
    picked_by: list[str] | None = None,
    model_path: str = MAP_MODEL_PATH,
) -> tuple[dict[str, float], dict]:
    baseline_probabilities = dict(team_features.get("map_probabilities", {}))
    elo_probability = float(team_features.get("elo_probability", 0.5))
    payload = load_model_payload(model_path)
    if payload is None or not map_order:
        return {
            map_name: float(baseline_probabilities.get(map_name, elo_probability))
            for map_name in map_order
        }, {"map_model_used": False, "map_model_reliability": 0.0}

    picked_by = picked_by or [""] * len(map_order)
    rows = []
    for index, map_name in enumerate(map_order, start=1):
        picker = picked_by[index - 1] if index - 1 < len(picked_by) else ""
        row = {
            **team_features,
            "map_name": map_name,
            "map_number": float(index),
            "map_baseline_probability": float(
                baseline_probabilities.get(map_name, elo_probability)
            ),
            "map_pick_by_team": float(picker == "team1"),
            "map_pick_by_opponent": float(picker == "team2"),
            "map_is_decider": float(picker == "decider"),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    x = map_feature_frame(frame, payload.get("map_names", []))
    x = x.reindex(columns=payload.get("features", x.columns), fill_value=0.0)
    baseline = frame["map_baseline_probability"].astype(float)
    model_kind = payload.get("model_kind", "residual")
    if model_kind == "baseline":
        raw = baseline.to_numpy(dtype=float)
    elif model_kind == "classifier":
        raw = payload["model"].predict_proba(x)[:, 1]
    else:
        raw = baseline.to_numpy(dtype=float) + payload["model"].predict(x)
    if model_kind != "baseline" and "model_blend_weight" in payload:
        calibrated = apply_probability_blend_temperature(
            np.clip(raw, 0.03, 0.97),
            baseline.to_numpy(dtype=float),
            blend_weight=payload.get("model_blend_weight", 1.0),
            temperature=payload.get("probability_temperature", 1.0),
            max_model_delta=payload.get("max_model_delta", 0.20),
        )
    else:
        calibrated = apply_symmetric_calibration(
            payload.get("calibrator"),
            np.clip(raw, 0.03, 0.97),
        )
    metrics = payload.get("metadata", {}).get("map_metrics", {})
    enabled = bool(
        metrics.get(
            "active_for_predictions",
            metrics.get("enabled", metrics.get("model_reliability", 0.0) > 0.0),
        )
    )
    probabilities = calibrated if enabled else baseline.to_numpy(dtype=float)
    return {
        map_name: float(probability)
        for map_name, probability in zip(map_order, probabilities)
    }, {
        "map_model_used": enabled,
        "map_model_reliability": float(metrics.get("model_reliability", 0.0)),
        "map_model_candidate": metrics.get("selected_candidate", ""),
        "map_model_recommended_candidate": metrics.get("recommended_candidate", ""),
        "map_model_selection_mode": metrics.get("selection_mode", "auto"),
    }
