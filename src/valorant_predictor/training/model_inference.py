import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.form_calculations import (
    calculate_team_form,
    clean_match_data,
    normalize_map_name,
    normalize_player_name,
    weighted_recent_mean,
)
from ..config import MAP_MODEL_PATH
from ..features.team_context import (
    build_sequential_team_context,
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
    apply_team_anchored_map_correction,
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
    selected_maps: list[str] | None = None,
    dataset_fingerprint_value: str = "",
    model_path: str = PLAYER_MODEL_PATH,
) -> pd.DataFrame:
    payload = load_model_payload(model_path)
    if payload is None or profiles.empty:
        return profiles

    cleaned = clean_match_data(matches)
    output = profiles.copy()
    selected_maps = {
        normalize_map_name(map_name)
        for map_name in (selected_maps or [])
        if normalize_map_name(map_name)
    }
    team_forms = {
        team: calculate_team_form(cleaned, team)
        for team in {team1, team2}
    }
    cleaned_player_names = cleaned["player"].map(normalize_player_name)
    team_maps = team_rows_from_cleaned(cleaned)
    prediction_date = (
        team_maps["match_date_sort"].max() + pd.Timedelta(days=1)
        if not team_maps.empty and team_maps["match_date_sort"].notna().any()
        else pd.Timestamp.now(tz="UTC")
    )

    lineup_by_team = {}
    for team in [team1, team2]:
        team_profiles = output[output["team"] == team].copy()
        sort_columns = [
            column
            for column in ["lineup_certainty", "data_reliability", "base_rating"]
            if column in team_profiles.columns
        ]
        if sort_columns:
            team_profiles = team_profiles.sort_values(
                sort_columns,
                ascending=[False] * len(sort_columns),
            )
        lineup_by_team[team] = {
            normalize_lineup_identity(row.get("player"), row.get("player_id"))
            for _, row in team_profiles.head(5).iterrows()
        }
    team_payload = load_model_payload(TEAM_MODEL_PATH)
    can_reuse_team_state = bool(
        team_payload
        and team_payload.get("sequential_state") is not None
        and dataset_fingerprint_value
        and team_payload.get("metadata", {}).get("dataset_fingerprint")
        == dataset_fingerprint_value
    )
    if can_reuse_team_state:
        sequential_state = team_payload["sequential_state"]
    else:
        _, sequential_state = build_sequential_team_context(cleaned, team_maps)
    team_contexts = {
        team1: team_context_from_state(
            sequential_state,
            team1,
            team2,
            lineup_a=lineup_by_team.get(team1),
            lineup_b=lineup_by_team.get(team2),
            as_of_date=prediction_date,
        ),
        team2: team_context_from_state(
            sequential_state,
            team2,
            team1,
            lineup_a=lineup_by_team.get(team2),
            lineup_b=lineup_by_team.get(team1),
            as_of_date=prediction_date,
        ),
    }
    feature_rows = []
    for _, row in output.iterrows():
        team_form = team_forms.get(row["team"])
        if team_form is None:
            team_form = calculate_team_form(cleaned, row["team"])
        opponent = team2 if row["team"] == team1 else team1
        opponent_form = team_forms.get(opponent)
        if opponent_form is None:
            opponent_form = calculate_team_form(cleaned, opponent)
        player_id = pd.to_numeric(pd.Series([row.get("player_id")]), errors="coerce").iloc[0]
        if pd.notna(player_id) and "player_id" in cleaned.columns:
            player_history = cleaned[
                pd.to_numeric(cleaned["player_id"], errors="coerce").eq(player_id)
            ]
        else:
            player_history = cleaned[
                cleaned_player_names == normalize_player_name(row["player"])
            ]
        player_vs_opponent = player_history[player_history["opponent"] == opponent]
        shrunk_rating = float(row.get("base_rating", 1.0) or 1.0)
        if selected_maps and "map_name" in player_history.columns:
            map_history = player_history[
                player_history["map_name"].map(normalize_map_name).isin(selected_maps)
            ].tail(20)
        else:
            map_history = player_history.iloc[:0]
        if map_history.empty:
            map_rating = shrunk_rating
            map_maps = 0
        else:
            map_raw_rating = weighted_recent_mean(map_history["rating_for_model"])
            map_maps = len(map_history)
            map_reliability = map_maps / (map_maps + 6.0)
            map_rating = shrunk_rating + map_reliability * (
                map_raw_rating - shrunk_rating
            )
        sequential = team_contexts.get(row["team"], {})

        feature_rows.append(
            {
                "player_last_3_rating": row.get("last_3_rating", row.get("base_rating", 1.0)),
                "player_last_5_rating": row.get("last_5_rating", row.get("base_rating", 1.0)),
                "player_last_10_rating": row.get("last_10_rating", row.get("base_rating", 1.0)),
                "player_60d_rating": row.get("recent_60d_rating", row.get("raw_form_rating", row.get("base_rating", 1.0))),
                "player_overall_rating": row.get("overall_rating", row.get("base_rating", 1.0)),
                "player_shrunk_rating": shrunk_rating,
                "player_map_rating": map_rating,
                "player_map_maps": map_maps,
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
                "team_elo_advantage": float(sequential.get("elo_diff", 0.0)),
                "region_elo_advantage": float(sequential.get("region_elo_diff", 0.0)),
                "strength_of_schedule_diff": float(
                    sequential.get("strength_of_schedule_diff", 0.0)
                ),
                "lineup_rating_diff": float(sequential.get("lineup_rating_diff", 0.0)),
                "map_pool_elo_diff": float(sequential.get("map_pool_elo_diff", 0.0)),
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
    baseline_feature = payload.get("baseline_feature", "player_last_10_rating")
    baseline_source = (
        output.get("base_rating", 1.0)
        if baseline_feature == "player_shrunk_rating"
        else output.get("last_10_rating", output.get("base_rating", 1.0))
    )
    baseline = pd.to_numeric(baseline_source, errors="coerce").fillna(1.0)
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
    baseline_probabilities: dict[str, float] | None = None,
    team_anchor_probability: float | None = None,
    model_path: str = MAP_MODEL_PATH,
) -> tuple[dict[str, float], dict]:
    baseline_probabilities = dict(
        baseline_probabilities
        if baseline_probabilities is not None
        else team_features.get("map_probabilities", {})
    )
    elo_probability = float(team_features.get("elo_probability", 0.5))
    team_anchor_probability = float(
        team_anchor_probability
        if team_anchor_probability is not None
        else team_features.get("team_anchor_probability", elo_probability)
    )
    map_reliabilities = dict(team_features.get("map_reliabilities", {}))
    map_signal_probabilities = dict(
        team_features.get("map_signal_probabilities", {})
    )
    map_team_games = dict(team_features.get("map_team_games", {}))
    map_opponent_games = dict(team_features.get("map_opponent_games", {}))
    payload = load_model_payload(model_path)
    if payload is None or not map_order:
        return {
            map_name: float(
                baseline_probabilities.get(map_name, team_anchor_probability)
            )
            for map_name in map_order
        }, {"map_model_used": False, "map_model_reliability": 0.0}

    picked_by = picked_by or [""] * len(map_order)
    rows = []
    for index, map_name in enumerate(map_order, start=1):
        picker = picked_by[index - 1] if index - 1 < len(picked_by) else ""
        baseline_probability = float(
            baseline_probabilities.get(map_name, team_anchor_probability)
        )
        row = {
            **team_features,
            "map_name": map_name,
            "map_number": float(index),
            "map_team_anchor_probability": team_anchor_probability,
            "map_baseline_probability": baseline_probability,
            "map_signal_probability": float(
                map_signal_probabilities.get(map_name, baseline_probability)
            ),
            "map_specific_probability_delta": (
                baseline_probability - team_anchor_probability
            ),
            "map_history_reliability": float(
                map_reliabilities.get(map_name, 0.0)
            ),
            "map_team_history_games": float(map_team_games.get(map_name, 0.0)),
            "map_opponent_history_games": float(
                map_opponent_games.get(map_name, 0.0)
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
        if payload.get("hierarchical_team_anchor", False):
            calibrated = apply_team_anchored_map_correction(
                np.clip(raw, 0.03, 0.97),
                baseline.to_numpy(dtype=float),
                frame["map_history_reliability"].to_numpy(dtype=float),
                blend_weight=payload.get("model_blend_weight", 1.0),
                max_model_delta=payload.get("max_model_delta", 0.10),
            )
        else:
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
        "map_model_team_anchored": bool(
            payload.get("hierarchical_team_anchor", False)
        ),
    }
