import pickle
from pathlib import Path

import pandas as pd

from ..features.form_calculations import calculate_team_form, clean_match_data
from .train_models import (
    PLAYER_FEATURES,
    PLAYER_MODEL_PATH,
    TEAM_FEATURES,
    TEAM_MODEL_PATH,
    team_feature_row,
    team_rows_from_cleaned,
)


def load_model_payload(path: str) -> dict | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        return pickle.load(handle)


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
    feature_rows = []
    for _, row in output.iterrows():
        team_form = calculate_team_form(cleaned, row["team"])
        opponent = team2 if row["team"] == team1 else team1
        opponent_form = calculate_team_form(cleaned, opponent)
        player_vs_opponent = cleaned[
            (cleaned["player"].str.lower() == str(row["player"]).lower())
            & (cleaned["opponent"] == opponent)
        ]

        feature_rows.append(
            {
                "player_last_3_rating": row.get("last_3_rating", row.get("base_rating", 1.0)),
                "player_last_5_rating": row.get("last_5_rating", row.get("base_rating", 1.0)),
                "player_last_10_rating": row.get("last_10_rating", row.get("base_rating", 1.0)),
                "player_60d_rating": row.get("raw_form_rating", row.get("base_rating", 1.0)),
                "player_overall_rating": row.get("overall_rating", row.get("base_rating", 1.0)),
                "player_rating_trend": row.get("rating_trend", 0.0),
                "player_recent_acs": row.get("avg_acs", 200.0),
                "player_recent_kd": row.get("kd_ratio", 1.0),
                "player_recent_assists": row.get("avg_assists", 5.0),
                "player_maps": row.get("recent_maps", 0),
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

    x = pd.DataFrame(feature_rows)[payload["features"]].fillna(0.0)
    output["trained_base_rating"] = payload["model"].predict(x).clip(0.45, 1.70)
    output["base_rating"] = output["trained_base_rating"]
    return output


def predict_team_win_probability(
    matches: pd.DataFrame,
    team1: str,
    team2: str,
    model_path: str = TEAM_MODEL_PATH,
) -> tuple[float | None, dict | None]:
    payload = load_model_payload(model_path)
    if payload is None:
        return None, None

    cleaned = clean_match_data(matches)
    team_maps = team_rows_from_cleaned(cleaned)
    if team_maps.empty:
        return None, None

    target = pd.Series(
        {
            "match_date_sort": team_maps["match_date_sort"].max() + pd.Timedelta(days=1)
            if team_maps["match_date_sort"].notna().any()
            else pd.NaT,
            "match_id": team_maps["match_id"].max() + 1
            if team_maps["match_id"].notna().any()
            else None,
        }
    )
    features = team_feature_row(team_maps, target, team1, team2)
    if features is None:
        return None, None

    x = pd.DataFrame([features])[payload["features"]].fillna(0.0)
    probability = float(payload["model"].predict_proba(x)[0, 1])
    return probability, features
