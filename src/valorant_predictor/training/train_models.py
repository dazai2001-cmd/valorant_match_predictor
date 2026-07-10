import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MATCHES_CSV, MODELS_DIR, PLAYER_MODEL_PATH, TEAM_MODEL_PATH, TRAINING_METRICS_PATH
from ..features.form_calculations import (
    clean_match_data,
    match_id_from_url,
    parse_match_datetime,
    weighted_recent_mean,
)
from ..team_registry import registry_team_pages
from ..vlr_client import scrape_matches


MODEL_DIR = MODELS_DIR
METRICS_PATH = TRAINING_METRICS_PATH

PLAYER_FEATURES = [
    "player_last_3_rating",
    "player_last_5_rating",
    "player_last_10_rating",
    "player_60d_rating",
    "player_overall_rating",
    "player_rating_trend",
    "player_recent_acs",
    "player_recent_kd",
    "player_recent_assists",
    "player_maps",
    "team_recent_rating",
    "team_recent_win_rate",
    "opponent_recent_rating",
    "opponent_recent_win_rate",
    "player_vs_opponent_rating",
    "player_vs_opponent_maps",
]

TEAM_FEATURES = [
    "team_rating_diff",
    "team_win_rate_diff",
    "team_score_margin_diff",
    "team_consistency_diff",
    "team_maps_diff",
    "h2h_win_rate",
    "h2h_rating_diff",
    "h2h_maps",
]


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def ensure_model_dir() -> None:
    Path(MODEL_DIR).mkdir(exist_ok=True)


def chronological_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["match_date_sort", "match_id"], na_position="first").reset_index(drop=True)


def restrict_history_window(df: pd.DataFrame, season_year: int | None, fallback_years: int) -> pd.DataFrame:
    if df.empty or season_year is None or df["match_date_sort"].isna().all():
        return df

    min_year = season_year - fallback_years
    years = df["match_date_sort"].dt.year
    return df[(years >= min_year) & (years <= season_year)].copy()


def target_rows_for_season(df: pd.DataFrame, season_year: int | None) -> pd.DataFrame:
    if df.empty or season_year is None or df["match_date_sort"].isna().all():
        return df
    return df[df["match_date_sort"].dt.year == season_year].copy()


def history_before(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    target_date = target.get("match_date_sort")
    target_match_id = target.get("match_id")

    if pd.notna(target_date):
        return df[df["match_date_sort"] < target_date].copy()

    if pd.notna(target_match_id):
        return df[df["match_id"] < target_match_id].copy()

    return df.iloc[:0].copy()


def recent_by_days(history: pd.DataFrame, target: pd.Series, recent_days: int) -> pd.DataFrame:
    target_date = target.get("match_date_sort")
    if pd.isna(target_date) or history["match_date_sort"].isna().all():
        return history.tail(10)
    cutoff = target_date - pd.Timedelta(days=recent_days)
    return history[history["match_date_sort"] >= cutoff].copy()


def rating_stats(history: pd.DataFrame, target: pd.Series, recent_days: int) -> dict:
    if history.empty:
        return {
            "last_3_rating": 1.0,
            "last_5_rating": 1.0,
            "last_10_rating": 1.0,
            "recent_days_rating": 1.0,
            "overall_rating": 1.0,
            "rating_trend": 0.0,
            "recent_acs": 200.0,
            "recent_kd": 1.0,
            "recent_assists": 5.0,
            "maps": 0,
        }

    ordered = chronological_sort(history)
    recent = recent_by_days(ordered, target, recent_days)
    if recent.empty:
        recent = ordered.tail(10)

    deaths = recent["deaths"].replace(0, 1)
    last_3 = weighted_recent_mean(ordered.tail(3)["rating_for_model"])
    last_10 = weighted_recent_mean(ordered.tail(10)["rating_for_model"])

    return {
        "last_3_rating": last_3,
        "last_5_rating": weighted_recent_mean(ordered.tail(5)["rating_for_model"]),
        "last_10_rating": last_10,
        "recent_days_rating": weighted_recent_mean(recent["rating_for_model"]),
        "overall_rating": ordered["rating_for_model"].mean(),
        "rating_trend": last_3 - last_10,
        "recent_acs": recent["acs"].mean(),
        "recent_kd": (recent["kills"] / deaths).mean(),
        "recent_assists": recent["assists"].mean(),
        "maps": len(ordered),
    }


def team_rows_from_cleaned(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (match_url, team), group in cleaned.groupby(["match_url", "team"], sort=False):
        team_score = group["team_score"].dropna().iloc[0] if group["team_score"].notna().any() else None
        opp_score = group["opp_score"].dropna().iloc[0] if group["opp_score"].notna().any() else None
        opponent = group["opponent"].dropna().iloc[0] if group["opponent"].notna().any() else ""
        rows.append(
            {
                "match_url": match_url,
                "match_id": group["match_id"].dropna().max(),
                "match_date_sort": group["match_date_sort"].dropna().max()
                if group["match_date_sort"].notna().any()
                else pd.NaT,
                "team": team,
                "opponent": opponent,
                "team_avg_rating": group["rating_for_model"].mean(),
                "team_win": group["is_winner"].astype(float).max(),
                "score_margin": float(team_score - opp_score)
                if team_score is not None and opp_score is not None
                else 0.0,
            }
        )
    return chronological_sort(pd.DataFrame(rows))


def team_history_features(team_history: pd.DataFrame) -> dict:
    if team_history.empty:
        return {
            "recent_rating": 1.0,
            "recent_win_rate": 0.5,
            "score_margin": 0.0,
            "consistency": 0.0,
            "maps": 0,
        }

    recent = chronological_sort(team_history).tail(10)
    consistency = float(recent["team_avg_rating"].std()) if len(recent) > 1 else 0.0
    return {
        "recent_rating": weighted_recent_mean(recent["team_avg_rating"]),
        "recent_win_rate": weighted_recent_mean(recent["team_win"]),
        "score_margin": weighted_recent_mean(recent["score_margin"]),
        "consistency": consistency,
        "maps": len(team_history),
    }


def h2h_features(team_maps: pd.DataFrame, team_a: str, team_b: str, target: pd.Series) -> dict:
    prior = history_before(team_maps, target)
    h2h = prior[(prior["team"] == team_a) & (prior["opponent"] == team_b)].copy()
    if h2h.empty:
        return {"h2h_win_rate": 0.5, "h2h_rating_diff": 0.0, "h2h_maps": 0}

    opponent_h2h = prior[(prior["team"] == team_b) & (prior["opponent"] == team_a)].copy()
    rating_diff = h2h["team_avg_rating"].mean() - opponent_h2h["team_avg_rating"].mean()
    reliability = len(h2h) / (len(h2h) + 6)
    return {
        "h2h_win_rate": 0.5 + reliability * (h2h["team_win"].mean() - 0.5),
        "h2h_rating_diff": reliability * rating_diff,
        "h2h_maps": len(h2h),
    }


def build_player_training_data(
    matches: pd.DataFrame,
    season_year: int | None,
    recent_days: int,
    fallback_years: int,
    min_history_maps: int,
) -> pd.DataFrame:
    cleaned = restrict_history_window(clean_match_data(matches), season_year, fallback_years)
    cleaned = chronological_sort(cleaned)
    if cleaned.empty:
        return pd.DataFrame()

    cleaned["player_norm"] = cleaned["player"].str.lower()
    targets = target_rows_for_season(cleaned, season_year)
    rows = []

    for _, target in targets.iterrows():
        prior = history_before(cleaned, target)
        player_history = prior[prior["player_norm"] == target["player_norm"]]
        if len(player_history) < min_history_maps:
            continue

        player_stats = rating_stats(player_history, target, recent_days)
        prior_team_maps = team_rows_from_cleaned(prior) if not prior.empty else pd.DataFrame()
        team_stats = team_history_features(
            prior_team_maps[prior_team_maps["team"] == target["team"]]
            if not prior_team_maps.empty
            else pd.DataFrame()
        )
        opponent_stats = team_history_features(
            prior_team_maps[prior_team_maps["team"] == target["opponent"]]
            if not prior_team_maps.empty
            else pd.DataFrame()
        )
        player_vs_opp = player_history[player_history["opponent"] == target["opponent"]]

        rows.append(
            {
                "target_rating": target["rating_for_model"],
                "player": target["player"],
                "team": target["team"],
                "opponent": target["opponent"],
                "match_url": target["match_url"],
                "player_last_3_rating": player_stats["last_3_rating"],
                "player_last_5_rating": player_stats["last_5_rating"],
                "player_last_10_rating": player_stats["last_10_rating"],
                "player_60d_rating": player_stats["recent_days_rating"],
                "player_overall_rating": player_stats["overall_rating"],
                "player_rating_trend": player_stats["rating_trend"],
                "player_recent_acs": player_stats["recent_acs"],
                "player_recent_kd": player_stats["recent_kd"],
                "player_recent_assists": player_stats["recent_assists"],
                "player_maps": player_stats["maps"],
                "team_recent_rating": team_stats["recent_rating"],
                "team_recent_win_rate": team_stats["recent_win_rate"],
                "opponent_recent_rating": opponent_stats["recent_rating"],
                "opponent_recent_win_rate": opponent_stats["recent_win_rate"],
                "player_vs_opponent_rating": player_vs_opp["rating_for_model"].mean()
                if not player_vs_opp.empty
                else player_stats["overall_rating"],
                "player_vs_opponent_maps": len(player_vs_opp),
            }
        )

    return pd.DataFrame(rows)


def team_feature_row(team_maps: pd.DataFrame, target: pd.Series, team_a: str, team_b: str) -> dict | None:
    prior = history_before(team_maps, target)
    a_stats = team_history_features(prior[prior["team"] == team_a])
    b_stats = team_history_features(prior[prior["team"] == team_b])
    if a_stats["maps"] == 0 or b_stats["maps"] == 0:
        return None

    h2h = h2h_features(team_maps, team_a, team_b, target)
    return {
        "team_rating_diff": a_stats["recent_rating"] - b_stats["recent_rating"],
        "team_win_rate_diff": a_stats["recent_win_rate"] - b_stats["recent_win_rate"],
        "team_score_margin_diff": a_stats["score_margin"] - b_stats["score_margin"],
        "team_consistency_diff": b_stats["consistency"] - a_stats["consistency"],
        "team_maps_diff": a_stats["maps"] - b_stats["maps"],
        **h2h,
    }


def build_team_training_data(
    matches: pd.DataFrame,
    season_year: int | None,
    fallback_years: int,
) -> pd.DataFrame:
    cleaned = restrict_history_window(clean_match_data(matches), season_year, fallback_years)
    team_maps = team_rows_from_cleaned(cleaned)
    targets = target_rows_for_season(team_maps, season_year)
    rows = []

    for match_url, group in targets.groupby("match_url", sort=False):
        if len(group) != 2:
            continue
        first, second = group.iloc[0], group.iloc[1]
        for team_a_row, team_b_row in [(first, second), (second, first)]:
            features = team_feature_row(team_maps, team_a_row, team_a_row["team"], team_b_row["team"])
            if features is None:
                continue
            rows.append(
                {
                    "target_win": int(team_a_row["team_win"]),
                    "match_url": match_url,
                    "team": team_a_row["team"],
                    "opponent": team_b_row["team"],
                    **features,
                }
            )

    return pd.DataFrame(rows)


def train_player_model(training_df: pd.DataFrame) -> tuple[Pipeline | None, dict]:
    if len(training_df) < 5:
        return None, {"status": "skipped", "reason": "Need at least 5 player training rows."}

    x = training_df[PLAYER_FEATURES].fillna(0.0)
    y = training_df["target_rating"]
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    if len(training_df) >= 25:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "validation_rows": len(x_test),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }
    else:
        model.fit(x, y)
        preds = model.predict(x)
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "validation_rows": 0,
            "train_mae": mean_absolute_error(y, preds),
            "train_r2": r2_score(y, preds) if len(training_df) > 1 else 0.0,
        }

    return model, metrics


def train_team_model(training_df: pd.DataFrame) -> tuple[Pipeline | None, dict]:
    if len(training_df) < 6:
        return None, {"status": "skipped", "reason": "Need at least 6 team training rows."}
    if training_df["target_win"].nunique() < 2:
        return None, {"status": "skipped", "reason": "Need both wins and losses in team training rows."}

    x = training_df[TEAM_FEATURES].fillna(0.0)
    y = training_df["target_win"]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )

    if len(training_df) >= 30 and y.value_counts().min() >= 4:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=7, stratify=y
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        probs = model.predict_proba(x_test)[:, 1]
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "validation_rows": len(x_test),
            "accuracy": accuracy_score(y_test, preds),
            "log_loss": log_loss(y_test, probs),
        }
    else:
        model.fit(x, y)
        preds = model.predict(x)
        probs = model.predict_proba(x)[:, 1]
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "validation_rows": 0,
            "train_accuracy": accuracy_score(y, preds),
            "train_log_loss": log_loss(y, probs),
        }

    return model, metrics


def save_model(path: str, payload: dict) -> None:
    ensure_model_dir()
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def train_models(
    matches_csv: str = MATCHES_CSV,
    season_year: int | None = None,
    recent_days: int = 60,
    fallback_years: int = 1,
    min_player_history_maps: int = 3,
) -> dict:
    matches = load_csv(matches_csv)
    if matches.empty:
        raise ValueError("No match data found. Run python scripts\\scrape.py --matches first.")

    player_training = build_player_training_data(
        matches,
        season_year=season_year,
        recent_days=recent_days,
        fallback_years=fallback_years,
        min_history_maps=min_player_history_maps,
    )
    team_training = build_team_training_data(
        matches,
        season_year=season_year,
        fallback_years=fallback_years,
    )

    player_model, player_metrics = train_player_model(player_training)
    team_model, team_metrics = train_team_model(team_training)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "matches_csv": str(matches_csv),
        "season_year": season_year,
        "recent_days": recent_days,
        "fallback_years": fallback_years,
        "min_player_history_maps": min_player_history_maps,
        "player_training_rows": len(player_training),
        "team_training_rows": len(team_training),
        "player_metrics": player_metrics,
        "team_metrics": team_metrics,
    }

    if player_model is not None:
        save_model(
            PLAYER_MODEL_PATH,
            {"model": player_model, "features": PLAYER_FEATURES, "metadata": metadata},
        )
    if team_model is not None:
        save_model(
            TEAM_MODEL_PATH,
            {"model": team_model, "features": TEAM_FEATURES, "metadata": metadata},
        )

    ensure_model_dir()
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Valorant player-rating and team-win models.")
    parser.add_argument("--matches-csv", default=MATCHES_CSV)
    parser.add_argument("--season-year", type=int, default=datetime.utcnow().year)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--fallback-years", type=int, default=1)
    parser.add_argument("--min-player-history-maps", type=int, default=3)
    parser.add_argument("--refresh-matches", action="store_true")
    parser.add_argument("--limit-per-team", type=int, default=75)
    args = parser.parse_args()

    if args.refresh_matches:
        scrape_matches(
            output_csv=args.matches_csv,
            limit_per_team=args.limit_per_team,
            team_pages=registry_team_pages(),
            season_year=args.season_year,
        )

    metrics = train_models(
        matches_csv=args.matches_csv,
        season_year=args.season_year,
        recent_days=args.recent_days,
        fallback_years=args.fallback_years,
        min_player_history_maps=args.min_player_history_maps,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
