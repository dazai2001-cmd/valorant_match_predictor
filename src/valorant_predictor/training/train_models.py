import argparse
import hashlib
import json
import math
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Loky clamps this to one worker and skips a broken WMIC physical-core probe.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "0")
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    DATA_QUALITY_PATH,
    MATCH_COVERAGE_CSV,
    MATCHES_CSV,
    MAP_MODEL_PATH,
    MODELS_DIR,
    NEWS_CSV,
    PLAYER_MODEL_PATH,
    ROSTERS_CSV,
    TEAM_MODEL_PATH,
    TRAINING_METRICS_PATH,
)
from ..data_quality import (
    data_quality_report,
    enrich_match_metadata,
    filter_training_ready_matches,
)
from ..features.form_calculations import (
    clean_match_data,
    filter_curated_competition_history,
    match_id_from_url,
    normalize_map_name,
    normalize_player_name,
    parse_match_datetime,
    weighted_recent_mean,
)
from ..features.team_context import (
    MAX_MAP_SPECIFIC_WEIGHT,
    SEQUENTIAL_TEAM_FEATURES,
    build_sequential_team_context,
    freeze_team_state,
)
from ..model_selection import (
    load_model_selection,
    normalize_model_selection,
    save_model_selection,
    selected_candidate as resolve_selected_candidate,
)
from ..team_registry import (
    filter_registry_tier1_matchups,
    load_team_registry,
    registry_team_pages,
)
from ..vlr_client import canonicalize_match_dataframe, match_coverage_report, scrape_matches


MODEL_DIR = MODELS_DIR
METRICS_PATH = TRAINING_METRICS_PATH
MODEL_VERSION = "logic-v14-team-anchored-map-hierarchy"
TRAINING_HALF_LIFE_DAYS = 365.0

PLAYER_FEATURES = [
    "player_last_3_rating",
    "player_last_5_rating",
    "player_last_10_rating",
    "player_60d_rating",
    "player_overall_rating",
    "player_shrunk_rating",
    "player_map_rating",
    "player_map_maps",
    "player_rating_trend",
    "player_recent_acs",
    "player_recent_kd",
    "player_recent_assists",
    "player_maps",
    "player_60d_maps",
    "player_effective_maps",
    "player_days_since_last_match",
    "player_freshness",
    "player_rating_std",
    "player_agent_pool_size",
    "team_recent_rating",
    "team_recent_win_rate",
    "opponent_recent_rating",
    "opponent_recent_win_rate",
    "player_vs_opponent_rating",
    "player_vs_opponent_maps",
    "team_elo_advantage",
    "region_elo_advantage",
    "strength_of_schedule_diff",
    "lineup_rating_diff",
    "map_pool_elo_diff",
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
    *SEQUENTIAL_TEAM_FEATURES,
]

MAP_CONTEXT_FEATURES = [
    "map_number",
    "map_team_anchor_probability",
    "map_baseline_probability",
    "map_signal_probability",
    "map_specific_probability_delta",
    "map_history_reliability",
    "map_team_history_games",
    "map_opponent_history_games",
    "map_pick_by_team",
    "map_pick_by_opponent",
    "map_is_decider",
]


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def ensure_model_dir() -> None:
    Path(MODEL_DIR).mkdir(exist_ok=True)


def chronological_sort(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["match_date_sort", "match_id"]
    if "map_number" in df.columns:
        columns.append("map_number")
    return df.sort_values(columns, na_position="first").reset_index(drop=True)


def restrict_history_window(
    df: pd.DataFrame,
    season_year: int | None,
    fallback_years: int | None = None,
) -> pd.DataFrame:
    if df.empty or season_year is None or df["match_date_sort"].isna().all():
        return df
    years = df["match_date_sort"].dt.year
    return df[years <= season_year].copy()


def target_rows_for_season(df: pd.DataFrame, season_year: int | None) -> pd.DataFrame:
    if df.empty or season_year is None or df["match_date_sort"].isna().all():
        return df
    return df[df["match_date_sort"].dt.year <= season_year].copy()


def add_training_weights(
    frame: pd.DataFrame,
    half_life_days: float = TRAINING_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    dates = pd.to_datetime(output["target_date"], utc=True, errors="coerce")
    reference = dates.max()
    if pd.isna(reference):
        output["training_weight"] = 1.0
        return output
    ages = (reference - dates).dt.total_seconds().div(86400.0).clip(lower=0.0)
    time_weight = (0.5 ** (ages / half_life_days)).fillna(0.25).clip(lower=0.05, upper=1.0)
    importance = pd.to_numeric(
        output.get("match_importance", 1.0),
        errors="coerce",
    )
    if not isinstance(importance, pd.Series):
        importance = pd.Series(importance, index=output.index, dtype=float)
    competition = pd.to_numeric(
        output.get("competition_strength_weight", 1.0),
        errors="coerce",
    )
    if not isinstance(competition, pd.Series):
        competition = pd.Series(competition, index=output.index, dtype=float)
    output["training_weight"] = (
        time_weight * importance.fillna(1.0) * competition.fillna(1.0)
    ).clip(0.03, 1.15)
    return output


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
            "shrunk_rating": 1.0,
            "map_rating": 1.0,
            "map_maps": 0,
            "rating_trend": 0.0,
            "recent_acs": 200.0,
            "recent_kd": 1.0,
            "recent_assists": 5.0,
            "maps": 0,
            "recent_days_maps": 0,
            "effective_maps": 0.0,
            "days_since_last_match": 365.0,
            "freshness": 0.0,
            "rating_std": 0.18,
            "agent_pool_size": 0,
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
        "recent_days_maps": len(recent),
        "effective_maps": float(len(recent)),
        "days_since_last_match": 0.0,
        "freshness": 1.0,
        "rating_std": float(ordered.tail(20)["rating_for_model"].std(ddof=0)) if len(ordered) > 1 else 0.18,
        "agent_pool_size": int(
            len(
                {
                    agent.strip()
                    for value in ordered.tail(20).get("agents", pd.Series(dtype=str)).dropna().astype(str)
                    for agent in value.split(";")
                    if agent.strip()
                }
            )
        ),
    }


def _weighted_array(values, decay: float = 0.85) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return 0.0
    weights = decay ** np.arange(clean.size - 1, -1, -1)
    return float(np.average(clean, weights=weights))


def rating_stats_from_ordered(history: pd.DataFrame, target: pd.Series, recent_days: int) -> dict:
    if history.empty:
        return rating_stats(history, target, recent_days)

    target_date = target.get("match_date_sort")
    if pd.notna(target_date) and history["match_date_sort"].notna().any():
        cutoff = target_date - pd.Timedelta(days=recent_days)
        recent = history[history["match_date_sort"] >= cutoff]
    else:
        recent = history.tail(10)
    if recent.empty:
        recent = history.tail(10)

    ratings = history["rating_for_model"].to_numpy(dtype=float)
    recent_ratings = recent["rating_for_model"].to_numpy(dtype=float)
    recent_deaths = recent["deaths"].replace(0, 1).to_numpy(dtype=float)
    last_3 = _weighted_array(ratings[-3:])
    last_10 = _weighted_array(ratings[-10:])
    latest_date = history["match_date_sort"].dropna().max() if history["match_date_sort"].notna().any() else pd.NaT
    days_since_last_match = (
        max(0.0, float((target_date - latest_date).total_seconds() / 86400))
        if pd.notna(target_date) and pd.notna(latest_date)
        else 0.0
    )
    if pd.notna(target_date) and recent["match_date_sort"].notna().any():
        ages = (target_date - recent["match_date_sort"]).dt.total_seconds().div(86400).clip(lower=0)
        evidence_weights = 0.5 ** (ages / 30.0)
        effective_maps = float(evidence_weights.sum() ** 2 / evidence_weights.pow(2).sum())
    else:
        effective_maps = float(len(recent))
    volatility = float(np.nanstd(ratings[-20:])) if len(ratings) > 1 else 0.18
    volatility_weight = min(1.0, len(ratings[-20:]) / 8.0)
    rating_std = math.sqrt(volatility_weight * volatility**2 + (1.0 - volatility_weight) * 0.18**2)
    agents = {
        agent.strip()
        for value in history.tail(20).get("agents", pd.Series(dtype=str)).dropna().astype(str)
        for agent in value.split(";")
        if agent.strip()
    }
    freshness = math.exp(-days_since_last_match / 45.0)
    blended_rating = (
        0.45 * _weighted_array(ratings[-5:])
        + 0.35 * _weighted_array(recent_ratings)
        + 0.20 * float(np.nanmean(ratings))
    )
    shrink_reliability = (
        effective_maps / (effective_maps + 8.0) * freshness
        if effective_maps > 0
        else 0.0
    )
    shrunk_rating = 1.0 + shrink_reliability * (blended_rating - 1.0)

    target_map = normalize_map_name(target.get("map_name", ""))
    if target_map and "map_name" in history.columns:
        normalized_history_maps = history["map_name"].map(normalize_map_name)
        map_history = history[normalized_history_maps == target_map].tail(20)
    else:
        map_history = history.iloc[:0]
    map_maps = len(map_history)
    if map_maps:
        map_raw_rating = _weighted_array(map_history["rating_for_model"].to_numpy(dtype=float))
        map_reliability = map_maps / (map_maps + 6.0)
        map_rating = shrunk_rating + map_reliability * (map_raw_rating - shrunk_rating)
    else:
        map_rating = shrunk_rating

    return {
        "last_3_rating": last_3,
        "last_5_rating": _weighted_array(ratings[-5:]),
        "last_10_rating": last_10,
        "recent_days_rating": _weighted_array(recent_ratings),
        "overall_rating": float(np.nanmean(ratings)),
        "shrunk_rating": shrunk_rating,
        "map_rating": map_rating,
        "map_maps": map_maps,
        "rating_trend": last_3 - last_10,
        "recent_acs": float(np.nanmean(recent["acs"].to_numpy(dtype=float))),
        "recent_kd": float(np.nanmean(recent["kills"].to_numpy(dtype=float) / recent_deaths)),
        "recent_assists": float(np.nanmean(recent["assists"].to_numpy(dtype=float))),
        "maps": len(history),
        "recent_days_maps": len(recent),
        "effective_maps": effective_maps,
        "days_since_last_match": days_since_last_match,
        "freshness": freshness,
        "rating_std": rating_std,
        "agent_pool_size": len(agents),
    }


def team_rows_from_cleaned(cleaned: pd.DataFrame) -> pd.DataFrame:
    if cleaned.empty:
        return pd.DataFrame()
    frame = cleaned.copy()
    defaults = {
        "match_url": "",
        "match_id": pd.NA,
        "match_date_sort": pd.NaT,
        "opponent": "",
        "event_name": "",
        "event_series": "",
        "event_stage": "",
        "event_tier": "unknown",
        "event_region": "",
        "is_lan": False,
        "patch": "",
        "team_region": "",
        "opponent_region": "",
        "match_importance": 1.0,
        "competition_tier": "unknown",
        "competition_strength_weight": 1.0,
        "team_score": pd.NA,
        "opp_score": pd.NA,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["_team_win"] = frame["is_winner"].astype(float)
    grouped = frame.groupby(["match_key", "team"], sort=False, dropna=False)
    output = grouped.agg(
        match_url=("match_url", "first"),
        match_id=("match_id", "max"),
        match_date_sort=("match_date_sort", "max"),
        opponent=("opponent", "first"),
        event_name=("event_name", "first"),
        event_series=("event_series", "first"),
        event_stage=("event_stage", "first"),
        event_tier=("event_tier", "first"),
        event_region=("event_region", "first"),
        is_lan=("is_lan", "first"),
        patch=("patch", "first"),
        team_region=("team_region", "first"),
        opponent_region=("opponent_region", "first"),
        match_importance=("match_importance", "first"),
        competition_tier=("competition_tier", "first"),
        competition_strength_weight=("competition_strength_weight", "first"),
        team_avg_rating=("rating_for_model", "mean"),
        team_win=("_team_win", "max"),
        team_score=("team_score", "first"),
        opp_score=("opp_score", "first"),
    ).reset_index()
    output["match_importance"] = pd.to_numeric(
        output["match_importance"],
        errors="coerce",
    ).fillna(1.0)
    output["competition_strength_weight"] = pd.to_numeric(
        output["competition_strength_weight"],
        errors="coerce",
    ).fillna(1.0)
    team_scores = pd.to_numeric(output.pop("team_score"), errors="coerce")
    opponent_scores = pd.to_numeric(output.pop("opp_score"), errors="coerce")
    output["score_margin"] = (team_scores - opponent_scores).fillna(0.0)
    return chronological_sort(output)


def team_history_features(
    team_history: pd.DataFrame,
    reference_date=None,
    half_life_days: float = 90.0,
) -> dict:
    if team_history.empty:
        return {
            "recent_rating": 1.0,
            "recent_win_rate": 0.5,
            "score_margin": 0.0,
            "consistency": 0.0,
            "maps": 0,
        }

    recent = chronological_sort(team_history).tail(20)
    reference = pd.to_datetime(reference_date, utc=True, errors="coerce")
    if pd.isna(reference):
        reference = recent["match_date_sort"].max()
    ages = (reference - recent["match_date_sort"]).dt.total_seconds().div(86400.0).clip(lower=0.0)
    weights = (0.5 ** (ages / half_life_days)).fillna(0.25).to_numpy(dtype=float)
    def weighted(column: str) -> float:
        values = recent[column].to_numpy(dtype=float)
        return float(np.average(values, weights=weights))

    consistency = float(recent["team_avg_rating"].std()) if len(recent) > 1 else 0.0
    return {
        "recent_rating": weighted("team_avg_rating"),
        "recent_win_rate": weighted("team_win"),
        "score_margin": weighted("score_margin"),
        "consistency": consistency,
        "maps": float(weights.sum()),
    }


def h2h_features(team_maps: pd.DataFrame, team_a: str, team_b: str, target: pd.Series) -> dict:
    prior = history_before(team_maps, target)
    h2h = prior[(prior["team"] == team_a) & (prior["opponent"] == team_b)].copy()
    target_date = pd.to_datetime(target.get("match_date_sort"), utc=True, errors="coerce")
    if pd.notna(target_date) and not h2h.empty:
        h2h = h2h[
            h2h["match_date_sort"] >= target_date - pd.Timedelta(days=180)
        ].copy()
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

    player_ids = pd.to_numeric(cleaned.get("player_id"), errors="coerce")
    cleaned["player_norm"] = cleaned["player"].map(normalize_player_name)
    cleaned.loc[player_ids.notna(), "player_norm"] = (
        "id:" + player_ids[player_ids.notna()].astype("int64").astype(str)
    )
    targets = target_rows_for_season(cleaned, season_year)
    if targets.empty:
        return pd.DataFrame()

    team_maps = team_rows_from_cleaned(cleaned)
    sequential_context, _ = build_sequential_team_context(cleaned, team_maps)
    sequential_lookup = {
        (row["match_key"], row["team"], row["opponent"]): row
        for row in sequential_context.to_dict("records")
    }
    context_cache = {}
    contexts = targets[
        ["match_key", "match_date_sort", "match_id", "team", "opponent"]
    ].drop_duplicates(subset=["match_key", "team", "opponent"])
    for _, target in contexts.iterrows():
        prior_team_matches = history_before(team_maps, target)
        context_cache[(target["match_key"], target["team"], target["opponent"])] = (
            team_history_features(
                prior_team_matches[prior_team_matches["team"] == target["team"]],
                reference_date=target["match_date_sort"],
            ),
            team_history_features(
                prior_team_matches[prior_team_matches["team"] == target["opponent"]],
                reference_date=target["match_date_sort"],
            ),
        )

    rows = []

    for player_norm, player_targets in targets.groupby("player_norm", sort=False):
        player_rows = cleaned[cleaned["player_norm"] == player_norm]
        player_feature_cache = {}
        for _, target in player_targets.iterrows():
            cache_key = (
                target["match_key"],
                target["opponent"],
                normalize_map_name(target.get("map_name", "")),
            )
            if cache_key not in player_feature_cache:
                player_history = history_before(player_rows, target)
                if len(player_history) < 1:
                    player_feature_cache[cache_key] = None
                else:
                    player_stats = rating_stats_from_ordered(player_history, target, recent_days)
                    player_vs_opp = player_history[player_history["opponent"] == target["opponent"]]
                    player_feature_cache[cache_key] = (player_stats, player_vs_opp)

            cached_player_features = player_feature_cache[cache_key]
            if cached_player_features is None:
                continue

            player_stats, player_vs_opp = cached_player_features
            team_stats, opponent_stats = context_cache[
                (target["match_key"], target["team"], target["opponent"])
            ]
            sequential = sequential_lookup.get(
                (target["match_key"], target["team"], target["opponent"]),
                {},
            )

            rows.append(
                {
                    "target_rating": target["rating_for_model"],
                    "player": target["player"],
                    "team": target["team"],
                    "opponent": target["opponent"],
                    "match_url": target["match_url"],
                    "match_key": target["match_key"],
                    "target_date": target["match_date_sort"],
                    "target_match_id": target["match_id"],
                    "match_importance": float(target.get("match_importance", 1.0) or 1.0),
                    "competition_strength_weight": float(
                        target.get("competition_strength_weight", 1.0) or 1.0
                    ),
                    "player_last_3_rating": player_stats["last_3_rating"],
                    "player_last_5_rating": player_stats["last_5_rating"],
                    "player_last_10_rating": player_stats["last_10_rating"],
                    "player_60d_rating": player_stats["recent_days_rating"],
                    "player_overall_rating": player_stats["overall_rating"],
                    "player_shrunk_rating": player_stats["shrunk_rating"],
                    "player_map_rating": player_stats["map_rating"],
                    "player_map_maps": player_stats["map_maps"],
                    "player_rating_trend": player_stats["rating_trend"],
                    "player_recent_acs": player_stats["recent_acs"],
                    "player_recent_kd": player_stats["recent_kd"],
                    "player_recent_assists": player_stats["recent_assists"],
                    "player_maps": player_stats["maps"],
                    "player_60d_maps": player_stats["recent_days_maps"],
                    "player_effective_maps": player_stats["effective_maps"],
                    "player_days_since_last_match": player_stats["days_since_last_match"],
                    "player_freshness": player_stats["freshness"],
                    "player_rating_std": player_stats["rating_std"],
                    "player_agent_pool_size": player_stats["agent_pool_size"],
                    "team_recent_rating": team_stats["recent_rating"],
                    "team_recent_win_rate": team_stats["recent_win_rate"],
                    "opponent_recent_rating": opponent_stats["recent_rating"],
                    "opponent_recent_win_rate": opponent_stats["recent_win_rate"],
                    "player_vs_opponent_rating": player_vs_opp["rating_for_model"].mean()
                    if not player_vs_opp.empty
                    else player_stats["overall_rating"],
                    "player_vs_opponent_maps": len(player_vs_opp),
                    "team_elo_advantage": float(sequential.get("elo_diff", 0.0)),
                    "region_elo_advantage": float(sequential.get("region_elo_diff", 0.0)),
                    "strength_of_schedule_diff": float(
                        sequential.get("strength_of_schedule_diff", 0.0)
                    ),
                    "lineup_rating_diff": float(sequential.get("lineup_rating_diff", 0.0)),
                    "map_pool_elo_diff": float(sequential.get("map_pool_elo_diff", 0.0)),
                }
            )

    return add_training_weights(pd.DataFrame(rows))


def team_feature_row(team_maps: pd.DataFrame, target: pd.Series, team_a: str, team_b: str) -> dict | None:
    prior = history_before(team_maps, target)
    a_stats = team_history_features(
        prior[prior["team"] == team_a],
        reference_date=target.get("match_date_sort"),
    )
    b_stats = team_history_features(
        prior[prior["team"] == team_b],
        reference_date=target.get("match_date_sort"),
    )
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

    for match_key, group in targets.groupby("match_key", sort=False):
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
                    "match_key": match_key,
                    "match_url": team_a_row["match_url"],
                    "target_date": team_a_row["match_date_sort"],
                    "target_match_id": team_a_row["match_id"],
                    "team": team_a_row["team"],
                    "opponent": team_b_row["team"],
                    "match_importance": float(team_a_row.get("match_importance", 1.0) or 1.0),
                    "competition_strength_weight": float(
                        team_a_row.get("competition_strength_weight", 1.0) or 1.0
                    ),
                    **features,
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        return output

    sequential_context, sequential_state = build_sequential_team_context(cleaned, team_maps)
    if not sequential_context.empty:
        output = output.merge(
            sequential_context,
            on=["match_key", "team", "opponent"],
            how="left",
        )
    for feature in SEQUENTIAL_TEAM_FEATURES:
        if feature not in output.columns:
            output[feature] = 0.0
    if "elo_probability" not in output.columns:
        output["elo_probability"] = 0.5
    output.attrs["sequential_state"] = freeze_team_state(sequential_state)
    weighted = add_training_weights(output)
    weighted.attrs["sequential_state"] = output.attrs["sequential_state"]
    return weighted


def map_rows_from_cleaned(cleaned: pd.DataFrame) -> pd.DataFrame:
    if cleaned.empty or "map_id" not in cleaned.columns:
        return pd.DataFrame()
    scoped = cleaned[cleaned["map_id"].notna()].copy()
    rows = []
    for (match_key, map_id, team), group in scoped.groupby(
        ["match_key", "map_id", "team"],
        sort=False,
    ):
        team_scores = (
            pd.to_numeric(group["map_team_score"], errors="coerce").dropna()
            if "map_team_score" in group.columns
            else pd.Series(dtype=float)
        )
        opponent_scores = (
            pd.to_numeric(group["map_opp_score"], errors="coerce").dropna()
            if "map_opp_score" in group.columns
            else pd.Series(dtype=float)
        )
        if team_scores.empty or opponent_scores.empty:
            continue
        team_score = float(team_scores.iloc[0])
        opponent_score = float(opponent_scores.iloc[0])
        if team_score == opponent_score:
            continue

        def first_value(column: str, default=""):
            if column not in group.columns or not group[column].notna().any():
                return default
            return group[column].dropna().iloc[0]

        rows.append(
            {
                "match_key": match_key,
                "map_id": map_id,
                "map_number": float(first_value("map_number", 0) or 0),
                "map_name": normalize_map_name(first_value("map_name", "")),
                "team": team,
                "opponent": first_value("opponent", ""),
                "target_map_win": float(team_score > opponent_score),
                "map_pick_team": first_value("map_pick_team", ""),
                "map_pick_type": str(first_value("map_pick_type", "unknown") or "unknown").lower(),
            }
        )
    return pd.DataFrame(rows)


def build_map_training_data(
    matches: pd.DataFrame,
    team_training: pd.DataFrame,
    season_year: int | None,
) -> pd.DataFrame:
    cleaned = restrict_history_window(clean_match_data(matches), season_year)
    map_targets = map_rows_from_cleaned(cleaned)
    if map_targets.empty or team_training.empty:
        return pd.DataFrame()

    context_columns = [
        "match_key",
        "team",
        "opponent",
        "target_date",
        "target_match_id",
        "match_importance",
        "training_weight",
        "elo_probability",
        "map_probabilities",
        "map_signal_probabilities",
        "map_reliabilities",
        "map_team_games",
        "map_opponent_games",
        *TEAM_FEATURES,
    ]
    context_columns = [column for column in context_columns if column in team_training.columns]
    context = team_training[context_columns].drop_duplicates(
        subset=["match_key", "team", "opponent"],
        keep="last",
    )
    output = map_targets.merge(
        context,
        on=["match_key", "team", "opponent"],
        how="inner",
    )
    if output.empty:
        return output

    def baseline_probability(row) -> float:
        probabilities = row.get("map_probabilities")
        if isinstance(probabilities, dict):
            value = probabilities.get(row.get("map_name"))
            if value is not None:
                return float(value)
        return float(row.get("elo_probability", 0.5) or 0.5)

    def map_context_value(row, column: str, default: float) -> float:
        values = row.get(column)
        if isinstance(values, dict):
            value = values.get(row.get("map_name"))
            if value is not None:
                return float(value)
        return float(default)

    output["map_team_anchor_probability"] = pd.to_numeric(
        output.get("elo_probability", pd.Series(0.5, index=output.index)),
        errors="coerce",
    ).fillna(0.5).clip(0.05, 0.95)
    output["map_baseline_probability"] = output.apply(baseline_probability, axis=1).clip(0.05, 0.95)
    output["map_signal_probability"] = output.apply(
        lambda row: map_context_value(
            row,
            "map_signal_probabilities",
            row["map_team_anchor_probability"],
        ),
        axis=1,
    ).clip(0.05, 0.95)
    output["map_history_reliability"] = output.apply(
        lambda row: map_context_value(row, "map_reliabilities", 0.0),
        axis=1,
    ).clip(0.0, 1.0)
    output["map_team_history_games"] = output.apply(
        lambda row: map_context_value(row, "map_team_games", 0.0),
        axis=1,
    ).clip(lower=0.0)
    output["map_opponent_history_games"] = output.apply(
        lambda row: map_context_value(row, "map_opponent_games", 0.0),
        axis=1,
    ).clip(lower=0.0)
    output["map_specific_probability_delta"] = (
        output["map_baseline_probability"]
        - output["map_team_anchor_probability"]
    )
    pick_team = output["map_pick_team"].fillna("").astype(str)
    output["map_pick_by_team"] = (pick_team == output["team"].astype(str)).astype(float)
    output["map_pick_by_opponent"] = (pick_team == output["opponent"].astype(str)).astype(float)
    output["map_is_decider"] = output["map_pick_type"].eq("decider").astype(float)
    return output


def map_feature_frame(frame: pd.DataFrame, map_names: list[str]) -> pd.DataFrame:
    output = frame.reindex(columns=[*TEAM_FEATURES, *MAP_CONTEXT_FEATURES], fill_value=0.0).copy()
    normalized_names = frame.get("map_name", pd.Series("", index=frame.index)).fillna("").astype(str)
    for map_name in map_names:
        output[f"map_name::{map_name}"] = (normalized_names == map_name).astype(float)
    return output.fillna(0.0).astype(float)


def grouped_chronological_split(
    training_df: pd.DataFrame,
    validation_fraction: float = 0.25,
) -> tuple[pd.Series, pd.Series]:
    groups = training_df[
        ["match_key", "target_date", "target_match_id"]
    ].drop_duplicates(subset=["match_key"])
    groups = groups.copy()
    groups["target_date"] = pd.to_datetime(groups["target_date"], utc=True, errors="coerce")
    groups = groups.sort_values(["target_date", "target_match_id"], na_position="first")
    validation_groups = max(1, int(math.ceil(len(groups) * validation_fraction)))
    split_at = max(1, len(groups) - validation_groups)
    train_groups = set(groups.iloc[:split_at]["match_key"])
    validation_groups_set = set(groups.iloc[split_at:]["match_key"])
    return training_df["match_key"].isin(train_groups), training_df["match_key"].isin(validation_groups_set)


def grouped_chronological_three_way_split(
    training_df: pd.DataFrame,
    calibration_fraction: float = 0.15,
    test_fraction: float = 0.20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    groups = training_df[
        ["match_key", "target_date", "target_match_id"]
    ].drop_duplicates(subset=["match_key"])
    groups = groups.copy()
    groups["target_date"] = pd.to_datetime(groups["target_date"], utc=True, errors="coerce")
    groups = groups.sort_values(["target_date", "target_match_id"], na_position="first")
    total_groups = len(groups)
    test_groups = max(1, int(math.ceil(total_groups * test_fraction)))
    calibration_groups = max(1, int(math.ceil(total_groups * calibration_fraction)))
    train_end = max(1, total_groups - test_groups - calibration_groups)
    calibration_end = max(train_end + 1, total_groups - test_groups)

    train_keys = set(groups.iloc[:train_end]["match_key"])
    calibration_keys = set(groups.iloc[train_end:calibration_end]["match_key"])
    test_keys = set(groups.iloc[calibration_end:]["match_key"])
    return (
        training_df["match_key"].isin(train_keys),
        training_df["match_key"].isin(calibration_keys),
        training_df["match_key"].isin(test_keys),
    )


def rolling_origin_splits(
    training_df: pd.DataFrame,
    folds: int = 3,
    calibration_fraction: float = 0.10,
    test_fraction: float = 0.15,
) -> list[tuple[pd.Series, pd.Series, pd.Series]]:
    groups = training_df[
        ["match_key", "target_date", "target_match_id"]
    ].drop_duplicates(subset=["match_key"])
    groups = groups.copy()
    groups["target_date"] = pd.to_datetime(groups["target_date"], utc=True, errors="coerce")
    groups = groups.sort_values(["target_date", "target_match_id"], na_position="first")
    total = len(groups)
    test_size = max(1, int(math.floor(total * test_fraction)))
    calibration_size = max(1, int(math.floor(total * calibration_fraction)))
    first_test_start = total - folds * test_size
    splits = []
    for fold in range(folds):
        test_start = first_test_start + fold * test_size
        test_end = min(total, test_start + test_size)
        calibration_start = test_start - calibration_size
        if calibration_start < 1 or test_start >= test_end:
            continue
        train_keys = set(groups.iloc[:calibration_start]["match_key"])
        calibration_keys = set(groups.iloc[calibration_start:test_start]["match_key"])
        test_keys = set(groups.iloc[test_start:test_end]["match_key"])
        splits.append(
            (
                training_df["match_key"].isin(train_keys),
                training_df["match_key"].isin(calibration_keys),
                training_df["match_key"].isin(test_keys),
            )
        )
    return splits


def _new_player_model(
    loss: str = "squared_error",
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    kwargs = {
        "loss": loss,
        "learning_rate": 0.05,
        "max_iter": 180,
        "max_leaf_nodes": 24,
        "min_samples_leaf": 30,
        "l2_regularization": 1.5,
        "random_state": 7,
    }
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


PLAYER_CANDIDATES = ("hgb_squared", "hgb_absolute", "extra_trees")


def _new_player_candidate(name: str):
    if name == "hgb_absolute":
        return _new_player_model(loss="absolute_error")
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=18,
            max_features=0.85,
            n_jobs=1,
            random_state=17,
        )
    return _new_player_model()


def train_player_model(
    training_df: pd.DataFrame,
    requested_candidate: str = "auto",
) -> tuple[dict | None, dict]:
    if len(training_df) < 5:
        return None, {"status": "skipped", "reason": "Need at least 5 player training rows."}

    x = training_df[PLAYER_FEATURES].fillna(0.0)
    y = training_df["target_rating"].astype(float)
    baseline = training_df["player_shrunk_rating"].astype(float).clip(0.35, 1.90)
    residual_target = y - baseline
    weights = training_df.get("training_weight", pd.Series(1.0, index=training_df.index)).astype(float)
    lower_model = _new_player_model(loss="quantile", quantile=0.10)
    upper_model = _new_player_model(loss="quantile", quantile=0.90)
    interval_adjustment = 0.0
    selected_candidate = "hgb_squared"
    recommended_candidate = selected_candidate
    selection_mode = "auto"
    correction_weight = 0.0

    if len(training_df) >= 25 and training_df["match_key"].nunique() >= 8:
        train_mask, calibration_mask, test_mask = grouped_chronological_three_way_split(training_df)
        candidate_scores = {}
        candidate_models = {}
        calibration_baseline = baseline[calibration_mask].to_numpy(dtype=float)
        calibration_y = y[calibration_mask].to_numpy(dtype=float)
        for candidate_name in PLAYER_CANDIDATES:
            candidate_model = _new_player_candidate(candidate_name)
            candidate_model.fit(
                x[train_mask],
                residual_target[train_mask],
                sample_weight=weights[train_mask],
            )
            residual_predictions = candidate_model.predict(x[calibration_mask])
            best_weight = 0.0
            best_mae = float(mean_absolute_error(calibration_y, calibration_baseline))
            for blend_weight in np.linspace(0.0, 1.0, 21):
                blended = np.clip(
                    calibration_baseline + blend_weight * residual_predictions,
                    0.35,
                    1.90,
                )
                blend_mae = float(mean_absolute_error(calibration_y, blended))
                if blend_mae < best_mae:
                    best_mae = blend_mae
                    best_weight = float(blend_weight)
            candidate_scores[candidate_name] = {
                "mae": best_mae,
                "correction_weight": best_weight,
            }
            candidate_models[candidate_name] = candidate_model

        rolling_splits = rolling_origin_splits(training_df)
        for candidate_name in PLAYER_CANDIDATES:
            rolling_maes = []
            rolling_baseline_maes = []
            rolling_skills = []
            rolling_wins = []
            for rolling_train, rolling_calibration, rolling_test in rolling_splits:
                rolling_model = _new_player_candidate(candidate_name)
                rolling_model.fit(
                    x[rolling_train],
                    residual_target[rolling_train],
                    sample_weight=weights[rolling_train],
                )
                calibration_residual = rolling_model.predict(x[rolling_calibration])
                calibration_y_fold = y[rolling_calibration].to_numpy(dtype=float)
                calibration_baseline_fold = baseline[rolling_calibration].to_numpy(dtype=float)
                fold_weight = 0.0
                fold_best_mae = float(
                    mean_absolute_error(
                        calibration_y_fold,
                        calibration_baseline_fold,
                    )
                )
                for blend_weight in np.linspace(0.0, 1.0, 21):
                    candidate_predictions = np.clip(
                        calibration_baseline_fold
                        + float(blend_weight) * calibration_residual,
                        0.35,
                        1.90,
                    )
                    candidate_mae = float(
                        mean_absolute_error(
                            calibration_y_fold,
                            candidate_predictions,
                        )
                    )
                    if candidate_mae < fold_best_mae:
                        fold_best_mae = candidate_mae
                        fold_weight = float(blend_weight)

                rolling_baseline = baseline[rolling_test].to_numpy(dtype=float)
                rolling_predictions = np.clip(
                    rolling_baseline
                    + fold_weight * rolling_model.predict(x[rolling_test]),
                    0.35,
                    1.90,
                )
                rolling_y = y[rolling_test].to_numpy(dtype=float)
                rolling_mae = float(mean_absolute_error(rolling_y, rolling_predictions))
                rolling_baseline_mae = float(
                    mean_absolute_error(rolling_y, rolling_baseline)
                )
                rolling_maes.append(rolling_mae)
                rolling_baseline_maes.append(rolling_baseline_mae)
                rolling_skills.append(
                    1.0 - rolling_mae / rolling_baseline_mae
                    if rolling_baseline_mae
                    else 0.0
                )
                rolling_wins.append(rolling_mae + 0.001 < rolling_baseline_mae)
            candidate_scores[candidate_name].update(
                {
                    "rolling_folds": len(rolling_maes),
                    "rolling_mae_mean": float(np.mean(rolling_maes))
                    if rolling_maes
                    else None,
                    "rolling_baseline_mae_mean": float(
                        np.mean(rolling_baseline_maes)
                    )
                    if rolling_baseline_maes
                    else None,
                    "rolling_skill_mean": float(np.mean(rolling_skills))
                    if rolling_skills
                    else None,
                    "rolling_wins": int(sum(rolling_wins)),
                }
            )

        recommended_candidate = min(
            candidate_scores,
            key=lambda name: (
                candidate_scores[name].get("rolling_mae_mean")
                if candidate_scores[name].get("rolling_mae_mean") is not None
                else candidate_scores[name]["mae"]
            ),
        )
        selected_candidate, selection_mode = resolve_selected_candidate(
            "player",
            requested_candidate,
            recommended_candidate,
        )
        if selected_candidate == "last_10_baseline":
            correction_weight = 0.0
            evaluation_model = candidate_models[recommended_candidate]
        else:
            correction_weight = candidate_scores[selected_candidate]["correction_weight"]
            evaluation_model = candidate_models[selected_candidate]
        evaluation_lower = _new_player_model(loss="quantile", quantile=0.10)
        evaluation_upper = _new_player_model(loss="quantile", quantile=0.90)
        evaluation_lower.fit(
            x[train_mask],
            residual_target[train_mask],
            sample_weight=weights[train_mask],
        )
        evaluation_upper.fit(
            x[train_mask],
            residual_target[train_mask],
            sample_weight=weights[train_mask],
        )

        calibration_low_raw = calibration_baseline + evaluation_lower.predict(x[calibration_mask])
        calibration_high_raw = calibration_baseline + evaluation_upper.predict(x[calibration_mask])
        calibration_low = np.minimum(calibration_low_raw, calibration_high_raw)
        calibration_high = np.maximum(calibration_low_raw, calibration_high_raw)
        nonconformity = np.maximum.reduce(
            [
                calibration_low - calibration_y,
                calibration_y - calibration_high,
                np.zeros(len(calibration_y)),
            ]
        )
        interval_adjustment = float(np.quantile(nonconformity, 0.80, method="higher"))

        baseline_preds = baseline[test_mask]
        preds = (
            baseline_preds
            if selected_candidate == "last_10_baseline"
            else (
                baseline_preds
                + correction_weight * evaluation_model.predict(x[test_mask])
            ).clip(0.35, 1.90)
        )
        y_test = y[test_mask]
        test_low_raw = baseline_preds.to_numpy(dtype=float) + evaluation_lower.predict(x[test_mask])
        test_high_raw = baseline_preds.to_numpy(dtype=float) + evaluation_upper.predict(x[test_mask])
        test_low = np.minimum(test_low_raw, test_high_raw) - interval_adjustment
        test_high = np.maximum(test_low_raw, test_high_raw) + interval_adjustment
        mae = float(mean_absolute_error(y_test, preds))
        baseline_mae = float(mean_absolute_error(y_test, baseline_preds))
        model_reliability = max(0.0, min(1.0, (baseline_mae - mae) / baseline_mae)) if baseline_mae else 0.0
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "training_rows": int(train_mask.sum()),
            "calibration_rows": int(calibration_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_matches": int(training_df.loc[test_mask, "match_key"].nunique()),
            "split_strategy": "chronological_match_grouped_train_calibration_test",
            "mae": mae,
            "r2": float(r2_score(y_test, preds)),
            "baseline": "hierarchical_shrunk_form",
            "baseline_mae": baseline_mae,
            "selected_candidate": selected_candidate,
            "recommended_candidate": recommended_candidate,
            "selection_mode": selection_mode,
            "candidate_calibration": candidate_scores,
            "correction_weight": correction_weight,
            "skill_vs_baseline": model_reliability,
            "model_reliability": model_reliability,
            "residual_std": float(np.std(y_test.to_numpy(dtype=float) - preds)),
            "interval_nominal_coverage": 0.80,
            "interval_test_coverage": float(np.mean((y_test >= test_low) & (y_test <= test_high))),
            "interval_mean_width": float(np.mean(test_high - test_low)),
            "interval_adjustment": interval_adjustment,
            "test_start": str(training_df.loc[test_mask, "target_date"].min()),
        }
        rolling_metrics = candidate_scores.get(selected_candidate, {})
        metrics["rolling_backtest_folds"] = int(
            rolling_metrics.get("rolling_folds", 0) or 0
        )
        metrics["rolling_mae_mean"] = rolling_metrics.get("rolling_mae_mean")
        metrics["rolling_baseline_mae_mean"] = rolling_metrics.get(
            "rolling_baseline_mae_mean"
        )
        metrics["rolling_skill_vs_baseline_mean"] = rolling_metrics.get(
            "rolling_skill_mean"
        )
        metrics["rolling_wins"] = int(rolling_metrics.get("rolling_wins", 0) or 0)
    else:
        preds = y.mean() + np.zeros(len(y))
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "validation_rows": 0,
            "split_strategy": "insufficient_matches_for_holdout",
            "train_mae": float(mean_absolute_error(y, preds)),
            "train_r2": 0.0,
            "model_reliability": 0.0,
            "selected_candidate": selected_candidate,
            "recommended_candidate": recommended_candidate,
            "selection_mode": selection_mode,
            "correction_weight": correction_weight,
            "residual_std": float(np.std(y.to_numpy(dtype=float) - preds)),
        }

    metrics["evaluated_correction_weight"] = correction_weight
    rolling_folds = int(metrics.get("rolling_backtest_folds", 0) or 0)
    rolling_wins = int(metrics.get("rolling_wins", 0) or 0)
    holdout_pass = (
        selected_candidate == "last_10_baseline"
        or float(metrics.get("mae", float("inf"))) + 0.001
        < float(metrics.get("baseline_mae", float("inf")))
    )
    rolling_pass = (
        selected_candidate == "last_10_baseline"
        or (
            rolling_folds >= 2
            and rolling_wins >= math.ceil(rolling_folds * 2.0 / 3.0)
            and float(metrics.get("rolling_skill_vs_baseline_mean") or 0.0) > 0.005
        )
    )
    enabled = selected_candidate == "last_10_baseline" or (holdout_pass and rolling_pass)
    active_for_predictions = enabled or selection_mode == "manual"
    metrics["holdout_safeguard_passed"] = holdout_pass
    metrics["rolling_safeguard_passed"] = rolling_pass
    metrics["enabled"] = enabled
    metrics["active_for_predictions"] = active_for_predictions
    if selection_mode == "auto" and not enabled:
        correction_weight = 0.0
        metrics["model_reliability"] = 0.0
    metrics["correction_weight"] = correction_weight

    fit_candidate = (
        recommended_candidate
        if selected_candidate == "last_10_baseline"
        else selected_candidate
    )
    point_model = _new_player_candidate(fit_candidate)
    point_model.fit(x, residual_target, sample_weight=weights)
    lower_model.fit(x, residual_target, sample_weight=weights)
    upper_model.fit(x, residual_target, sample_weight=weights)
    return {
        "point": point_model,
        "lower": lower_model,
        "upper": upper_model,
        "interval_adjustment": interval_adjustment,
        "selected_candidate": selected_candidate,
        "recommended_candidate": recommended_candidate,
        "selection_mode": selection_mode,
        "correction_weight": correction_weight,
    }, metrics


def _new_team_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=20,
        min_samples_leaf=24,
        l2_regularization=2.0,
        random_state=7,
    )


PROBABILITY_CANDIDATES = (
    "hgb_residual",
    "hgb_classifier",
    "logistic_classifier",
)

PROBABILITY_BLEND_WEIGHTS = tuple(np.linspace(0.0, 1.0, 21))
PROBABILITY_TEMPERATURES = (0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)
PROBABILITY_MAX_MODEL_DELTA = 0.20
MAP_MODEL_MAX_CORRECTION = 0.10


def _new_probability_candidate(name: str, min_samples_leaf: int = 24):
    if name == "hgb_classifier":
        return HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=180,
            max_leaf_nodes=20,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=2.0,
            random_state=7,
        ), "classifier"
    if name == "logistic_classifier":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=1200, random_state=7),
        ), "classifier"
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=20,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=2.0,
        random_state=7,
    ), "residual"


def _fit_probability_candidate(
    model,
    kind: str,
    x: pd.DataFrame,
    y: pd.Series,
    baseline: pd.Series,
    weights: pd.Series,
) -> None:
    target = y - baseline if kind == "residual" else y
    if hasattr(model, "steps"):
        final_step = model.steps[-1][0]
        model.fit(x, target, **{f"{final_step}__sample_weight": weights})
    else:
        model.fit(x, target, sample_weight=weights)


def _predict_probability_candidate(
    model,
    kind: str,
    x: pd.DataFrame,
    baseline,
) -> np.ndarray:
    baseline_values = np.asarray(baseline, dtype=float)
    if kind == "residual":
        return np.clip(baseline_values + model.predict(x), 0.03, 0.97)
    return np.clip(model.predict_proba(x)[:, 1], 0.03, 0.97)


def apply_probability_blend_temperature(
    model_probabilities,
    baseline_probabilities,
    blend_weight: float = 1.0,
    temperature: float = 1.0,
    max_model_delta: float = PROBABILITY_MAX_MODEL_DELTA,
) -> np.ndarray:
    model_values = np.asarray(model_probabilities, dtype=float)
    baseline_values = np.asarray(baseline_probabilities, dtype=float)
    model_values = np.clip(
        model_values,
        baseline_values - float(max_model_delta),
        baseline_values + float(max_model_delta),
    )
    blended = baseline_values + float(blend_weight) * (
        model_values - baseline_values
    )
    clipped = np.clip(blended, 0.02, 0.98)
    logits = np.log(clipped / (1.0 - clipped)) / max(0.05, float(temperature))
    return np.clip(1.0 / (1.0 + np.exp(-logits)), 0.02, 0.98)


def optimize_probability_blend_temperature(
    model_probabilities,
    baseline_probabilities,
    targets,
) -> dict:
    targets = np.asarray(targets, dtype=float)
    best = None
    for blend_weight in PROBABILITY_BLEND_WEIGHTS:
        for temperature in PROBABILITY_TEMPERATURES:
            probabilities = apply_probability_blend_temperature(
                model_probabilities,
                baseline_probabilities,
                blend_weight=blend_weight,
                temperature=temperature,
            )
            score = float(log_loss(targets, probabilities, labels=[0, 1]))
            if best is None or score < best["log_loss"]:
                best = {
                    "log_loss": score,
                    "brier_score": float(brier_score_loss(targets, probabilities)),
                    "accuracy": float(accuracy_score(targets, probabilities >= 0.5)),
                    "model_blend_weight": float(blend_weight),
                    "probability_temperature": float(temperature),
                    "max_model_delta": PROBABILITY_MAX_MODEL_DELTA,
                }
    return best


def apply_team_anchored_map_correction(
    model_probabilities,
    baseline_probabilities,
    map_reliabilities,
    blend_weight: float = 1.0,
    max_model_delta: float = MAP_MODEL_MAX_CORRECTION,
) -> np.ndarray:
    model_values = np.asarray(model_probabilities, dtype=float)
    baseline_values = np.asarray(baseline_probabilities, dtype=float)
    reliabilities = np.clip(np.asarray(map_reliabilities, dtype=float), 0.0, 1.0)
    model_values = np.clip(
        model_values,
        baseline_values - float(max_model_delta),
        baseline_values + float(max_model_delta),
    )
    corrected = baseline_values + (
        float(blend_weight)
        * reliabilities
        * (model_values - baseline_values)
    )
    return np.clip(corrected, 0.02, 0.98)


def optimize_team_anchored_map_correction(
    model_probabilities,
    baseline_probabilities,
    map_reliabilities,
    targets,
) -> dict:
    targets = np.asarray(targets, dtype=float)
    best = None
    for blend_weight in PROBABILITY_BLEND_WEIGHTS:
        probabilities = apply_team_anchored_map_correction(
            model_probabilities,
            baseline_probabilities,
            map_reliabilities,
            blend_weight=blend_weight,
        )
        score = float(log_loss(targets, probabilities, labels=[0, 1]))
        if best is None or score < best["log_loss"]:
            best = {
                "log_loss": score,
                "brier_score": float(brier_score_loss(targets, probabilities)),
                "accuracy": float(accuracy_score(targets, probabilities >= 0.5)),
                "model_blend_weight": float(blend_weight),
                "probability_temperature": 1.0,
                "max_model_delta": MAP_MODEL_MAX_CORRECTION,
            }
    return best


def _select_probability_candidate(
    x: pd.DataFrame,
    y: pd.Series,
    baseline: pd.Series,
    weights: pd.Series,
    train_mask: pd.Series,
    calibration_mask: pd.Series,
    min_samples_leaf: int = 24,
    training_df: pd.DataFrame | None = None,
    map_reliabilities: pd.Series | None = None,
) -> tuple[str, str, object, dict]:
    scores = {}
    models = {}
    kinds = {}
    for candidate_name in PROBABILITY_CANDIDATES:
        model, kind = _new_probability_candidate(candidate_name, min_samples_leaf)
        _fit_probability_candidate(
            model,
            kind,
            x[train_mask],
            y[train_mask],
            baseline[train_mask],
            weights[train_mask],
        )
        probabilities = _predict_probability_candidate(
            model,
            kind,
            x[calibration_mask],
            baseline[calibration_mask],
        )
        calibration_y = y[calibration_mask]
        if map_reliabilities is None:
            raw_evaluation = probabilities
            scores[candidate_name] = optimize_probability_blend_temperature(
                probabilities,
                baseline[calibration_mask],
                calibration_y,
            )
        else:
            raw_evaluation = apply_team_anchored_map_correction(
                probabilities,
                baseline[calibration_mask],
                map_reliabilities[calibration_mask],
            )
            scores[candidate_name] = optimize_team_anchored_map_correction(
                probabilities,
                baseline[calibration_mask],
                map_reliabilities[calibration_mask],
                calibration_y,
            )
        raw_log_loss = float(
            log_loss(calibration_y, raw_evaluation, labels=[0, 1])
        )
        scores[candidate_name]["raw_log_loss"] = raw_log_loss
        models[candidate_name] = model
        kinds[candidate_name] = kind

    if training_df is not None:
        rolling_splits = rolling_origin_splits(training_df)
        for candidate_name in PROBABILITY_CANDIDATES:
            rolling_log_losses = []
            rolling_brier_scores = []
            rolling_accuracies = []
            rolling_reference_log_losses = []
            rolling_reference_briers = []
            rolling_wins = []
            for rolling_train, rolling_calibration, rolling_test in rolling_splits:
                rolling_model, rolling_kind = _new_probability_candidate(
                    candidate_name,
                    min_samples_leaf,
                )
                _fit_probability_candidate(
                    rolling_model,
                    rolling_kind,
                    x[rolling_train],
                    y[rolling_train],
                    baseline[rolling_train],
                    weights[rolling_train],
                )
                calibration_probabilities = _predict_probability_candidate(
                    rolling_model,
                    rolling_kind,
                    x[rolling_calibration],
                    baseline[rolling_calibration],
                )
                if map_reliabilities is None:
                    settings = optimize_probability_blend_temperature(
                        calibration_probabilities,
                        baseline[rolling_calibration],
                        y[rolling_calibration],
                    )
                else:
                    settings = optimize_team_anchored_map_correction(
                        calibration_probabilities,
                        baseline[rolling_calibration],
                        map_reliabilities[rolling_calibration],
                        y[rolling_calibration],
                    )
                raw_probabilities = _predict_probability_candidate(
                    rolling_model,
                    rolling_kind,
                    x[rolling_test],
                    baseline[rolling_test],
                )
                if map_reliabilities is None:
                    probabilities = apply_probability_blend_temperature(
                        raw_probabilities,
                        baseline[rolling_test],
                        blend_weight=settings["model_blend_weight"],
                        temperature=settings["probability_temperature"],
                        max_model_delta=settings["max_model_delta"],
                    )
                else:
                    probabilities = apply_team_anchored_map_correction(
                        raw_probabilities,
                        baseline[rolling_test],
                        map_reliabilities[rolling_test],
                        blend_weight=settings["model_blend_weight"],
                        max_model_delta=settings["max_model_delta"],
                    )
                test_y = y[rolling_test]
                baseline_probabilities = baseline[rolling_test].to_numpy(dtype=float)
                coin_probabilities = np.full(len(test_y), 0.5)
                candidate_log_loss = float(
                    log_loss(test_y, probabilities, labels=[0, 1])
                )
                candidate_brier = float(brier_score_loss(test_y, probabilities))
                reference_log_loss = min(
                    float(log_loss(test_y, baseline_probabilities, labels=[0, 1])),
                    float(log_loss(test_y, coin_probabilities, labels=[0, 1])),
                )
                reference_brier = min(
                    float(brier_score_loss(test_y, baseline_probabilities)),
                    0.25,
                )
                rolling_log_losses.append(candidate_log_loss)
                rolling_brier_scores.append(candidate_brier)
                rolling_accuracies.append(
                    float(accuracy_score(test_y, probabilities >= 0.5))
                )
                rolling_reference_log_losses.append(reference_log_loss)
                rolling_reference_briers.append(reference_brier)
                rolling_wins.append(
                    candidate_log_loss + 0.001 < reference_log_loss
                    and candidate_brier + 0.0005 < reference_brier
                )
            scores[candidate_name].update(
                {
                    "rolling_folds": len(rolling_log_losses),
                    "rolling_log_loss_mean": float(np.mean(rolling_log_losses))
                    if rolling_log_losses
                    else None,
                    "rolling_brier_mean": float(np.mean(rolling_brier_scores))
                    if rolling_brier_scores
                    else None,
                    "rolling_accuracy_mean": float(np.mean(rolling_accuracies))
                    if rolling_accuracies
                    else None,
                    "rolling_reference_log_loss_mean": float(
                        np.mean(rolling_reference_log_losses)
                    )
                    if rolling_reference_log_losses
                    else None,
                    "rolling_reference_brier_mean": float(
                        np.mean(rolling_reference_briers)
                    )
                    if rolling_reference_briers
                    else None,
                    "rolling_wins": int(sum(rolling_wins)),
                }
            )

    selected = min(
        scores,
        key=lambda name: (
            scores[name].get("rolling_log_loss_mean")
            if scores[name].get("rolling_log_loss_mean") is not None
            else scores[name]["log_loss"]
        ),
    )
    return selected, kinds[selected], models[selected], scores


def apply_symmetric_calibration(
    calibrator: IsotonicRegression | None,
    probabilities,
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if calibrator is None:
        return raw
    forward = calibrator.predict(raw)
    reverse = 1.0 - calibrator.predict(1.0 - raw)
    return np.clip(0.5 * (forward + reverse), 0.02, 0.98)


def train_team_model(
    training_df: pd.DataFrame,
    requested_candidate: str = "auto",
) -> tuple[object | None, IsotonicRegression | None, dict]:
    if len(training_df) < 6:
        return None, None, {"status": "skipped", "reason": "Need at least 6 team training rows."}
    if training_df["target_win"].nunique() < 2:
        return None, None, {"status": "skipped", "reason": "Need both wins and losses in team training rows."}

    x = training_df[TEAM_FEATURES].fillna(0.0)
    y = training_df["target_win"].astype(float)
    baseline = training_df["elo_probability"].fillna(0.5).astype(float).clip(0.05, 0.95)
    weights = training_df.get("training_weight", pd.Series(1.0, index=training_df.index)).astype(float)
    calibrator = None
    selected_candidate = "hgb_residual"
    recommended_candidate = selected_candidate
    selection_mode = "auto"
    model_kind = "residual"
    candidate_scores = {}
    model_blend_weight = 0.0
    probability_temperature = 1.0
    max_model_delta = PROBABILITY_MAX_MODEL_DELTA

    if len(training_df) >= 30 and y.value_counts().min() >= 4 and training_df["match_key"].nunique() >= 12:
        train_mask, calibration_mask, test_mask = grouped_chronological_three_way_split(training_df)
        recommended_candidate, recommended_kind, recommended_model, candidate_scores = (
            _select_probability_candidate(
                x,
                y,
                baseline,
                weights,
                train_mask,
                calibration_mask,
                training_df=training_df,
            )
        )
        selected_candidate, selection_mode = resolve_selected_candidate(
            "team",
            requested_candidate,
            recommended_candidate,
        )
        if selected_candidate == "elo_baseline":
            evaluation_model = None
            model_kind = "baseline"
            calibration_probs = baseline[calibration_mask].to_numpy(dtype=float)
        elif selected_candidate == recommended_candidate:
            evaluation_model = recommended_model
            model_kind = recommended_kind
            calibration_probs = _predict_probability_candidate(
                evaluation_model,
                model_kind,
                x[calibration_mask],
                baseline[calibration_mask],
            )
        else:
            evaluation_model, model_kind = _new_probability_candidate(selected_candidate)
            _fit_probability_candidate(
                evaluation_model,
                model_kind,
                x[train_mask],
                y[train_mask],
                baseline[train_mask],
                weights[train_mask],
            )
            calibration_probs = _predict_probability_candidate(
                evaluation_model,
                model_kind,
                x[calibration_mask],
                baseline[calibration_mask],
            )
        if model_kind == "baseline":
            calibration_method = "none"
        else:
            calibration_settings = candidate_scores[selected_candidate]
            model_blend_weight = float(
                calibration_settings["model_blend_weight"]
            )
            probability_temperature = float(
                calibration_settings["probability_temperature"]
            )
            max_model_delta = float(calibration_settings["max_model_delta"])
            calibration_method = "baseline_blend_temperature"

        raw_test_probs = (
            baseline[test_mask].to_numpy(dtype=float)
            if model_kind == "baseline"
            else _predict_probability_candidate(
                evaluation_model,
                model_kind,
                x[test_mask],
                baseline[test_mask],
            )
        )
        y_test = y[test_mask]
        baseline_probs = baseline[test_mask].to_numpy(dtype=float)
        probs = (
            raw_test_probs
            if model_kind == "baseline"
            else apply_probability_blend_temperature(
                raw_test_probs,
                baseline_probs,
                blend_weight=model_blend_weight,
                temperature=probability_temperature,
                max_model_delta=max_model_delta,
            )
        )
        coin_probs = np.full(len(y_test), 0.5)
        model_log_loss = float(log_loss(y_test, probs, labels=[0, 1]))
        coin_log_loss = float(log_loss(y_test, coin_probs, labels=[0, 1]))
        baseline_log_loss = float(log_loss(y_test, baseline_probs, labels=[0, 1]))
        model_brier = float(brier_score_loss(y_test, probs))
        baseline_brier = float(brier_score_loss(y_test, baseline_probs))
        accuracy = float(accuracy_score(y_test, probs >= 0.5))
        baseline_accuracy = float(accuracy_score(y_test, baseline_probs >= 0.5))
        reference_log_loss = min(coin_log_loss, baseline_log_loss)
        reference_brier = min(0.25, baseline_brier)
        reference_accuracy = max(0.5, baseline_accuracy)
        log_skill = max(0.0, 1.0 - model_log_loss / reference_log_loss)
        brier_skill = max(0.0, 1.0 - model_brier / reference_brier)
        accuracy_skill = max(
            0.0,
            (accuracy - reference_accuracy) / max(0.01, 1.0 - reference_accuracy),
        )
        model_reliability = min(1.0, (log_skill + brier_skill + accuracy_skill) / 3.0)
        rolling_metrics = candidate_scores.get(selected_candidate, {})
        rolling_folds = int(rolling_metrics.get("rolling_folds", 0) or 0)
        rolling_wins = int(rolling_metrics.get("rolling_wins", 0) or 0)
        rolling_model_log_loss = rolling_metrics.get("rolling_log_loss_mean")
        rolling_reference_log_loss = rolling_metrics.get(
            "rolling_reference_log_loss_mean"
        )
        rolling_model_brier = rolling_metrics.get("rolling_brier_mean")
        rolling_reference_brier = rolling_metrics.get(
            "rolling_reference_brier_mean"
        )
        holdout_pass = (
            model_log_loss + 0.002 < reference_log_loss
            and model_brier + 0.001 < reference_brier
        )
        rolling_pass = (
            rolling_folds >= 2
            and rolling_wins >= math.ceil(rolling_folds * 2.0 / 3.0)
            and rolling_model_log_loss is not None
            and rolling_reference_log_loss is not None
            and rolling_model_log_loss + 0.001 < rolling_reference_log_loss
            and rolling_model_brier is not None
            and rolling_reference_brier is not None
            and rolling_model_brier + 0.0005 < rolling_reference_brier
        )
        enabled = model_kind == "baseline" or (holdout_pass and rolling_pass)
        active_for_predictions = enabled or selection_mode == "manual"
        if not enabled and selection_mode == "auto":
            model_reliability = 0.0
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "training_rows": int(train_mask.sum()),
            "calibration_rows": int(calibration_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_matches": int(training_df.loc[test_mask, "match_key"].nunique()),
            "split_strategy": "chronological_match_grouped_train_calibration_test",
            "model_target": "series_win_probability",
            "enabled": enabled,
            "active_for_predictions": active_for_predictions,
            "selected_candidate": selected_candidate,
            "recommended_candidate": recommended_candidate,
            "selection_mode": selection_mode,
            "model_kind": model_kind,
            "candidate_calibration": candidate_scores,
            "accuracy": accuracy,
            "log_loss": model_log_loss,
            "brier_score": model_brier,
            "calibration": calibration_method,
            "model_blend_weight": model_blend_weight,
            "probability_temperature": probability_temperature,
            "max_model_delta": max_model_delta,
            "coinflip_accuracy": 0.5,
            "coinflip_log_loss": coin_log_loss,
            "elo_accuracy": baseline_accuracy,
            "elo_log_loss": baseline_log_loss,
            "elo_brier_score": baseline_brier,
            "reliability_baseline": "best_of_elo_or_coinflip",
            "model_reliability": model_reliability,
            "holdout_safeguard_passed": holdout_pass,
            "rolling_safeguard_passed": rolling_pass,
            "rolling_backtest_folds": rolling_folds,
            "rolling_wins": rolling_wins,
            "rolling_log_loss_mean": rolling_model_log_loss,
            "rolling_reference_log_loss_mean": rolling_reference_log_loss,
            "rolling_brier_mean": rolling_model_brier,
            "rolling_reference_brier_mean": rolling_reference_brier,
            "rolling_accuracy_mean": rolling_metrics.get("rolling_accuracy_mean"),
            "test_start": str(training_df.loc[test_mask, "target_date"].min()),
        }
    else:
        metrics = {
            "status": "trained",
            "rows": len(training_df),
            "validation_rows": 0,
            "split_strategy": "insufficient_matches_for_holdout",
            "selected_candidate": selected_candidate,
            "recommended_candidate": recommended_candidate,
            "selection_mode": selection_mode,
            "model_kind": model_kind,
            "model_reliability": 0.0,
            "model_blend_weight": model_blend_weight,
            "probability_temperature": probability_temperature,
            "max_model_delta": max_model_delta,
        }

    fit_candidate = (
        recommended_candidate if selected_candidate == "elo_baseline" else selected_candidate
    )
    model, fitted_kind = _new_probability_candidate(fit_candidate)
    _fit_probability_candidate(model, fitted_kind, x, y, baseline, weights)
    metrics["model_kind"] = "baseline" if selected_candidate == "elo_baseline" else fitted_kind
    return model, calibrator, metrics


def train_map_model(
    training_df: pd.DataFrame,
    requested_candidate: str = "auto",
) -> tuple[dict | None, dict]:
    if len(training_df) < 20 or training_df.get("target_map_win", pd.Series(dtype=float)).nunique() < 2:
        return None, {"status": "skipped", "reason": "Need scored maps with both wins and losses."}

    map_names = sorted(training_df["map_name"].dropna().astype(str).unique())
    x = map_feature_frame(training_df, map_names)
    y = training_df["target_map_win"].astype(float)
    baseline = training_df["map_baseline_probability"].fillna(0.5).astype(float).clip(0.05, 0.95)
    map_reliabilities = training_df.get(
        "map_history_reliability",
        pd.Series(0.0, index=training_df.index),
    ).fillna(0.0).astype(float).clip(0.0, 1.0)
    weights = training_df.get("training_weight", pd.Series(1.0, index=training_df.index)).astype(float)
    train_mask, calibration_mask, test_mask = grouped_chronological_three_way_split(training_df)
    recommended_candidate, recommended_kind, recommended_model, candidate_scores = _select_probability_candidate(
        x,
        y,
        baseline,
        weights,
        train_mask,
        calibration_mask,
        min_samples_leaf=20,
        training_df=training_df,
        map_reliabilities=map_reliabilities,
    )
    selected_candidate, selection_mode = resolve_selected_candidate(
        "map",
        requested_candidate,
        recommended_candidate,
    )
    model_blend_weight = 0.0
    probability_temperature = 1.0
    max_model_delta = MAP_MODEL_MAX_CORRECTION
    if selected_candidate == "map_form_baseline":
        evaluation_model = None
        model_kind = "baseline"
        calibration_probs = baseline[calibration_mask].to_numpy(dtype=float)
    elif selected_candidate == recommended_candidate:
        evaluation_model = recommended_model
        model_kind = recommended_kind
        calibration_probs = _predict_probability_candidate(
            evaluation_model,
            model_kind,
            x[calibration_mask],
            baseline[calibration_mask],
        )
    else:
        evaluation_model, model_kind = _new_probability_candidate(
            selected_candidate,
            min_samples_leaf=20,
        )
        _fit_probability_candidate(
            evaluation_model,
            model_kind,
            x[train_mask],
            y[train_mask],
            baseline[train_mask],
            weights[train_mask],
        )
        calibration_probs = _predict_probability_candidate(
            evaluation_model,
            model_kind,
            x[calibration_mask],
            baseline[calibration_mask],
        )
    calibrator = None
    if model_kind == "baseline":
        calibration_method = "none"
    else:
        calibration_settings = candidate_scores[selected_candidate]
        model_blend_weight = float(calibration_settings["model_blend_weight"])
        probability_temperature = float(
            calibration_settings["probability_temperature"]
        )
        max_model_delta = float(calibration_settings["max_model_delta"])
        calibration_method = "team_anchored_reliability_scaled_correction"

    raw_test = (
        baseline[test_mask].to_numpy(dtype=float)
        if model_kind == "baseline"
        else _predict_probability_candidate(
            evaluation_model,
            model_kind,
            x[test_mask],
            baseline[test_mask],
        )
    )
    y_test = y[test_mask]
    baseline_test = baseline[test_mask].to_numpy(dtype=float)
    probabilities = (
        raw_test
        if model_kind == "baseline"
        else apply_team_anchored_map_correction(
            raw_test,
            baseline_test,
            map_reliabilities[test_mask],
            blend_weight=model_blend_weight,
            max_model_delta=max_model_delta,
        )
    )
    model_log_loss = float(log_loss(y_test, probabilities, labels=[0, 1]))
    baseline_log_loss = float(log_loss(y_test, baseline_test, labels=[0, 1]))
    coin_probabilities = np.full(len(y_test), 0.5)
    coin_log_loss = float(log_loss(y_test, coin_probabilities, labels=[0, 1]))
    model_brier = float(brier_score_loss(y_test, probabilities))
    baseline_brier = float(brier_score_loss(y_test, baseline_test))
    accuracy = float(accuracy_score(y_test, probabilities >= 0.5))
    baseline_accuracy = float(accuracy_score(y_test, baseline_test >= 0.5))
    reference_log_loss = min(coin_log_loss, baseline_log_loss)
    reference_brier = min(0.25, baseline_brier)
    reference_accuracy = max(0.5, baseline_accuracy)
    log_skill = max(0.0, 1.0 - model_log_loss / reference_log_loss) if reference_log_loss else 0.0
    brier_skill = max(0.0, 1.0 - model_brier / reference_brier) if reference_brier else 0.0
    accuracy_skill = max(
        0.0,
        (accuracy - reference_accuracy) / max(0.01, 1.0 - reference_accuracy),
    )
    model_reliability = min(1.0, (log_skill + brier_skill + accuracy_skill) / 3.0)
    rolling_metrics = candidate_scores.get(selected_candidate, {})
    rolling_folds = int(rolling_metrics.get("rolling_folds", 0) or 0)
    rolling_wins = int(rolling_metrics.get("rolling_wins", 0) or 0)
    rolling_model_log_loss = rolling_metrics.get("rolling_log_loss_mean")
    rolling_reference_log_loss = rolling_metrics.get("rolling_reference_log_loss_mean")
    rolling_model_brier = rolling_metrics.get("rolling_brier_mean")
    rolling_reference_brier = rolling_metrics.get("rolling_reference_brier_mean")
    holdout_pass = (
        model_log_loss + 0.002 < reference_log_loss
        and model_brier + 0.001 < reference_brier
    )
    rolling_pass = (
        rolling_folds >= 2
        and rolling_wins >= math.ceil(rolling_folds * 2.0 / 3.0)
        and rolling_model_log_loss is not None
        and rolling_reference_log_loss is not None
        and rolling_model_log_loss + 0.001 < rolling_reference_log_loss
        and rolling_model_brier is not None
        and rolling_reference_brier is not None
        and rolling_model_brier + 0.0005 < rolling_reference_brier
    )
    enabled = model_kind == "baseline" or (holdout_pass and rolling_pass)
    active_for_predictions = enabled or selection_mode == "manual"
    if not enabled and selection_mode == "auto":
        model_reliability = 0.0
    metrics = {
        "status": "trained",
        "rows": len(training_df),
        "maps": int(training_df[["match_key", "map_id"]].drop_duplicates().shape[0]),
        "training_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_matches": int(training_df.loc[test_mask, "match_key"].nunique()),
        "split_strategy": "chronological_match_grouped_train_calibration_test",
        "selected_candidate": selected_candidate,
        "recommended_candidate": recommended_candidate,
        "selection_mode": selection_mode,
        "model_kind": model_kind,
        "candidate_calibration": candidate_scores,
        "accuracy": accuracy,
        "log_loss": model_log_loss,
        "brier_score": model_brier,
        "baseline_accuracy": baseline_accuracy,
        "baseline_log_loss": baseline_log_loss,
        "baseline_brier_score": baseline_brier,
        "coinflip_accuracy": 0.5,
        "coinflip_log_loss": coin_log_loss,
        "reliability_baseline": "best_of_team_anchored_map_form_or_coinflip",
        "probability_structure": "team_anchor_plus_reliability_scaled_map_correction",
        "minimum_team_anchor_weight": 1.0 - MAX_MAP_SPECIFIC_WEIGHT,
        "maximum_map_specific_weight": MAX_MAP_SPECIFIC_WEIGHT,
        "mean_map_history_reliability": float(map_reliabilities.mean()),
        "model_reliability": model_reliability,
        "holdout_safeguard_passed": holdout_pass,
        "rolling_safeguard_passed": rolling_pass,
        "rolling_backtest_folds": rolling_folds,
        "rolling_wins": rolling_wins,
        "rolling_log_loss_mean": rolling_model_log_loss,
        "rolling_reference_log_loss_mean": rolling_reference_log_loss,
        "rolling_brier_mean": rolling_model_brier,
        "rolling_reference_brier_mean": rolling_reference_brier,
        "rolling_accuracy_mean": rolling_metrics.get("rolling_accuracy_mean"),
        "enabled": enabled,
        "active_for_predictions": active_for_predictions,
        "calibration": calibration_method,
        "model_blend_weight": model_blend_weight,
        "probability_temperature": probability_temperature,
        "max_model_delta": max_model_delta,
        "test_start": str(training_df.loc[test_mask, "target_date"].min()),
        "map_names": map_names,
    }

    fit_candidate = (
        recommended_candidate
        if selected_candidate == "map_form_baseline"
        else selected_candidate
    )
    final_model, final_kind = _new_probability_candidate(fit_candidate, min_samples_leaf=20)
    _fit_probability_candidate(final_model, final_kind, x, y, baseline, weights)
    return {
        "model": final_model,
        "model_kind": "baseline" if selected_candidate == "map_form_baseline" else final_kind,
        "calibrator": calibrator,
        "model_blend_weight": model_blend_weight,
        "probability_temperature": probability_temperature,
        "max_model_delta": max_model_delta,
        "maximum_map_specific_weight": MAX_MAP_SPECIFIC_WEIGHT,
        "hierarchical_team_anchor": True,
        "map_names": map_names,
        "features": list(x.columns),
    }, metrics


def save_model(path: str, payload: dict) -> None:
    ensure_model_dir()
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def dataset_fingerprint(matches: pd.DataFrame) -> str:
    columns = [
        column
        for column in [
            "match_id",
            "map_id",
            "event_id",
            "season_year",
            "team",
            "opponent",
            "player",
            "player_id",
            "vlr_rating",
            "acs",
            "kills",
            "deaths",
            "assists",
            "event_name",
            "event_stage",
            "event_tier",
            "is_lan",
            "patch",
            "map_veto",
            "map_pick_team",
            "map_pick_type",
            "map_veto_order",
            "match_importance",
            "competition_tier",
            "competition_strength_weight",
            "team_tier_at_match",
            "opponent_tier_at_match",
            "team_promoted_next_season",
            "opponent_promoted_next_season",
        ]
        if column in matches.columns
    ]
    if not columns:
        return hashlib.sha256(b"empty").hexdigest()
    stable = matches[columns].copy().fillna("").astype(str).sort_values(columns).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(stable, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def coverage_summary(matches: pd.DataFrame, min_team_matches: int, season_year: int | None) -> dict:
    coverage = match_coverage_report(
        matches,
        registry_team_pages(include_missing=True),
        min_matches_per_team=min_team_matches,
        season_year=season_year,
    )
    if coverage.empty:
        return {
            "target_matches_per_team": min_team_matches,
            "teams_at_target": 0,
            "teams_below_target": 0,
            "missing_url_count": 0,
            "lowest_coverage": [],
        }

    coverage.to_csv(MATCH_COVERAGE_CSV, index=False)
    below = coverage[coverage["status"] != "ok"]
    return {
        "target_matches_per_team": min_team_matches,
        "teams_at_target": int((coverage["status"] == "ok").sum()),
        "teams_below_target": int(len(below)),
        "missing_url_count": int((coverage["status"] == "missing_url").sum()),
        "lowest_coverage": below.sort_values(["matches", "team"]).head(12).to_dict("records"),
    }


def train_models(
    matches_csv: str = MATCHES_CSV,
    season_year: int | None = None,
    recent_days: int = 60,
    fallback_years: int | None = None,
    min_player_history_maps: int = 1,
    min_team_matches: int = 20,
    force: bool = False,
    model_selection: dict | None = None,
) -> dict:
    selections = normalize_model_selection(
        model_selection if model_selection is not None else load_model_selection()
    )
    matches = load_csv(matches_csv)
    if matches.empty:
        raise ValueError("No match data found. Run python scripts\\scrape.py --matches first.")
    matches = canonicalize_match_dataframe(
        matches,
        registry_team_pages(season_year=None),
    )
    matches = enrich_match_metadata(
        matches,
        registry=load_team_registry(),
        rosters=load_csv(ROSTERS_CSV),
    )
    source_matches = matches.copy()
    source_quality = data_quality_report(matches)
    matches = filter_registry_tier1_matchups(
        filter_curated_competition_history(matches)
    )
    matches = filter_training_ready_matches(matches)
    training_quality = data_quality_report(matches)
    quality_payload = {
        "source": source_quality,
        "training": training_quality,
        "validation": matches.attrs.get("training_validation", {}),
    }
    Path(DATA_QUALITY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DATA_QUALITY_PATH).write_text(
        json.dumps(quality_payload, indent=2),
        encoding="utf-8",
    )
    if matches.empty:
        raise ValueError("No classified, complete Tier 1 matches are ready for training.")
    fingerprint = dataset_fingerprint(matches)
    metrics_path = Path(METRICS_PATH)
    if (
        not force
        and metrics_path.exists()
        and Path(PLAYER_MODEL_PATH).exists()
        and Path(TEAM_MODEL_PATH).exists()
        and Path(MAP_MODEL_PATH).exists()
    ):
        try:
            existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_metrics = {}
        if (
            existing_metrics.get("dataset_fingerprint") == fingerprint
            and existing_metrics.get("model_version") == MODEL_VERSION
            and existing_metrics.get("model_selection") == selections
        ):
            return {**existing_metrics, "training_status": "skipped_unchanged"}

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
    sequential_state = team_training.attrs.get("sequential_state")
    map_training = build_map_training_data(
        matches,
        team_training,
        season_year=season_year,
    )

    player_models, player_metrics = train_player_model(
        player_training,
        selections["player"],
    )
    team_model, team_calibrator, team_metrics = train_team_model(
        team_training,
        selections["team"],
    )
    map_model_payload, map_metrics = train_map_model(
        map_training,
        selections["map"],
    )
    news_history = load_csv(NEWS_CSV)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "training_status": "trained",
        "model_version": MODEL_VERSION,
        "dataset_fingerprint": fingerprint,
        "matches_csv": str(matches_csv),
        "season_year": season_year,
        "recent_days": recent_days,
        "model_selection": selections,
        "history_scope": "curated_vct_history_plus_current_season",
        "training_half_life_days": TRAINING_HALF_LIFE_DAYS,
        "minimum_player_history_maps": 1,
        "team_match_coverage": coverage_summary(
            source_matches,
            min_team_matches,
            season_year,
        ),
        "match_schema": "player-map-v7-quality-region-patch-veto",
        "data_quality": quality_payload,
        "player_training_rows": len(player_training),
        "team_training_rows": len(team_training),
        "map_training_rows": len(map_training),
        "player_effective_training_weight": float(
            player_training.get("training_weight", pd.Series(dtype=float)).sum()
        ),
        "team_effective_training_weight": float(
            team_training.get("training_weight", pd.Series(dtype=float)).sum()
        ),
        "map_effective_training_weight": float(
            map_training.get("training_weight", pd.Series(dtype=float)).sum()
        ),
        "news_history_rows": len(news_history),
        "news_training_status": "structured_rules_only_until_historical_event_coverage_is_sufficient",
        "player_metrics": player_metrics,
        "team_metrics": team_metrics,
        "map_metrics": map_metrics,
    }

    if player_models is not None:
        save_model(
            PLAYER_MODEL_PATH,
            {
                "model": player_models["point"],
                "lower_model": player_models["lower"],
                "upper_model": player_models["upper"],
                "interval_adjustment": player_models["interval_adjustment"],
                "selected_candidate": player_models["selected_candidate"],
                "recommended_candidate": player_models["recommended_candidate"],
                "selection_mode": player_models["selection_mode"],
                "correction_weight": player_models["correction_weight"],
                "baseline_feature": "player_shrunk_rating",
                "features": PLAYER_FEATURES,
                "residual_std": player_metrics.get("residual_std", 0.18),
                "metadata": metadata,
            },
        )
    if team_model is not None:
        save_model(
            TEAM_MODEL_PATH,
            {
                "model": team_model,
                "calibrator": team_calibrator,
                "model_kind": team_metrics.get("model_kind", "residual"),
                "model_blend_weight": team_metrics.get("model_blend_weight", 1.0),
                "probability_temperature": team_metrics.get(
                    "probability_temperature",
                    1.0,
                ),
                "max_model_delta": team_metrics.get(
                    "max_model_delta",
                    PROBABILITY_MAX_MODEL_DELTA,
                ),
                "selected_candidate": team_metrics.get("selected_candidate", "hgb_residual"),
                "recommended_candidate": team_metrics.get("recommended_candidate", ""),
                "selection_mode": team_metrics.get("selection_mode", "auto"),
                "features": TEAM_FEATURES,
                "sequential_state": sequential_state,
                "metadata": metadata,
            },
        )
    if map_model_payload is not None:
        save_model(
            MAP_MODEL_PATH,
            {
                **map_model_payload,
                "sequential_state": sequential_state,
                "metadata": metadata,
            },
        )

    ensure_model_dir()
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    save_model_selection(selections)
    from ..storage import record_model_run

    record_model_run(metadata)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Valorant player-rating and team-win models.")
    parser.add_argument("--matches-csv", default=MATCHES_CSV)
    parser.add_argument("--season-year", type=int, default=datetime.utcnow().year)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--fallback-years", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--min-player-history-maps", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--min-team-matches", type=int, default=20)
    parser.add_argument("--refresh-matches", action="store_true")
    parser.add_argument("--limit-per-team", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Retrain even when the match dataset is unchanged.")
    args = parser.parse_args()

    if args.refresh_matches:
        scrape_matches(
            output_csv=args.matches_csv,
            limit_per_team=args.limit_per_team or None,
            team_pages=registry_team_pages(),
            season_year=args.season_year,
            min_matches_per_team=args.min_team_matches,
        )

    metrics = train_models(
        matches_csv=args.matches_csv,
        season_year=args.season_year,
        recent_days=args.recent_days,
        fallback_years=None,
        min_player_history_maps=1,
        min_team_matches=args.min_team_matches,
        force=args.force,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
