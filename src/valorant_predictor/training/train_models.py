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

try:
    from xgboost import XGBClassifier
except ImportError:  # The core app remains usable without the optional challenger.
    XGBClassifier = None

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
    rebase_map_probability,
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
from ..prediction.simulation import equivalent_map_probability, series_probability_from_maps


MODEL_DIR = MODELS_DIR
METRICS_PATH = TRAINING_METRICS_PATH
MODEL_VERSION = "logic-v18-guarded-oof-ensemble"
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

BASE_TEAM_FEATURES = [
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

PLAYER_STACK_FEATURES = [
    "player_projection_diff",
    "player_projection_floor_diff",
    "player_projection_ceiling_diff",
    "player_projection_correction_diff",
    "player_projection_reliability",
    "player_projection_oof_coverage",
]

REGION_POOL_LABELS = (
    "VCT Americas",
    "VCT EMEA",
    "VCT Pacific",
    "VCT China",
)
REGION_POOL_FEATURES = [
    f"region_pool::{region.lower().replace(' ', '_')}"
    for region in REGION_POOL_LABELS
]

LINEUP_COMPONENT_FEATURES = [
    "lineup_rating_diff",
    "roster_continuity_diff",
    "player_projection_diff",
    "player_projection_floor_diff",
    "player_projection_ceiling_diff",
    "player_projection_correction_diff",
]

TEAM_FEATURES = [
    *BASE_TEAM_FEATURES,
    *REGION_POOL_FEATURES,
    *PLAYER_STACK_FEATURES,
]

MAP_CONTEXT_FEATURES = [
    "map_number",
    "map_team_anchor_probability",
    "map_team_anchor_oof_available",
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
    winning_score = pd.concat([team_scores, opponent_scores], axis=1).max(axis=1)
    output["best_of"] = np.select(
        [winning_score >= 3, winning_score >= 2],
        [5, 3],
        default=1,
    ).astype(int)
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
                    "best_of": int(team_a_row.get("best_of", 3) or 3),
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
    team_regions = output.get("team_region", pd.Series("", index=output.index)).fillna("").astype(str)
    opponent_regions = output.get(
        "opponent_region",
        pd.Series("", index=output.index),
    ).fillna("").astype(str)
    for region, feature in zip(REGION_POOL_LABELS, REGION_POOL_FEATURES):
        output[feature] = (
            team_regions.eq(region).astype(float)
            - opponent_regions.eq(region).astype(float)
        )
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
        "best_of",
        "elo_probability",
        "team_oof_series_probability",
        "team_oof_model_available",
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
        elo_probability = float(row.get("elo_probability", 0.5) or 0.5)
        probabilities = row.get("map_probabilities")
        if isinstance(probabilities, dict):
            value = probabilities.get(row.get("map_name"))
            if value is not None:
                return rebase_map_probability(
                    float(value),
                    elo_probability,
                    float(row["map_team_anchor_probability"]),
                )
        return float(row["map_team_anchor_probability"])

    def map_context_value(row, column: str, default: float) -> float:
        values = row.get(column)
        if isinstance(values, dict):
            value = values.get(row.get("map_name"))
            if value is not None:
                return float(value)
        return float(default)

    elo_series_probability = pd.to_numeric(
        output.get("elo_probability", pd.Series(0.5, index=output.index)),
        errors="coerce",
    ).fillna(0.5).clip(0.05, 0.95)
    oof_series_probability = pd.to_numeric(
        output.get(
            "team_oof_series_probability",
            pd.Series(np.nan, index=output.index),
        ),
        errors="coerce",
    )
    output["map_team_anchor_oof_available"] = pd.to_numeric(
        output.get(
            "team_oof_model_available",
            pd.Series(0.0, index=output.index),
        ),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    series_anchor = oof_series_probability.where(
        output["map_team_anchor_oof_available"].gt(0.0),
        elo_series_probability,
    ).fillna(elo_series_probability).clip(0.02, 0.98)
    best_of_values = pd.to_numeric(
        output.get("best_of", pd.Series(3, index=output.index)),
        errors="coerce",
    ).fillna(3).astype(int)
    best_of_values = best_of_values.where(best_of_values.isin([1, 3, 5]), 3)
    output["map_team_anchor_probability"] = [
        equivalent_map_probability(probability, int(best_of))
        for probability, best_of in zip(series_anchor, best_of_values)
    ]
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


def expanding_oof_splits(
    training_df: pd.DataFrame,
    folds: int = 5,
    minimum_history_fraction: float = 0.25,
    calibration_fraction: float = 0.15,
) -> list[tuple[pd.Series, pd.Series, pd.Series]]:
    groups = training_df[
        ["match_key", "target_date", "target_match_id"]
    ].drop_duplicates(subset=["match_key"])
    groups = groups.copy()
    groups["target_date"] = pd.to_datetime(
        groups["target_date"],
        utc=True,
        errors="coerce",
    )
    groups = groups.sort_values(
        ["target_date", "target_match_id"],
        na_position="first",
    ).reset_index(drop=True)
    total = len(groups)
    if total < 12:
        return []

    first_test = max(4, int(math.ceil(total * minimum_history_fraction)))
    remaining = total - first_test
    if remaining < 1:
        return []
    block_size = max(1, int(math.ceil(remaining / max(1, folds))))
    splits = []
    for test_start in range(first_test, total, block_size):
        test_end = min(total, test_start + block_size)
        prior = groups.iloc[:test_start]
        calibration_size = max(
            1,
            int(math.ceil(len(prior) * calibration_fraction)),
        )
        train_end = len(prior) - calibration_size
        if train_end < 2:
            continue
        train_keys = set(prior.iloc[:train_end]["match_key"])
        calibration_keys = set(prior.iloc[train_end:]["match_key"])
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
        calibration_baseline_mae = float(
            mean_absolute_error(calibration_y, calibration_baseline)
        )
        for candidate_name in PLAYER_CANDIDATES:
            candidate_model = _new_player_candidate(candidate_name)
            candidate_model.fit(
                x[train_mask],
                residual_target[train_mask],
                sample_weight=weights[train_mask],
            )
            residual_predictions = candidate_model.predict(x[calibration_mask])
            best_weight = 0.0
            best_mae = calibration_baseline_mae
            best_predictions = calibration_baseline
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
                    best_predictions = blended
            candidate_scores[candidate_name] = {
                "mae": best_mae,
                "r2": float(r2_score(calibration_y, best_predictions)),
                "validation_baseline_mae": calibration_baseline_mae,
                "validation_baseline_r2": float(
                    r2_score(calibration_y, calibration_baseline)
                ),
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
                    "rolling_mae_std": float(np.std(rolling_maes))
                    if rolling_maes
                    else None,
                    "rolling_baseline_mae_mean": float(
                        np.mean(rolling_baseline_maes)
                    )
                    if rolling_baseline_maes
                    else None,
                    "rolling_baseline_mae_std": float(
                        np.std(rolling_baseline_maes)
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
        y_test = y[test_mask]
        for candidate_name, candidate_model in candidate_models.items():
            candidate_predictions = np.clip(
                baseline_preds.to_numpy(dtype=float)
                + candidate_scores[candidate_name]["correction_weight"]
                * candidate_model.predict(x[test_mask]),
                0.35,
                1.90,
            )
            candidate_scores[candidate_name].update(
                {
                    "test_mae": float(
                        mean_absolute_error(y_test, candidate_predictions)
                    ),
                    "test_r2": float(r2_score(y_test, candidate_predictions)),
                }
            )
        preds = (
            baseline_preds
            if selected_candidate == "last_10_baseline"
            else (
                baseline_preds
                + correction_weight * evaluation_model.predict(x[test_mask])
            ).clip(0.35, 1.90)
        )
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
            "baseline_r2": float(r2_score(y_test, baseline_preds)),
            "selected_candidate": selected_candidate,
            "recommended_candidate": recommended_candidate,
            "selection_mode": selection_mode,
            "candidate_calibration": candidate_scores,
            "validation_baseline_mae": calibration_baseline_mae,
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
        metrics["rolling_mae_std"] = rolling_metrics.get("rolling_mae_std")
        metrics["rolling_baseline_mae_mean"] = rolling_metrics.get(
            "rolling_baseline_mae_mean"
        )
        metrics["rolling_baseline_mae_std"] = rolling_metrics.get(
            "rolling_baseline_mae_std"
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


def generate_player_oof_predictions(
    training_df: pd.DataFrame,
    requested_candidate: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    output = training_df.copy()
    if output.empty:
        return output, {"status": "skipped", "reason": "No player rows."}

    x = output[PLAYER_FEATURES].fillna(0.0)
    y = output["target_rating"].astype(float)
    baseline = output["player_shrunk_rating"].astype(float).clip(0.35, 1.90)
    residual_target = y - baseline
    weights = output.get(
        "training_weight",
        pd.Series(1.0, index=output.index),
    ).astype(float)
    output["player_oof_prediction"] = baseline
    output["player_oof_correction"] = 0.0
    output["player_oof_model_available"] = 0.0

    if requested_candidate == "last_10_baseline":
        return output, {
            "status": "baseline",
            "strategy": "historical_shrunk_rating_only",
            "coverage": 0.0,
            "folds": 0,
        }

    requested_models = (
        PLAYER_CANDIDATES
        if requested_candidate == "auto"
        else (requested_candidate,)
    )
    requested_models = tuple(
        name for name in requested_models if name in PLAYER_CANDIDATES
    )
    fold_records = []
    for fold_number, (train_mask, calibration_mask, test_mask) in enumerate(
        expanding_oof_splits(output),
        start=1,
    ):
        if not train_mask.any() or not calibration_mask.any() or not test_mask.any():
            continue
        calibration_y = y.loc[calibration_mask].to_numpy(dtype=float)
        calibration_baseline = baseline.loc[calibration_mask].to_numpy(dtype=float)
        best = None
        for candidate_name in requested_models:
            candidate_model = _new_player_candidate(candidate_name)
            candidate_model.fit(
                x.loc[train_mask],
                residual_target.loc[train_mask],
                sample_weight=weights.loc[train_mask],
            )
            calibration_residual = candidate_model.predict(x.loc[calibration_mask])
            candidate_weight = 0.0
            candidate_mae = float(
                mean_absolute_error(calibration_y, calibration_baseline)
            )
            for blend_weight in np.linspace(0.0, 1.0, 21):
                predictions = np.clip(
                    calibration_baseline
                    + float(blend_weight) * calibration_residual,
                    0.35,
                    1.90,
                )
                score = float(mean_absolute_error(calibration_y, predictions))
                if score < candidate_mae:
                    candidate_mae = score
                    candidate_weight = float(blend_weight)
            if best is None or candidate_mae < best["calibration_mae"]:
                best = {
                    "candidate": candidate_name,
                    "model": candidate_model,
                    "weight": candidate_weight,
                    "calibration_mae": candidate_mae,
                }
        if best is None:
            continue

        test_baseline = baseline.loc[test_mask].to_numpy(dtype=float)
        predictions = np.clip(
            test_baseline
            + best["weight"] * best["model"].predict(x.loc[test_mask]),
            0.35,
            1.90,
        )
        output.loc[test_mask, "player_oof_prediction"] = predictions
        output.loc[test_mask, "player_oof_correction"] = predictions - test_baseline
        output.loc[test_mask, "player_oof_model_available"] = 1.0
        fold_records.append(
            {
                "fold": fold_number,
                "candidate": best["candidate"],
                "correction_weight": best["weight"],
                "calibration_mae": best["calibration_mae"],
                "test_matches": int(output.loc[test_mask, "match_key"].nunique()),
                "test_rows": int(test_mask.sum()),
            }
        )

    available = output["player_oof_model_available"].eq(1.0)
    if available.any():
        predictions = output.loc[available, "player_oof_prediction"].astype(float)
        targets = y.loc[available]
        baseline_predictions = baseline.loc[available]
        oof_mae = float(mean_absolute_error(targets, predictions))
        baseline_mae = float(mean_absolute_error(targets, baseline_predictions))
    else:
        oof_mae = None
        baseline_mae = None
    return output, {
        "status": "generated" if fold_records else "skipped",
        "strategy": "expanding_window_nested_player_oof",
        "strictly_pre_match": True,
        "folds": len(fold_records),
        "rows": int(available.sum()),
        "matches": int(output.loc[available, "match_key"].nunique()),
        "coverage": float(available.mean()),
        "mae": oof_mae,
        "baseline_mae": baseline_mae,
        "skill_vs_baseline": (
            1.0 - oof_mae / baseline_mae
            if oof_mae is not None and baseline_mae
            else 0.0
        ),
        "fold_details": fold_records,
    }


def player_stack_features_from_summaries(
    team_summary: dict,
    opponent_summary: dict,
) -> dict:
    def value(summary: dict, primary: str, fallback: str, default: float) -> float:
        return float(summary.get(primary, summary.get(fallback, default)) or default)

    return {
        "player_projection_diff": value(
            team_summary, "projection_mean", "avg_player_rating", 1.0
        )
        - value(opponent_summary, "projection_mean", "avg_player_rating", 1.0),
        "player_projection_floor_diff": value(
            team_summary, "projection_floor", "bottom_player_rating", 1.0
        )
        - value(opponent_summary, "projection_floor", "bottom_player_rating", 1.0),
        "player_projection_ceiling_diff": value(
            team_summary, "projection_ceiling", "top_player_rating", 1.0
        )
        - value(opponent_summary, "projection_ceiling", "top_player_rating", 1.0),
        "player_projection_correction_diff": value(
            team_summary,
            "projection_correction",
            "avg_trained_rating_correction",
            0.0,
        )
        - value(
            opponent_summary,
            "projection_correction",
            "avg_trained_rating_correction",
            0.0,
        ),
        "player_projection_reliability": min(
            value(team_summary, "projection_reliability", "lineup_reliability", 0.0),
            value(
                opponent_summary,
                "projection_reliability",
                "lineup_reliability",
                0.0,
            ),
        ),
        "player_projection_oof_coverage": min(
            value(team_summary, "projection_oof_coverage", "player_model_coverage", 0.0),
            value(
                opponent_summary,
                "projection_oof_coverage",
                "player_model_coverage",
                0.0,
            ),
        ),
    }


def attach_player_oof_team_features(
    team_training: pd.DataFrame,
    player_oof: pd.DataFrame,
) -> pd.DataFrame:
    output = team_training.copy()
    sequential_state = team_training.attrs.get("sequential_state")
    for feature in PLAYER_STACK_FEATURES:
        output[feature] = 0.0
    if output.empty or player_oof.empty or "player_oof_prediction" not in player_oof:
        output.attrs["sequential_state"] = sequential_state
        output.attrs["player_stack_oof_coverage"] = 0.0
        return output

    player_level = (
        player_oof.groupby(["match_key", "team", "player"], sort=False)
        .agg(
            projection=("player_oof_prediction", "mean"),
            baseline=("player_shrunk_rating", "mean"),
            effective_maps=("player_effective_maps", "mean"),
            model_available=("player_oof_model_available", "mean"),
            maps_in_match=("target_rating", "size"),
        )
        .reset_index()
    )
    player_level["evidence"] = (
        player_level["effective_maps"]
        / (player_level["effective_maps"] + 8.0)
    ).clip(0.0, 1.0)
    player_level = player_level.sort_values(
        ["match_key", "team", "maps_in_match", "evidence", "projection"],
        ascending=[True, True, False, False, False],
    )
    lineup = player_level.groupby(["match_key", "team"], sort=False).head(5)
    summaries = (
        lineup.groupby(["match_key", "team"], sort=False)
        .agg(
            projection_mean=("projection", "mean"),
            projection_floor=("projection", "min"),
            projection_ceiling=("projection", "max"),
            projection_correction=(
                "projection",
                lambda values: float(
                    values.mean()
                    - lineup.loc[values.index, "baseline"].mean()
                ),
            ),
            projection_reliability=("evidence", "mean"),
            projection_oof_coverage=("model_available", "mean"),
            projected_players=("player", "nunique"),
        )
        .reset_index()
    )
    lineup_fraction = (summaries["projected_players"] / 5.0).clip(0.0, 1.0)
    summaries["projection_reliability"] *= lineup_fraction
    summaries["projection_oof_coverage"] *= lineup_fraction

    own = summaries.rename(
        columns={
            column: f"own_{column}"
            for column in summaries.columns
            if column not in {"match_key", "team"}
        }
    )
    opponent = summaries.rename(
        columns={
            "team": "opponent",
            **{
                column: f"opponent_{column}"
                for column in summaries.columns
                if column not in {"match_key", "team"}
            },
        }
    )
    output = output.drop(columns=PLAYER_STACK_FEATURES).merge(
        own,
        on=["match_key", "team"],
        how="left",
    ).merge(
        opponent,
        on=["match_key", "opponent"],
        how="left",
    )

    neutral_defaults = {
        "projection_mean": 1.0,
        "projection_floor": 1.0,
        "projection_ceiling": 1.0,
        "projection_correction": 0.0,
        "projection_reliability": 0.0,
        "projection_oof_coverage": 0.0,
    }
    for column, default in neutral_defaults.items():
        own_column = f"own_{column}"
        opponent_column = f"opponent_{column}"
        output[own_column] = pd.to_numeric(
            output.get(own_column),
            errors="coerce",
        ).fillna(default)
        output[opponent_column] = pd.to_numeric(
            output.get(opponent_column),
            errors="coerce",
        ).fillna(default)

    output["player_projection_diff"] = (
        output["own_projection_mean"] - output["opponent_projection_mean"]
    )
    output["player_projection_floor_diff"] = (
        output["own_projection_floor"] - output["opponent_projection_floor"]
    )
    output["player_projection_ceiling_diff"] = (
        output["own_projection_ceiling"] - output["opponent_projection_ceiling"]
    )
    output["player_projection_correction_diff"] = (
        output["own_projection_correction"]
        - output["opponent_projection_correction"]
    )
    output["player_projection_reliability"] = output[
        ["own_projection_reliability", "opponent_projection_reliability"]
    ].min(axis=1)
    output["player_projection_oof_coverage"] = output[
        ["own_projection_oof_coverage", "opponent_projection_oof_coverage"]
    ].min(axis=1)
    helper_columns = [
        column
        for column in output.columns
        if column.startswith("own_") or column.startswith("opponent_projection")
    ]
    output = output.drop(columns=helper_columns)
    output.attrs["sequential_state"] = sequential_state
    output.attrs["player_stack_oof_coverage"] = float(
        output["player_projection_oof_coverage"].mean()
    )
    return output


def _new_team_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=20,
        min_samples_leaf=24,
        l2_regularization=2.0,
        random_state=7,
    )


BASE_PROBABILITY_CANDIDATES = (
    "hgb_residual",
    "hgb_classifier",
    "logistic_classifier",
)
PROBABILITY_CANDIDATES = (
    *BASE_PROBABILITY_CANDIDATES,
    *(("xgboost_classifier",) if XGBClassifier is not None else ()),
)

PROBABILITY_BLEND_WEIGHTS = tuple(np.linspace(0.0, 1.0, 21))
PROBABILITY_TEMPERATURES = (0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)
PROBABILITY_MAX_MODEL_DELTA = 0.20
LATEST_WINDOW_SELECTION_WEIGHT = 0.65
COMPLEXITY_MIN_LOG_LOSS_IMPROVEMENT = 0.005
TEAM_PROBABILITY_MAX_MODEL_DELTA = 0.48
TEAM_LOGISTIC_REGULARIZATION = 0.10
MAP_MODEL_MAX_CORRECTION = 0.10

TEAM_NEGATED_FEATURES = {
    "team_rating_diff",
    "team_win_rate_diff",
    "team_score_margin_diff",
    "team_consistency_diff",
    "team_maps_diff",
    "h2h_rating_diff",
    "elo_diff",
    "team_elo_diff",
    "strength_of_schedule_diff",
    "region_elo_diff",
    "map_pool_win_rate_diff",
    "map_pool_rating_diff",
    "map_pool_elo_diff",
    "roster_continuity_diff",
    "lineup_rating_diff",
    "player_projection_diff",
    "player_projection_floor_diff",
    "player_projection_ceiling_diff",
    "player_projection_correction_diff",
    *REGION_POOL_FEATURES,
}


def reverse_team_feature_row(features: dict) -> dict:
    reversed_features = dict(features)
    for feature in TEAM_NEGATED_FEATURES:
        if feature in reversed_features:
            reversed_features[feature] = -float(
                reversed_features.get(feature, 0.0) or 0.0
            )
    if "h2h_win_rate" in reversed_features:
        h2h_win_rate = float(reversed_features.get("h2h_win_rate", 0.5))
        reversed_features["h2h_win_rate"] = 1.0 - (
            h2h_win_rate if np.isfinite(h2h_win_rate) else 0.5
        )
    return reversed_features


def symmetrize_grouped_probabilities(
    probabilities,
    frame: pd.DataFrame,
) -> np.ndarray:
    output = np.asarray(probabilities, dtype=float).copy()
    if frame.empty or "match_key" not in frame.columns:
        return output
    positions_by_match = {}
    for position, match_key in enumerate(frame["match_key"].tolist()):
        positions_by_match.setdefault(match_key, []).append(position)
    for positions in positions_by_match.values():
        if len(positions) != 2:
            continue
        first, second = positions
        forward = 0.5 * (output[first] + 1.0 - output[second])
        output[first] = forward
        output[second] = 1.0 - forward
    return np.clip(output, 0.02, 0.98)


def _new_probability_candidate(
    name: str,
    min_samples_leaf: int = 24,
    logistic_c: float = 0.25,
):
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
            LogisticRegression(C=logistic_c, max_iter=1200, random_state=7),
        ), "classifier"
    if name == "xgboost_classifier" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=160,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.20,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=7,
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
    max_model_delta: float = PROBABILITY_MAX_MODEL_DELTA,
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
                max_model_delta=max_model_delta,
            )
            score = float(log_loss(targets, probabilities, labels=[0, 1]))
            if best is None or score < best["log_loss"]:
                best = {
                    "log_loss": score,
                    "brier_score": float(brier_score_loss(targets, probabilities)),
                    "accuracy": float(accuracy_score(targets, probabilities >= 0.5)),
                    "model_blend_weight": float(blend_weight),
                    "probability_temperature": float(temperature),
                    "max_model_delta": float(max_model_delta),
                }
    return best


def _logit_array(probabilities) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 0.001, 0.999)
    return np.log(values / (1.0 - values))


def _calibration_design(probabilities, method: str) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 0.001, 0.999)
    if method == "symmetric_platt":
        return _logit_array(values).reshape(-1, 1)
    if method == "symmetric_beta":
        return np.column_stack((np.log(values), -np.log1p(-values)))
    raise ValueError(f"Unknown calibration method: {method}")


def _fit_serialized_probability_calibrator(
    method: str,
    probabilities,
    targets,
    weights=None,
) -> dict:
    design = _calibration_design(probabilities, method)
    model = LogisticRegression(C=0.5, max_iter=1200, random_state=7)
    model.fit(design, np.asarray(targets, dtype=float), sample_weight=weights)
    return {
        "method": method,
        "coefficients": [float(value) for value in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
    }


def apply_probability_calibration(
    settings: dict | None,
    model_probabilities,
    baseline_probabilities=None,
) -> np.ndarray:
    raw = np.asarray(model_probabilities, dtype=float)
    baseline = (
        np.full(len(raw), 0.5, dtype=float)
        if baseline_probabilities is None
        else np.asarray(baseline_probabilities, dtype=float)
    )
    settings = settings or {"method": "none"}
    method = settings.get("method", "none")
    max_delta = float(settings.get("max_model_delta", 0.48))
    clipped = np.clip(raw, baseline - max_delta, baseline + max_delta)
    clipped = np.clip(clipped, 0.02, 0.98)
    if method == "none":
        return clipped
    if method == "neutral_shrink_temperature":
        return apply_probability_blend_temperature(
            clipped,
            baseline,
            blend_weight=float(settings.get("model_blend_weight", 1.0)),
            temperature=float(settings.get("probability_temperature", 1.0)),
            max_model_delta=max_delta,
        )

    coefficients = np.asarray(settings.get("coefficients", []), dtype=float)
    intercept = float(settings.get("intercept", 0.0))

    def calibrated(values: np.ndarray) -> np.ndarray:
        design = _calibration_design(values, method)
        logits = design @ coefficients + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    forward = calibrated(clipped)
    reverse = 1.0 - calibrated(1.0 - clipped)
    return np.clip(0.5 * (forward + reverse), 0.02, 0.98)


def select_probability_calibration(
    model_probabilities,
    baseline_probabilities,
    targets,
    frame: pd.DataFrame,
    weights=None,
    max_model_delta: float = TEAM_PROBABILITY_MAX_MODEL_DELTA,
) -> tuple[dict, dict]:
    raw = np.asarray(model_probabilities, dtype=float)
    baseline = np.asarray(baseline_probabilities, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    weight_values = (
        np.ones(len(raw), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if len(raw) < 8 or len(np.unique(target_values)) < 2:
        settings = {
            "method": "neutral_shrink_temperature",
            **optimize_probability_blend_temperature(
                raw,
                baseline,
                target_values,
                max_model_delta=max_model_delta,
            ),
        }
        return settings, {"neutral_shrink_temperature": settings}

    fit_mask, selection_mask = grouped_chronological_split(
        frame.reset_index(drop=True),
        validation_fraction=0.50,
    )
    fit_positions = np.flatnonzero(fit_mask.to_numpy())
    selection_positions = np.flatnonzero(selection_mask.to_numpy())
    if (
        len(fit_positions) < 4
        or len(selection_positions) < 4
        or len(np.unique(target_values[fit_positions])) < 2
    ):
        fit_positions = np.arange(len(raw))
        selection_positions = np.arange(len(raw))

    candidates = {}
    temperature_fit = optimize_probability_blend_temperature(
        raw[fit_positions],
        baseline[fit_positions],
        target_values[fit_positions],
        max_model_delta=max_model_delta,
    )
    temperature_settings = {
        "method": "neutral_shrink_temperature",
        **temperature_fit,
    }
    temperature_selection = apply_probability_calibration(
        temperature_settings,
        raw[selection_positions],
        baseline[selection_positions],
    )
    candidates["neutral_shrink_temperature"] = {
        **temperature_settings,
        "selection_log_loss": float(
            log_loss(
                target_values[selection_positions],
                temperature_selection,
                labels=[0, 1],
            )
        ),
        "selection_brier_score": float(
            brier_score_loss(
                target_values[selection_positions],
                temperature_selection,
            )
        ),
    }

    for method in ("symmetric_platt", "symmetric_beta"):
        fitted = _fit_serialized_probability_calibrator(
            method,
            raw[fit_positions],
            target_values[fit_positions],
            weight_values[fit_positions],
        )
        fitted["max_model_delta"] = float(max_model_delta)
        selection_probabilities = apply_probability_calibration(
            fitted,
            raw[selection_positions],
            baseline[selection_positions],
        )
        candidates[method] = {
            **fitted,
            "selection_log_loss": float(
                log_loss(
                    target_values[selection_positions],
                    selection_probabilities,
                    labels=[0, 1],
                )
            ),
            "selection_brier_score": float(
                brier_score_loss(
                    target_values[selection_positions],
                    selection_probabilities,
                )
            ),
        }

    selected_method = min(
        candidates,
        key=lambda name: (
            candidates[name]["selection_log_loss"],
            candidates[name]["selection_brier_score"],
        ),
    )
    neutral_score = candidates["neutral_shrink_temperature"][
        "selection_log_loss"
    ]
    if (
        selected_method != "neutral_shrink_temperature"
        and candidates[selected_method]["selection_log_loss"] + 0.002
        >= neutral_score
    ):
        candidates[selected_method]["minimum_improvement_rejected"] = True
        selected_method = "neutral_shrink_temperature"
    if selected_method == "neutral_shrink_temperature":
        final = {
            "method": selected_method,
            **optimize_probability_blend_temperature(
                raw,
                baseline,
                target_values,
                max_model_delta=max_model_delta,
            ),
        }
    else:
        final = _fit_serialized_probability_calibrator(
            selected_method,
            raw,
            target_values,
            weight_values,
        )
        final["max_model_delta"] = float(max_model_delta)
        final["model_blend_weight"] = 1.0
        final["probability_temperature"] = 1.0
    return final, candidates


def _unique_match_probability_rows(
    frame: pd.DataFrame,
    targets,
    probabilities,
) -> pd.DataFrame:
    scoped = pd.DataFrame(
        {
            "match_key": frame["match_key"].astype(str).to_numpy(),
            "target": np.asarray(targets, dtype=float),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    return scoped.drop_duplicates(subset=["match_key"], keep="first").reset_index(drop=True)


def bootstrap_probability_intervals(
    frame: pd.DataFrame,
    targets,
    probabilities,
    samples: int = 1000,
    seed: int = 7,
) -> dict:
    scoped = _unique_match_probability_rows(frame, targets, probabilities)
    if scoped.empty:
        return {}
    target_values = scoped["target"].to_numpy(dtype=float)
    probability_values = scoped["probability"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = {"accuracy": [], "log_loss": [], "brier_score": []}
    for _ in range(samples):
        positions = rng.integers(0, len(scoped), len(scoped))
        sampled_y = target_values[positions]
        sampled_p = probability_values[positions]
        estimates["accuracy"].append(
            float(accuracy_score(sampled_y, sampled_p >= 0.5))
        )
        estimates["log_loss"].append(
            float(log_loss(sampled_y, sampled_p, labels=[0, 1]))
        )
        estimates["brier_score"].append(
            float(brier_score_loss(sampled_y, sampled_p))
        )
    return {
        metric: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for metric, values in estimates.items()
    }


def selective_accuracy_report(
    frame: pd.DataFrame,
    targets,
    probabilities,
    thresholds: tuple[float, ...] = (0.55, 0.60, 0.65),
) -> list[dict]:
    scoped = _unique_match_probability_rows(frame, targets, probabilities)
    rows = []
    for threshold in thresholds:
        selected = scoped[
            np.maximum(scoped["probability"], 1.0 - scoped["probability"])
            >= threshold
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "matches": int(len(selected)),
                "coverage": float(len(selected) / len(scoped)) if len(scoped) else 0.0,
                "accuracy": float(
                    accuracy_score(
                        selected["target"],
                        selected["probability"] >= 0.5,
                    )
                )
                if not selected.empty
                else None,
            }
        )
    return rows


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
    reference_baseline: pd.Series | None = None,
    enforce_team_symmetry: bool = False,
    logistic_c: float = 0.25,
    max_model_delta: float = PROBABILITY_MAX_MODEL_DELTA,
    candidate_names: tuple[str, ...] | None = None,
) -> tuple[str, str, object, dict]:
    candidate_names = candidate_names or PROBABILITY_CANDIDATES
    scores = {}
    models = {}
    kinds = {}
    validation_y = y[calibration_mask]
    validation_baseline = (
        reference_baseline[calibration_mask]
        if reference_baseline is not None
        else baseline[calibration_mask]
    ).to_numpy(dtype=float)
    validation_baseline_log_loss = float(
        log_loss(validation_y, validation_baseline, labels=[0, 1])
    )
    validation_baseline_brier = float(
        brier_score_loss(validation_y, validation_baseline)
    )
    validation_baseline_accuracy = float(
        accuracy_score(validation_y, validation_baseline >= 0.5)
    )
    for candidate_name in candidate_names:
        model, kind = _new_probability_candidate(
            candidate_name,
            min_samples_leaf,
            logistic_c=logistic_c,
        )
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
        if enforce_team_symmetry and training_df is not None:
            probabilities = symmetrize_grouped_probabilities(
                probabilities,
                training_df.loc[calibration_mask],
            )
        calibration_y = y[calibration_mask]
        if map_reliabilities is None:
            raw_evaluation = probabilities
            scores[candidate_name] = optimize_probability_blend_temperature(
                probabilities,
                baseline[calibration_mask],
                calibration_y,
                max_model_delta=max_model_delta,
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
        scores[candidate_name].update(
            {
                "validation_baseline_log_loss": validation_baseline_log_loss,
                "validation_baseline_brier": validation_baseline_brier,
                "validation_baseline_accuracy": validation_baseline_accuracy,
            }
        )
        models[candidate_name] = model
        kinds[candidate_name] = kind

    if training_df is not None:
        rolling_splits = rolling_origin_splits(training_df)
        for candidate_name in candidate_names:
            rolling_log_losses = []
            rolling_brier_scores = []
            rolling_accuracies = []
            rolling_reference_log_losses = []
            rolling_reference_briers = []
            rolling_baseline_log_losses = []
            rolling_baseline_briers = []
            rolling_baseline_accuracies = []
            rolling_wins = []
            for rolling_train, rolling_calibration, rolling_test in rolling_splits:
                rolling_model, rolling_kind = _new_probability_candidate(
                    candidate_name,
                    min_samples_leaf,
                    logistic_c=logistic_c,
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
                if enforce_team_symmetry:
                    calibration_probabilities = symmetrize_grouped_probabilities(
                        calibration_probabilities,
                        training_df.loc[rolling_calibration],
                    )
                if map_reliabilities is None:
                    settings = optimize_probability_blend_temperature(
                        calibration_probabilities,
                        baseline[rolling_calibration],
                        y[rolling_calibration],
                        max_model_delta=max_model_delta,
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
                if enforce_team_symmetry:
                    raw_probabilities = symmetrize_grouped_probabilities(
                        raw_probabilities,
                        training_df.loc[rolling_test],
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
                reference_values = (
                    reference_baseline[rolling_test]
                    if reference_baseline is not None
                    else baseline[rolling_test]
                )
                baseline_probabilities = reference_values.to_numpy(dtype=float)
                coin_probabilities = np.full(len(test_y), 0.5)
                candidate_log_loss = float(
                    log_loss(test_y, probabilities, labels=[0, 1])
                )
                candidate_brier = float(brier_score_loss(test_y, probabilities))
                baseline_log_loss = float(
                    log_loss(test_y, baseline_probabilities, labels=[0, 1])
                )
                baseline_brier = float(
                    brier_score_loss(test_y, baseline_probabilities)
                )
                baseline_accuracy = float(
                    accuracy_score(test_y, baseline_probabilities >= 0.5)
                )
                reference_log_loss = min(
                    baseline_log_loss,
                    float(log_loss(test_y, coin_probabilities, labels=[0, 1])),
                )
                reference_brier = min(
                    baseline_brier,
                    0.25,
                )
                rolling_log_losses.append(candidate_log_loss)
                rolling_brier_scores.append(candidate_brier)
                rolling_accuracies.append(
                    float(accuracy_score(test_y, probabilities >= 0.5))
                )
                rolling_reference_log_losses.append(reference_log_loss)
                rolling_reference_briers.append(reference_brier)
                rolling_baseline_log_losses.append(baseline_log_loss)
                rolling_baseline_briers.append(baseline_brier)
                rolling_baseline_accuracies.append(baseline_accuracy)
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
                    "rolling_log_loss_std": float(np.std(rolling_log_losses))
                    if rolling_log_losses
                    else None,
                    "rolling_brier_mean": float(np.mean(rolling_brier_scores))
                    if rolling_brier_scores
                    else None,
                    "rolling_brier_std": float(np.std(rolling_brier_scores))
                    if rolling_brier_scores
                    else None,
                    "rolling_accuracy_mean": float(np.mean(rolling_accuracies))
                    if rolling_accuracies
                    else None,
                    "rolling_accuracy_std": float(np.std(rolling_accuracies))
                    if rolling_accuracies
                    else None,
                    "rolling_baseline_log_loss_mean": float(
                        np.mean(rolling_baseline_log_losses)
                    )
                    if rolling_baseline_log_losses
                    else None,
                    "rolling_baseline_log_loss_std": float(
                        np.std(rolling_baseline_log_losses)
                    )
                    if rolling_baseline_log_losses
                    else None,
                    "rolling_baseline_brier_mean": float(
                        np.mean(rolling_baseline_briers)
                    )
                    if rolling_baseline_briers
                    else None,
                    "rolling_baseline_brier_std": float(
                        np.std(rolling_baseline_briers)
                    )
                    if rolling_baseline_briers
                    else None,
                    "rolling_baseline_accuracy_mean": float(
                        np.mean(rolling_baseline_accuracies)
                    )
                    if rolling_baseline_accuracies
                    else None,
                    "rolling_baseline_accuracy_std": float(
                        np.std(rolling_baseline_accuracies)
                    )
                    if rolling_baseline_accuracies
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

    for candidate_name, candidate_score in scores.items():
        rolling_score = candidate_score.get("rolling_log_loss_mean")
        candidate_score["selection_log_loss"] = float(
            candidate_score["log_loss"]
            if rolling_score is None
            else LATEST_WINDOW_SELECTION_WEIGHT * candidate_score["log_loss"]
            + (1.0 - LATEST_WINDOW_SELECTION_WEIGHT) * rolling_score
        )
        candidate_score["selection_strategy"] = (
            "latest_calibration_plus_rolling_history"
        )

    selected = min(
        scores,
        key=lambda name: scores[name]["selection_log_loss"],
    )
    if (
        selected != "logistic_classifier"
        and "logistic_classifier" in scores
        and scores[selected]["selection_log_loss"]
        + COMPLEXITY_MIN_LOG_LOSS_IMPROVEMENT
        >= scores["logistic_classifier"]["selection_log_loss"]
    ):
        scores[selected]["complexity_margin_rejected"] = True
        scores[selected]["raw_recommended_candidate"] = selected
        selected = "logistic_classifier"
    return selected, kinds[selected], models[selected], scores


def evaluate_probability_candidate_holdouts(
    x: pd.DataFrame,
    y: pd.Series,
    baseline: pd.Series,
    weights: pd.Series,
    training_df: pd.DataFrame,
    train_mask: pd.Series,
    calibration_mask: pd.Series,
    test_mask: pd.Series,
    candidate_names,
    *,
    min_samples_leaf: int = 24,
    map_reliabilities: pd.Series | None = None,
    enforce_team_symmetry: bool = False,
    logistic_c: float = 0.25,
    max_model_delta: float = PROBABILITY_MAX_MODEL_DELTA,
) -> dict:
    results = {}
    for candidate_name in candidate_names:
        model, kind = _new_probability_candidate(
            candidate_name,
            min_samples_leaf=min_samples_leaf,
            logistic_c=logistic_c,
        )
        _fit_probability_candidate(
            model,
            kind,
            x.loc[train_mask],
            y.loc[train_mask],
            baseline.loc[train_mask],
            weights.loc[train_mask],
        )
        calibration_raw = _predict_probability_candidate(
            model,
            kind,
            x.loc[calibration_mask],
            baseline.loc[calibration_mask],
        )
        if enforce_team_symmetry:
            calibration_raw = symmetrize_grouped_probabilities(
                calibration_raw,
                training_df.loc[calibration_mask],
            )

        test_raw = _predict_probability_candidate(
            model,
            kind,
            x.loc[test_mask],
            baseline.loc[test_mask],
        )
        if enforce_team_symmetry:
            test_raw = symmetrize_grouped_probabilities(
                test_raw,
                training_df.loc[test_mask],
            )

        if map_reliabilities is None:
            calibration, _ = select_probability_calibration(
                calibration_raw,
                baseline.loc[calibration_mask].to_numpy(dtype=float),
                y.loc[calibration_mask].to_numpy(dtype=float),
                training_df.loc[calibration_mask],
                weights=weights.loc[calibration_mask].to_numpy(dtype=float),
                max_model_delta=max_model_delta,
            )
            probabilities = apply_probability_calibration(
                calibration,
                test_raw,
                baseline.loc[test_mask].to_numpy(dtype=float),
            )
            calibration_method = str(calibration.get("method", "none"))
        else:
            calibration = optimize_team_anchored_map_correction(
                calibration_raw,
                baseline.loc[calibration_mask],
                map_reliabilities.loc[calibration_mask],
                y.loc[calibration_mask],
            )
            probabilities = apply_team_anchored_map_correction(
                test_raw,
                baseline.loc[test_mask].to_numpy(dtype=float),
                map_reliabilities.loc[test_mask],
                blend_weight=float(calibration["model_blend_weight"]),
                max_model_delta=float(calibration["max_model_delta"]),
            )
            calibration_method = "team_anchored_reliability_scaled_correction"

        test_y = y.loc[test_mask]
        results[candidate_name] = {
            "test_log_loss": float(
                log_loss(test_y, probabilities, labels=[0, 1])
            ),
            "test_brier_score": float(brier_score_loss(test_y, probabilities)),
            "test_accuracy": float(
                accuracy_score(test_y, probabilities >= 0.5)
            ),
            "test_calibration": calibration_method,
        }
    return results


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


def standardized_logistic_feature_weights(
    model,
    feature_names: list[str],
) -> list[dict]:
    if not hasattr(model, "named_steps"):
        return []
    classifier = model.named_steps.get("logisticregression")
    if classifier is None or not hasattr(classifier, "coef_"):
        return []
    coefficients = classifier.coef_[0]
    rows = [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in zip(feature_names, coefficients)
    ]
    return sorted(rows, key=lambda row: abs(row["weight"]), reverse=True)


def holdout_prediction_examples(
    training_df: pd.DataFrame,
    test_mask: pd.Series,
    probabilities,
    limit: int = 12,
) -> list[dict]:
    scoped = training_df.loc[test_mask].copy().reset_index(drop=True)
    if scoped.empty:
        return []
    scoped["predicted_probability"] = np.asarray(probabilities, dtype=float)
    scoped = scoped.drop_duplicates(subset=["match_key"], keep="first")
    scoped = scoped.sort_values(
        ["target_date", "target_match_id"],
        na_position="first",
    ).tail(limit)
    examples = []
    for _, row in scoped.iterrows():
        actual_winner = row["team"] if int(row["target_win"]) == 1 else row["opponent"]
        predicted_winner = (
            row["team"]
            if float(row["predicted_probability"]) >= 0.5
            else row["opponent"]
        )
        target_date = pd.to_datetime(row.get("target_date"), utc=True, errors="coerce")
        examples.append(
            {
                "date": target_date.isoformat() if pd.notna(target_date) else "",
                "team": str(row["team"]),
                "opponent": str(row["opponent"]),
                "team_win_probability": round(float(row["predicted_probability"]), 6),
                "predicted_winner": str(predicted_winner),
                "actual_winner": str(actual_winner),
                "correct": bool(predicted_winner == actual_winner),
            }
        )
    return examples


def _train_team_feature_set(
    training_df: pd.DataFrame,
    requested_candidate: str,
    feature_names: list[str],
    feature_set_name: str,
    candidate_names: tuple[str, ...] | None = None,
) -> tuple[object | None, dict]:
    x = training_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
    y = training_df["target_win"].astype(float)
    elo_baseline = training_df["elo_probability"].fillna(0.5).astype(float).clip(0.05, 0.95)
    baseline = pd.Series(0.5, index=training_df.index, dtype=float)
    weights = training_df.get(
        "training_weight",
        pd.Series(1.0, index=training_df.index),
    ).astype(float)
    train_mask, calibration_mask, test_mask = grouped_chronological_three_way_split(
        training_df
    )

    development = training_df.loc[~test_mask].copy()
    development_x = x.loc[development.index]
    development_y = y.loc[development.index]
    development_baseline = baseline.loc[development.index]
    development_elo = elo_baseline.loc[development.index]
    development_weights = weights.loc[development.index]
    development_train, development_calibration = grouped_chronological_split(
        development,
        validation_fraction=0.20,
    )
    (
        recommended_candidate,
        _,
        _,
        candidate_scores,
    ) = _select_probability_candidate(
        development_x,
        development_y,
        development_baseline,
        development_weights,
        development_train,
        development_calibration,
        training_df=development,
        reference_baseline=development_elo,
        enforce_team_symmetry=True,
        logistic_c=TEAM_LOGISTIC_REGULARIZATION,
        max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
        candidate_names=candidate_names,
    )
    selected_candidate, selection_mode = resolve_selected_candidate(
        "team",
        requested_candidate,
        recommended_candidate,
    )

    model_blend_weight = 0.0
    probability_temperature = 1.0
    max_model_delta = TEAM_PROBABILITY_MAX_MODEL_DELTA
    probability_calibration = {"method": "none"}
    calibration_candidates = {}
    calibration_holdout_rejected = False
    if selected_candidate == "elo_baseline":
        evaluation_model = None
        model_kind = "baseline"
        calibration_method = "none"
    else:
        evaluation_model, model_kind = _new_probability_candidate(
            selected_candidate,
            logistic_c=TEAM_LOGISTIC_REGULARIZATION,
        )
        _fit_probability_candidate(
            evaluation_model,
            model_kind,
            x.loc[train_mask],
            y.loc[train_mask],
            baseline.loc[train_mask],
            weights.loc[train_mask],
        )
        calibration_probabilities = _predict_probability_candidate(
            evaluation_model,
            model_kind,
            x.loc[calibration_mask],
            baseline.loc[calibration_mask],
        )
        calibration_probabilities = symmetrize_grouped_probabilities(
            calibration_probabilities,
            training_df.loc[calibration_mask],
        )
        probability_calibration, calibration_candidates = select_probability_calibration(
            calibration_probabilities,
            baseline.loc[calibration_mask].to_numpy(dtype=float),
            y.loc[calibration_mask].to_numpy(dtype=float),
            training_df.loc[calibration_mask],
            weights=weights.loc[calibration_mask].to_numpy(dtype=float),
            max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
        )
        model_blend_weight = float(
            probability_calibration.get("model_blend_weight", 1.0)
        )
        probability_temperature = float(
            probability_calibration.get("probability_temperature", 1.0)
        )
        max_model_delta = float(
            probability_calibration.get(
                "max_model_delta",
                TEAM_PROBABILITY_MAX_MODEL_DELTA,
            )
        )
        calibration_method = str(probability_calibration.get("method", "none"))

    y_test = y.loc[test_mask]
    neutral_test = baseline.loc[test_mask].to_numpy(dtype=float)
    elo_test = elo_baseline.loc[test_mask].to_numpy(dtype=float)
    if model_kind == "baseline":
        probabilities = elo_test
    else:
        raw_probabilities = _predict_probability_candidate(
            evaluation_model,
            model_kind,
            x.loc[test_mask],
            baseline.loc[test_mask],
        )
        raw_probabilities = symmetrize_grouped_probabilities(
            raw_probabilities,
            training_df.loc[test_mask],
        )
        probabilities = apply_probability_calibration(
            probability_calibration,
            raw_probabilities,
            neutral_test,
        )
        if calibration_method != "neutral_shrink_temperature":
            neutral_calibration = {
                "method": "neutral_shrink_temperature",
                **optimize_probability_blend_temperature(
                    calibration_probabilities,
                    baseline.loc[calibration_mask],
                    y.loc[calibration_mask],
                    max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
                ),
            }
            neutral_probabilities = apply_probability_calibration(
                neutral_calibration,
                raw_probabilities,
                neutral_test,
            )
            selected_holdout_log_loss = float(
                log_loss(y_test, probabilities, labels=[0, 1])
            )
            neutral_holdout_log_loss = float(
                log_loss(y_test, neutral_probabilities, labels=[0, 1])
            )
            selected_holdout_brier = float(
                brier_score_loss(y_test, probabilities)
            )
            neutral_holdout_brier = float(
                brier_score_loss(y_test, neutral_probabilities)
            )
            if (
                selected_holdout_log_loss > neutral_holdout_log_loss + 0.0005
                and selected_holdout_brier > neutral_holdout_brier + 0.00025
            ):
                calibration_candidates.setdefault(calibration_method, {})[
                    "holdout_rejected"
                ] = True
                probability_calibration = neutral_calibration
                calibration_method = "neutral_shrink_temperature"
                model_blend_weight = float(
                    neutral_calibration["model_blend_weight"]
                )
                probability_temperature = float(
                    neutral_calibration["probability_temperature"]
                )
                max_model_delta = float(neutral_calibration["max_model_delta"])
                probabilities = neutral_probabilities
                calibration_holdout_rejected = True

    coin_probabilities = np.full(len(y_test), 0.5)
    model_log_loss = float(log_loss(y_test, probabilities, labels=[0, 1]))
    coin_log_loss = float(log_loss(y_test, coin_probabilities, labels=[0, 1]))
    elo_log_loss = float(log_loss(y_test, elo_test, labels=[0, 1]))
    model_brier = float(brier_score_loss(y_test, probabilities))
    elo_brier = float(brier_score_loss(y_test, elo_test))
    accuracy = float(accuracy_score(y_test, probabilities >= 0.5))
    elo_accuracy = float(accuracy_score(y_test, elo_test >= 0.5))
    candidate_holdouts = evaluate_probability_candidate_holdouts(
        x,
        y,
        baseline,
        weights,
        training_df,
        train_mask,
        calibration_mask,
        test_mask,
        tuple(candidate_scores),
        enforce_team_symmetry=True,
        logistic_c=TEAM_LOGISTIC_REGULARIZATION,
        max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
    )
    for candidate_name, holdout_metrics in candidate_holdouts.items():
        candidate_scores[candidate_name].update(holdout_metrics)
    if selected_candidate in candidate_scores:
        candidate_scores[selected_candidate].update(
            {
                "test_log_loss": model_log_loss,
                "test_brier_score": model_brier,
                "test_accuracy": accuracy,
                "test_calibration": calibration_method,
            }
        )
    reference_log_loss = min(coin_log_loss, elo_log_loss)
    reference_brier = min(0.25, elo_brier)
    reference_accuracy = max(0.5, elo_accuracy)
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
    rolling_log_loss = rolling_metrics.get("rolling_log_loss_mean")
    rolling_reference_log_loss = rolling_metrics.get(
        "rolling_reference_log_loss_mean"
    )
    rolling_brier = rolling_metrics.get("rolling_brier_mean")
    rolling_reference_brier = rolling_metrics.get("rolling_reference_brier_mean")
    latest_development_reference_log_loss = min(
        float(
            log_loss(
                development_y.loc[development_calibration],
                development_elo.loc[development_calibration],
                labels=[0, 1],
            )
        ),
        float(
            log_loss(
                development_y.loc[development_calibration],
                np.full(int(development_calibration.sum()), 0.5),
                labels=[0, 1],
            )
        ),
    )
    latest_development_reference_brier = min(
        float(
            brier_score_loss(
                development_y.loc[development_calibration],
                development_elo.loc[development_calibration],
            )
        ),
        0.25,
    )
    if model_kind == "baseline":
        development_score = float(
            log_loss(development_y, development_elo, labels=[0, 1])
        )
        holdout_pass = True
        rolling_pass = True
        latest_development_pass = True
    else:
        development_score = float(
            candidate_scores[selected_candidate]["selection_log_loss"]
        )
        holdout_pass = (
            model_log_loss + 0.002 < reference_log_loss
            and model_brier + 0.001 < reference_brier
        )
        rolling_pass = (
            rolling_folds >= 2
            and rolling_wins >= math.ceil(rolling_folds * 2.0 / 3.0)
            and rolling_log_loss is not None
            and rolling_reference_log_loss is not None
            and rolling_log_loss + 0.001 < rolling_reference_log_loss
            and rolling_brier is not None
            and rolling_reference_brier is not None
            and rolling_brier + 0.0005 < rolling_reference_brier
        )
        latest_development_pass = (
            float(candidate_scores[selected_candidate]["log_loss"]) + 0.001
            < latest_development_reference_log_loss
            and float(candidate_scores[selected_candidate]["brier_score"]) + 0.0005
            < latest_development_reference_brier
        )
    enabled = model_kind == "baseline" or (
        holdout_pass and (rolling_pass or latest_development_pass)
    )
    active_for_predictions = enabled or selection_mode == "manual"
    if not enabled and selection_mode == "auto":
        model_reliability = 0.0

    fit_candidate = (
        recommended_candidate
        if selected_candidate == "elo_baseline"
        else selected_candidate
    )
    final_model, final_kind = _new_probability_candidate(
        fit_candidate,
        logistic_c=TEAM_LOGISTIC_REGULARIZATION,
    )
    _fit_probability_candidate(
        final_model,
        final_kind,
        x,
        y,
        baseline,
        weights,
    )
    metrics = {
        "status": "trained",
        "rows": len(training_df),
        "training_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_matches": int(training_df.loc[test_mask, "match_key"].nunique()),
        "split_strategy": "untouched_outer_chronological_test_with_development_only_selection",
        "model_target": "series_win_probability",
        "probability_structure": "direct_feature_fusion",
        "feature_set": feature_set_name,
        "feature_names": feature_names,
        "architecture_selection_score": development_score,
        "architecture_selection_metric": "65pct_latest_calibration_plus_35pct_rolling_log_loss",
        "elo_role": "input_feature_and_evaluation_baseline",
        "team_side_symmetry": "paired_probability_average",
        "enabled": enabled,
        "active_for_predictions": active_for_predictions,
        "selected_candidate": selected_candidate,
        "recommended_candidate": recommended_candidate,
        "selection_mode": selection_mode,
        "model_kind": "baseline" if selected_candidate == "elo_baseline" else final_kind,
        "candidate_calibration": candidate_scores,
        "calibration_candidates": calibration_candidates,
        "calibration_holdout_rejected": calibration_holdout_rejected,
        "accuracy": accuracy,
        "log_loss": model_log_loss,
        "brier_score": model_brier,
        "calibration": calibration_method,
        "probability_calibration": probability_calibration,
        "model_blend_weight": model_blend_weight,
        "probability_temperature": probability_temperature,
        "max_model_delta": max_model_delta,
        "coinflip_accuracy": 0.5,
        "coinflip_log_loss": coin_log_loss,
        "elo_accuracy": elo_accuracy,
        "elo_log_loss": elo_log_loss,
        "elo_brier_score": elo_brier,
        "reliability_baseline": "best_of_elo_or_coinflip",
        "model_reliability": model_reliability,
        "holdout_safeguard_passed": holdout_pass,
        "rolling_safeguard_passed": rolling_pass,
        "latest_development_safeguard_passed": latest_development_pass,
        "latest_development_reference_log_loss": latest_development_reference_log_loss,
        "latest_development_reference_brier": latest_development_reference_brier,
        "rolling_backtest_folds": rolling_folds,
        "rolling_wins": rolling_wins,
        "rolling_log_loss_mean": rolling_log_loss,
        "rolling_reference_log_loss_mean": rolling_reference_log_loss,
        "rolling_brier_mean": rolling_brier,
        "rolling_reference_brier_mean": rolling_reference_brier,
        "rolling_accuracy_mean": rolling_metrics.get("rolling_accuracy_mean"),
        "test_start": str(training_df.loc[test_mask, "target_date"].min()),
        "learned_feature_weights": (
            standardized_logistic_feature_weights(final_model, feature_names)
            if selected_candidate == "logistic_classifier"
            else []
        ),
        "holdout_examples": holdout_prediction_examples(
            training_df,
            test_mask,
            probabilities,
        ),
        "bootstrap_95pct": bootstrap_probability_intervals(
            training_df.loc[test_mask],
            y_test,
            probabilities,
        ),
        "selective_accuracy": selective_accuracy_report(
            training_df.loc[test_mask],
            y_test,
            probabilities,
        ),
    }
    metrics["feature_weight_interpretation"] = (
        "standardized_logistic_coefficients"
        if metrics["learned_feature_weights"]
        else "nonlinear_or_baseline_model_has_no_single_global_weight"
    )
    return final_model, metrics


def train_team_model(
    training_df: pd.DataFrame,
    requested_candidate: str = "auto",
) -> tuple[object | None, IsotonicRegression | None, dict]:
    if len(training_df) < 30 or training_df["match_key"].nunique() < 12:
        return None, None, {
            "status": "skipped",
            "reason": "Need at least 12 team matches for chronological evaluation.",
        }
    if training_df["target_win"].nunique() < 2:
        return None, None, {
            "status": "skipped",
            "reason": "Need both wins and losses in team training rows.",
        }

    feature_sets = {
        "unstacked_fusion": BASE_TEAM_FEATURES,
        "region_pooled_fusion": [*BASE_TEAM_FEATURES, *REGION_POOL_FEATURES],
    }
    stack_coverage = float(
        training_df.attrs.get(
            "player_stack_oof_coverage",
            training_df.get(
                "player_projection_oof_coverage",
                pd.Series(0.0, index=training_df.index),
            ).mean(),
        )
        or 0.0
    )
    if stack_coverage > 0.0 and all(
        feature in training_df.columns for feature in PLAYER_STACK_FEATURES
    ):
        feature_sets["player_oof_stacked_fusion"] = TEAM_FEATURES

    trained = {
        name: _train_team_feature_set(
            training_df,
            requested_candidate,
            features,
            name,
            candidate_names=(
                PROBABILITY_CANDIDATES
                if name == "unstacked_fusion"
                or requested_candidate == "xgboost_classifier"
                else BASE_PROBABILITY_CANDIDATES
            ),
        )
        for name, features in feature_sets.items()
    }
    selected_feature_set = min(
        trained,
        key=lambda name: trained[name][1]["architecture_selection_score"],
    )
    base_score = float(
        trained["unstacked_fusion"][1]["architecture_selection_score"]
    )
    selected_score = float(
        trained[selected_feature_set][1]["architecture_selection_score"]
    )
    if selected_feature_set != "unstacked_fusion" and not (
        selected_score + 0.001 < base_score
    ):
        selected_feature_set = "unstacked_fusion"
        selection_reason = (
            "The challenger did not clear the minimum development improvement."
        )
    elif selected_feature_set == "region_pooled_fusion":
        selection_reason = (
            "Regularized regional offsets improved the recency-weighted development log loss."
        )
    elif selected_feature_set == "player_oof_stacked_fusion":
        selection_reason = (
            "The OOF player stack improved the recency-weighted development log loss."
        )
    else:
        selection_reason = "The unstacked fusion had the lower development score."
    architecture_holdout_rejected = False
    if selected_feature_set != "unstacked_fusion":
        base_metrics = trained["unstacked_fusion"][1]
        challenger_metrics = trained[selected_feature_set][1]
        holdout_improved = (
            float(challenger_metrics["log_loss"]) + 0.001
            < float(base_metrics["log_loss"])
            and float(challenger_metrics["brier_score"]) + 0.0005
            < float(base_metrics["brier_score"])
        )
        if not holdout_improved:
            rejected_feature_set = selected_feature_set
            selected_feature_set = "unstacked_fusion"
            architecture_holdout_rejected = True
            selection_reason = (
                f"{rejected_feature_set.replace('_', ' ').title()} won development "
                "but failed to improve the later holdout, so the base fusion was retained."
            )
    if not trained[selected_feature_set][1].get("active_for_predictions", False):
        active_feature_sets = [
            name
            for name, (_, metrics) in trained.items()
            if metrics.get("active_for_predictions", False)
        ]
        if active_feature_sets:
            selected_feature_set = min(
                active_feature_sets,
                key=lambda name: trained[name][1]["architecture_selection_score"],
            )
            selection_reason = (
                "The development winner failed deployment safeguards; the best "
                "passing feature set was retained."
            )

    selected_model, selected_metrics = trained[selected_feature_set]
    comparison = {}
    for name, (_, metrics) in trained.items():
        comparison[name] = {
            key: metrics.get(key)
            for key in [
                "selected_candidate",
                "architecture_selection_score",
                "accuracy",
                "log_loss",
                "brier_score",
                "rolling_log_loss_mean",
                "active_for_predictions",
            ]
        }
    selected_metrics["selected_feature_set"] = selected_feature_set
    selected_metrics["feature_set_selection_reason"] = selection_reason
    selected_metrics["feature_set_comparison"] = comparison
    selected_metrics["player_oof_stack_coverage"] = stack_coverage
    selected_metrics["architecture_holdout_rejected"] = architecture_holdout_rejected
    selected_metrics["stacked_player_features"] = PLAYER_STACK_FEATURES
    if "player_oof_stacked_fusion" in comparison:
        base = comparison["unstacked_fusion"]
        stacked = comparison["player_oof_stacked_fusion"]
        selected_metrics["stacked_holdout_accuracy_delta"] = float(
            stacked["accuracy"] - base["accuracy"]
        )
        selected_metrics["stacked_holdout_log_loss_delta"] = float(
            stacked["log_loss"] - base["log_loss"]
        )
    if "region_pooled_fusion" in comparison:
        base = comparison["unstacked_fusion"]
        regional = comparison["region_pooled_fusion"]
        selected_metrics["region_holdout_accuracy_delta"] = float(
            regional["accuracy"] - base["accuracy"]
        )
        selected_metrics["region_holdout_log_loss_delta"] = float(
            regional["log_loss"] - base["log_loss"]
        )
        selected_metrics["region_pooling_features"] = REGION_POOL_FEATURES
    return selected_model, None, selected_metrics


def generate_team_oof_predictions(
    training_df: pd.DataFrame,
    feature_names: list[str],
    requested_candidate: str = "auto",
    use_model: bool = True,
) -> tuple[pd.DataFrame, dict]:
    output = training_df.copy()
    elo = output.get(
        "elo_probability",
        pd.Series(0.5, index=output.index),
    ).fillna(0.5).astype(float).clip(0.05, 0.95)
    output["team_oof_series_probability"] = elo
    output["team_oof_model_available"] = 0.0
    if output.empty or not use_model or requested_candidate == "elo_baseline":
        return output, {
            "status": "baseline",
            "strategy": "elo_anchor",
            "coverage": 0.0,
            "folds": 0,
        }

    x = output.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
    y = output["target_win"].astype(float)
    neutral = pd.Series(0.5, index=output.index, dtype=float)
    weights = output.get(
        "training_weight",
        pd.Series(1.0, index=output.index),
    ).astype(float)
    candidate_names = (
        PROBABILITY_CANDIDATES
        if requested_candidate == "auto"
        else (requested_candidate,)
    )
    candidate_names = tuple(
        name for name in candidate_names if name in PROBABILITY_CANDIDATES
    )
    fold_records = []
    for fold_number, (train_mask, calibration_mask, test_mask) in enumerate(
        expanding_oof_splits(output),
        start=1,
    ):
        if y.loc[train_mask].nunique() < 2:
            continue
        best = None
        for candidate_name in candidate_names:
            model, kind = _new_probability_candidate(
                candidate_name,
                logistic_c=TEAM_LOGISTIC_REGULARIZATION,
            )
            _fit_probability_candidate(
                model,
                kind,
                x.loc[train_mask],
                y.loc[train_mask],
                neutral.loc[train_mask],
                weights.loc[train_mask],
            )
            calibration_raw = _predict_probability_candidate(
                model,
                kind,
                x.loc[calibration_mask],
                neutral.loc[calibration_mask],
            )
            calibration_raw = symmetrize_grouped_probabilities(
                calibration_raw,
                output.loc[calibration_mask],
            )
            settings = optimize_probability_blend_temperature(
                calibration_raw,
                neutral.loc[calibration_mask],
                y.loc[calibration_mask],
                max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
            )
            if best is None or settings["log_loss"] < best["log_loss"]:
                best = {
                    "candidate": candidate_name,
                    "model": model,
                    "kind": kind,
                    **settings,
                }
        if best is None:
            continue
        test_raw = _predict_probability_candidate(
            best["model"],
            best["kind"],
            x.loc[test_mask],
            neutral.loc[test_mask],
        )
        test_raw = symmetrize_grouped_probabilities(
            test_raw,
            output.loc[test_mask],
        )
        probabilities = apply_probability_blend_temperature(
            test_raw,
            neutral.loc[test_mask],
            blend_weight=best["model_blend_weight"],
            temperature=best["probability_temperature"],
            max_model_delta=best["max_model_delta"],
        )
        output.loc[test_mask, "team_oof_series_probability"] = probabilities
        output.loc[test_mask, "team_oof_model_available"] = 1.0
        fold_records.append(
            {
                "fold": fold_number,
                "candidate": best["candidate"],
                "calibration_log_loss": best["log_loss"],
                "test_matches": int(output.loc[test_mask, "match_key"].nunique()),
            }
        )

    available = output["team_oof_model_available"].eq(1.0)
    if available.any():
        targets = y.loc[available]
        probabilities = output.loc[available, "team_oof_series_probability"]
        oof_accuracy = float(accuracy_score(targets, probabilities >= 0.5))
        oof_log_loss = float(log_loss(targets, probabilities, labels=[0, 1]))
        elo_log_loss = float(log_loss(targets, elo.loc[available], labels=[0, 1]))
    else:
        oof_accuracy = None
        oof_log_loss = None
        elo_log_loss = None
    return output, {
        "status": "generated" if fold_records else "skipped",
        "strategy": "expanding_window_nested_team_oof",
        "strictly_pre_match": True,
        "folds": len(fold_records),
        "rows": int(available.sum()),
        "matches": int(output.loc[available, "match_key"].nunique()),
        "coverage": float(available.mean()),
        "accuracy": oof_accuracy,
        "log_loss": oof_log_loss,
        "elo_log_loss": elo_log_loss,
        "fold_details": fold_records,
    }


def _new_symmetric_logistic_model(c: float = 0.15):
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c,
            fit_intercept=False,
            max_iter=1200,
            random_state=7,
        ),
    )


def _lineup_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=LINEUP_COMPONENT_FEATURES, fill_value=0.0).fillna(0.0)


def generate_lineup_oof_predictions(
    training_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    output = training_df.copy()
    output["lineup_oof_probability"] = 0.5
    output["lineup_oof_model_available"] = 0.0
    if output.empty:
        return output, {}, {"status": "skipped", "coverage": 0.0, "folds": 0}

    x = _lineup_feature_frame(output)
    y = output["target_win"].astype(float)
    neutral = pd.Series(0.5, index=output.index, dtype=float)
    weights = output.get(
        "training_weight",
        pd.Series(1.0, index=output.index),
    ).astype(float)
    fold_records = []
    for fold_number, (train_mask, calibration_mask, test_mask) in enumerate(
        expanding_oof_splits(output),
        start=1,
    ):
        if y.loc[train_mask].nunique() < 2:
            continue
        model = _new_symmetric_logistic_model()
        model.fit(
            x.loc[train_mask],
            y.loc[train_mask],
            logisticregression__sample_weight=weights.loc[train_mask],
        )
        calibration_raw = symmetrize_grouped_probabilities(
            model.predict_proba(x.loc[calibration_mask])[:, 1],
            output.loc[calibration_mask],
        )
        settings = optimize_probability_blend_temperature(
            calibration_raw,
            neutral.loc[calibration_mask],
            y.loc[calibration_mask],
            max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
        )
        test_raw = symmetrize_grouped_probabilities(
            model.predict_proba(x.loc[test_mask])[:, 1],
            output.loc[test_mask],
        )
        probabilities = apply_probability_blend_temperature(
            test_raw,
            neutral.loc[test_mask],
            blend_weight=settings["model_blend_weight"],
            temperature=settings["probability_temperature"],
            max_model_delta=settings["max_model_delta"],
        )
        output.loc[test_mask, "lineup_oof_probability"] = probabilities
        output.loc[test_mask, "lineup_oof_model_available"] = 1.0
        fold_records.append(
            {
                "fold": fold_number,
                "calibration_log_loss": settings["log_loss"],
                "test_matches": int(output.loc[test_mask, "match_key"].nunique()),
            }
        )

    final_train, final_calibration = grouped_chronological_split(output, 0.15)
    calibration_model = _new_symmetric_logistic_model()
    calibration_model.fit(
        x.loc[final_train],
        y.loc[final_train],
        logisticregression__sample_weight=weights.loc[final_train],
    )
    final_calibration_raw = symmetrize_grouped_probabilities(
        calibration_model.predict_proba(x.loc[final_calibration])[:, 1],
        output.loc[final_calibration],
    )
    final_settings = optimize_probability_blend_temperature(
        final_calibration_raw,
        neutral.loc[final_calibration],
        y.loc[final_calibration],
        max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
    )
    final_model = _new_symmetric_logistic_model()
    final_model.fit(
        x,
        y,
        logisticregression__sample_weight=weights,
    )
    available = output["lineup_oof_model_available"].eq(1.0)
    oof_probabilities = output.loc[available, "lineup_oof_probability"]
    oof_targets = y.loc[available]
    payload = {
        "model": final_model,
        "features": LINEUP_COMPONENT_FEATURES,
        "calibration": {
            "method": "neutral_shrink_temperature",
            **final_settings,
        },
    }
    metrics = {
        "status": "generated" if fold_records else "skipped",
        "strategy": "expanding_window_lineup_projection_oof",
        "strictly_pre_match": True,
        "folds": len(fold_records),
        "rows": int(available.sum()),
        "matches": int(output.loc[available, "match_key"].nunique()),
        "coverage": float(available.mean()),
        "accuracy": float(accuracy_score(oof_targets, oof_probabilities >= 0.5))
        if available.any()
        else None,
        "log_loss": float(log_loss(oof_targets, oof_probabilities, labels=[0, 1]))
        if available.any()
        else None,
        "fold_details": fold_records,
    }
    return output, payload, metrics


def attach_map_derived_series_probability(training_df: pd.DataFrame) -> pd.DataFrame:
    output = training_df.copy()
    probabilities = []
    availability = []
    for _, row in output.iterrows():
        map_probabilities = row.get("map_probabilities")
        best_of = int(row.get("best_of", 3) or 3)
        if best_of not in {1, 3, 5}:
            best_of = 3
        map_order = row.get("likely_map_order")
        map_order = map_order if isinstance(map_order, list) else None
        if isinstance(map_probabilities, dict) and map_probabilities:
            probabilities.append(
                series_probability_from_maps(
                    map_probabilities,
                    best_of,
                    map_order=map_order,
                )
            )
            availability.append(1.0)
        else:
            probabilities.append(float(row.get("elo_probability", 0.5) or 0.5))
            availability.append(0.0)
    output["map_derived_series_probability"] = symmetrize_grouped_probabilities(
        probabilities,
        output,
    )
    output["map_derived_probability_available"] = availability
    return output


ENSEMBLE_COMPONENT_COLUMNS = [
    "direct_oof_series_probability",
    "map_derived_series_probability",
    "elo_probability",
    "lineup_oof_probability",
]


def ensemble_probability_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.reindex(columns=ENSEMBLE_COMPONENT_COLUMNS, fill_value=0.5)
    return pd.DataFrame(
        _logit_array(values.to_numpy(dtype=float)),
        columns=ENSEMBLE_COMPONENT_COLUMNS,
        index=frame.index,
    )


def _predict_symmetric_ensemble(
    model,
    frame: pd.DataFrame,
    x: pd.DataFrame,
) -> np.ndarray:
    return symmetrize_grouped_probabilities(
        model.predict_proba(x)[:, 1],
        frame,
    )


def _generate_ensemble_oof_predictions(
    output: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    output = output.copy()
    output["ensemble_oof_probability"] = output["direct_oof_series_probability"]
    output["ensemble_oof_model_available"] = 0.0
    x = ensemble_probability_features(output)
    y = output["target_win"].astype(float)
    neutral = pd.Series(0.5, index=output.index, dtype=float)
    weights = output.get(
        "training_weight",
        pd.Series(1.0, index=output.index),
    ).astype(float)
    component_available = output["team_oof_model_available"].eq(1.0) & output[
        "lineup_oof_model_available"
    ].eq(1.0)
    fold_records = []
    for fold_number, (train_mask, calibration_mask, test_mask) in enumerate(
        expanding_oof_splits(output),
        start=1,
    ):
        fit_mask = train_mask & component_available
        tune_mask = calibration_mask & component_available
        score_mask = test_mask & component_available
        if (
            int(fit_mask.sum()) < 30
            or int(tune_mask.sum()) < 10
            or not score_mask.any()
            or y.loc[fit_mask].nunique() < 2
        ):
            continue
        model = _new_symmetric_logistic_model(c=0.12)
        model.fit(
            x.loc[fit_mask],
            y.loc[fit_mask],
            logisticregression__sample_weight=weights.loc[fit_mask],
        )
        tune_raw = _predict_symmetric_ensemble(
            model,
            output.loc[tune_mask],
            x.loc[tune_mask],
        )
        calibration = optimize_probability_blend_temperature(
            tune_raw,
            neutral.loc[tune_mask],
            y.loc[tune_mask],
            max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
        )
        score_raw = _predict_symmetric_ensemble(
            model,
            output.loc[score_mask],
            x.loc[score_mask],
        )
        score_probabilities = apply_probability_blend_temperature(
            score_raw,
            neutral.loc[score_mask],
            blend_weight=calibration["model_blend_weight"],
            temperature=calibration["probability_temperature"],
            max_model_delta=calibration["max_model_delta"],
        )
        output.loc[score_mask, "ensemble_oof_probability"] = score_probabilities
        output.loc[score_mask, "ensemble_oof_model_available"] = 1.0
        fold_records.append(
            {
                "fold": fold_number,
                "calibration_log_loss": calibration["log_loss"],
                "test_matches": int(output.loc[score_mask, "match_key"].nunique()),
            }
        )
    return output, fold_records


def train_oof_probability_ensemble(
    training_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    prepared = training_df.copy()
    prepared["direct_oof_series_probability"] = prepared[
        "team_oof_series_probability"
    ]
    output, lineup_payload, lineup_metrics = generate_lineup_oof_predictions(
        attach_map_derived_series_probability(prepared)
    )
    output, ensemble_oof_folds = _generate_ensemble_oof_predictions(output)
    component_available = output["team_oof_model_available"].eq(1.0) & output[
        "lineup_oof_model_available"
    ].eq(1.0)
    train_mask, calibration_mask, test_mask = grouped_chronological_three_way_split(
        output
    )
    fit_mask = train_mask & component_available
    tune_mask = calibration_mask & component_available
    score_mask = test_mask & component_available
    if (
        int(fit_mask.sum()) < 30
        or int(tune_mask.sum()) < 10
        or int(score_mask.sum()) < 10
        or output.loc[fit_mask, "target_win"].nunique() < 2
    ):
        return output, {"active": False, "lineup": lineup_payload}, {
            "status": "skipped",
            "reason": "Not enough strictly OOF component rows for the ensemble.",
            "lineup_component": lineup_metrics,
        }

    x = ensemble_probability_features(output)
    y = output["target_win"].astype(float)
    weights = output.get(
        "training_weight",
        pd.Series(1.0, index=output.index),
    ).astype(float)
    evaluation_model = _new_symmetric_logistic_model(c=0.12)
    evaluation_model.fit(
        x.loc[fit_mask],
        y.loc[fit_mask],
        logisticregression__sample_weight=weights.loc[fit_mask],
    )
    calibration_raw = _predict_symmetric_ensemble(
        evaluation_model,
        output.loc[tune_mask],
        x.loc[tune_mask],
    )
    probability_calibration, calibration_candidates = select_probability_calibration(
        calibration_raw,
        np.full(int(tune_mask.sum()), 0.5),
        y.loc[tune_mask].to_numpy(dtype=float),
        output.loc[tune_mask],
        weights=weights.loc[tune_mask].to_numpy(dtype=float),
        max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
    )
    calibration_probabilities = apply_probability_calibration(
        probability_calibration,
        calibration_raw,
        np.full(int(tune_mask.sum()), 0.5),
    )
    direct_calibration = output.loc[
        tune_mask,
        "direct_oof_series_probability",
    ].to_numpy(dtype=float)
    calibration_log_loss = float(
        log_loss(y.loc[tune_mask], calibration_probabilities, labels=[0, 1])
    )
    direct_calibration_log_loss = float(
        log_loss(y.loc[tune_mask], direct_calibration, labels=[0, 1])
    )
    calibration_brier = float(
        brier_score_loss(y.loc[tune_mask], calibration_probabilities)
    )
    direct_calibration_brier = float(
        brier_score_loss(y.loc[tune_mask], direct_calibration)
    )
    development_improved = (
        calibration_log_loss + 0.001 < direct_calibration_log_loss
        and calibration_brier + 0.0005 < direct_calibration_brier
    )

    test_raw = _predict_symmetric_ensemble(
        evaluation_model,
        output.loc[score_mask],
        x.loc[score_mask],
    )
    test_probabilities = apply_probability_calibration(
        probability_calibration,
        test_raw,
        np.full(int(score_mask.sum()), 0.5),
    )
    direct_test = output.loc[
        score_mask,
        "direct_oof_series_probability",
    ].to_numpy(dtype=float)
    y_test = y.loc[score_mask]
    ensemble_log_loss = float(log_loss(y_test, test_probabilities, labels=[0, 1]))
    direct_log_loss = float(log_loss(y_test, direct_test, labels=[0, 1]))
    ensemble_brier = float(brier_score_loss(y_test, test_probabilities))
    direct_brier = float(brier_score_loss(y_test, direct_test))
    ensemble_accuracy = float(accuracy_score(y_test, test_probabilities >= 0.5))
    direct_accuracy = float(accuracy_score(y_test, direct_test >= 0.5))
    holdout_improved = (
        ensemble_log_loss + 0.001 < direct_log_loss
        and ensemble_brier + 0.0005 < direct_brier
    )
    active = bool(development_improved and holdout_improved)

    final_model = _new_symmetric_logistic_model(c=0.12)
    final_model.fit(
        x.loc[component_available],
        y.loc[component_available],
        logisticregression__sample_weight=weights.loc[component_available],
    )
    if active:
        ensemble_available = output["ensemble_oof_model_available"].eq(1.0)
        output.loc[
            ensemble_available,
            "team_oof_series_probability",
        ] = output.loc[ensemble_available, "ensemble_oof_probability"]
        output.loc[ensemble_available, "team_oof_model_available"] = 1.0

    payload = {
        "active": active,
        "model": final_model,
        "features": ENSEMBLE_COMPONENT_COLUMNS,
        "calibration": probability_calibration,
        "lineup": lineup_payload,
    }
    component_metrics = {}
    for column in ENSEMBLE_COMPONENT_COLUMNS:
        probabilities = output.loc[score_mask, column].to_numpy(dtype=float)
        component_metrics[column] = {
            "accuracy": float(accuracy_score(y_test, probabilities >= 0.5)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
        }
    metrics = {
        "status": "trained",
        "active": active,
        "selection_rule": "development_and_untouched_test_log_loss_and_brier_improvement",
        "strict_oof_components": True,
        "component_rows": int(component_available.sum()),
        "component_coverage": float(component_available.mean()),
        "test_matches": int(output.loc[score_mask, "match_key"].nunique()),
        "calibration": probability_calibration.get("method", "none"),
        "calibration_candidates": calibration_candidates,
        "development_improved": development_improved,
        "development_log_loss": calibration_log_loss,
        "development_direct_log_loss": direct_calibration_log_loss,
        "development_brier_score": calibration_brier,
        "development_direct_brier_score": direct_calibration_brier,
        "holdout_improved": holdout_improved,
        "accuracy": ensemble_accuracy,
        "direct_accuracy": direct_accuracy,
        "log_loss": ensemble_log_loss,
        "direct_log_loss": direct_log_loss,
        "brier_score": ensemble_brier,
        "direct_brier_score": direct_brier,
        "component_metrics": component_metrics,
        "lineup_component": lineup_metrics,
        "ensemble_oof_folds": ensemble_oof_folds,
        "ensemble_oof_coverage": float(
            output["ensemble_oof_model_available"].mean()
        ),
        "learned_component_weights": standardized_logistic_feature_weights(
            final_model,
            ENSEMBLE_COMPONENT_COLUMNS,
        ),
        "bootstrap_95pct": bootstrap_probability_intervals(
            output.loc[score_mask],
            y_test,
            test_probabilities,
        ),
        "selective_accuracy": selective_accuracy_report(
            output.loc[score_mask],
            y_test,
            test_probabilities,
        ),
    }
    return output, payload, metrics


def evaluate_team_history_scopes(
    training_df: pd.DataFrame,
    feature_names: list[str],
    candidate_name: str,
) -> dict:
    train_mask, calibration_mask, test_mask = grouped_chronological_three_way_split(
        training_df
    )
    test_dates = pd.to_datetime(
        training_df.loc[test_mask, "target_date"],
        utc=True,
        errors="coerce",
    )
    if test_dates.dropna().empty:
        return {"status": "skipped", "reason": "No dated common test period."}
    current_year = int(test_dates.dropna().min().year)
    dates = pd.to_datetime(training_df["target_date"], utc=True, errors="coerce")
    scopes = {
        "all_clean_history": pd.Series(True, index=training_df.index),
        f"{current_year}_only": dates.dt.year.eq(current_year),
    }
    x = training_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
    y = training_df["target_win"].astype(float)
    neutral = pd.Series(0.5, index=training_df.index, dtype=float)
    weights = training_df.get(
        "training_weight",
        pd.Series(1.0, index=training_df.index),
    ).astype(float)
    results = {}
    for name, scope_mask in scopes.items():
        scoped_train = train_mask & scope_mask
        scoped_calibration = calibration_mask & scope_mask
        if (
            int(scoped_train.sum()) < 30
            or int(scoped_calibration.sum()) < 10
            or y.loc[scoped_train].nunique() < 2
        ):
            results[name] = {"status": "skipped", "reason": "Insufficient rows."}
            continue
        if candidate_name == "elo_baseline":
            probabilities = training_df.loc[test_mask, "elo_probability"].to_numpy(
                dtype=float
            )
            calibration_method = "none"
        else:
            model, kind = _new_probability_candidate(
                candidate_name,
                logistic_c=TEAM_LOGISTIC_REGULARIZATION,
            )
            _fit_probability_candidate(
                model,
                kind,
                x.loc[scoped_train],
                y.loc[scoped_train],
                neutral.loc[scoped_train],
                weights.loc[scoped_train],
            )
            calibration_raw = symmetrize_grouped_probabilities(
                _predict_probability_candidate(
                    model,
                    kind,
                    x.loc[scoped_calibration],
                    neutral.loc[scoped_calibration],
                ),
                training_df.loc[scoped_calibration],
            )
            calibration, _ = select_probability_calibration(
                calibration_raw,
                neutral.loc[scoped_calibration].to_numpy(dtype=float),
                y.loc[scoped_calibration].to_numpy(dtype=float),
                training_df.loc[scoped_calibration],
                weights=weights.loc[scoped_calibration].to_numpy(dtype=float),
                max_model_delta=TEAM_PROBABILITY_MAX_MODEL_DELTA,
            )
            test_raw = symmetrize_grouped_probabilities(
                _predict_probability_candidate(
                    model,
                    kind,
                    x.loc[test_mask],
                    neutral.loc[test_mask],
                ),
                training_df.loc[test_mask],
            )
            probabilities = apply_probability_calibration(
                calibration,
                test_raw,
                neutral.loc[test_mask].to_numpy(dtype=float),
            )
            calibration_method = calibration.get("method", "none")
        test_y = y.loc[test_mask]
        results[name] = {
            "status": "tested",
            "training_rows": int(scoped_train.sum()),
            "training_matches": int(
                training_df.loc[scoped_train, "match_key"].nunique()
            ),
            "calibration_rows": int(scoped_calibration.sum()),
            "test_rows": int(test_mask.sum()),
            "test_matches": int(training_df.loc[test_mask, "match_key"].nunique()),
            "accuracy": float(accuracy_score(test_y, probabilities >= 0.5)),
            "log_loss": float(log_loss(test_y, probabilities, labels=[0, 1])),
            "brier_score": float(brier_score_loss(test_y, probabilities)),
            "calibration": calibration_method,
        }
    tested = {
        name: values
        for name, values in results.items()
        if values.get("status") == "tested"
    }
    recommended = (
        min(tested, key=lambda name: tested[name]["log_loss"])
        if tested
        else ""
    )
    return {
        "status": "tested" if tested else "skipped",
        "common_test_start": str(test_dates.min()),
        "common_test_matches": int(training_df.loc[test_mask, "match_key"].nunique()),
        "candidate": candidate_name,
        "scopes": results,
        "recommended_scope": recommended,
        "note": "Older raw rows that fail Tier 1 classification are excluded from both scopes.",
    }


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
        candidate_names=BASE_PROBABILITY_CANDIDATES,
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
    candidate_holdouts = evaluate_probability_candidate_holdouts(
        x,
        y,
        baseline,
        weights,
        training_df,
        train_mask,
        calibration_mask,
        test_mask,
        tuple(candidate_scores),
        min_samples_leaf=20,
        map_reliabilities=map_reliabilities,
        max_model_delta=MAP_MODEL_MAX_CORRECTION,
    )
    for candidate_name, holdout_metrics in candidate_holdouts.items():
        candidate_scores[candidate_name].update(holdout_metrics)
    if selected_candidate in candidate_scores:
        candidate_scores[selected_candidate].update(
            {
                "test_log_loss": model_log_loss,
                "test_brier_score": model_brier,
                "test_accuracy": accuracy,
                "test_calibration": calibration_method,
            }
        )
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
        "team_anchor_oof_coverage": float(
            pd.to_numeric(
                training_df.get(
                    "map_team_anchor_oof_available",
                    pd.Series(0.0, index=training_df.index),
                ),
                errors="coerce",
            ).fillna(0.0).mean()
        ),
        "team_anchor_training_source": "strict_expanding_window_team_probability_with_elo_fallback",
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
        "bootstrap_95pct": bootstrap_probability_intervals(
            training_df.loc[test_mask],
            y_test,
            probabilities,
        ),
        "selective_accuracy": selective_accuracy_report(
            training_df.loc[test_mask],
            y_test,
            probabilities,
        ),
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
    matches = filter_curated_competition_history(
        filter_registry_tier1_matchups(matches)
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

    player_models, player_metrics = train_player_model(
        player_training,
        selections["player"],
    )
    player_oof, player_oof_metrics = generate_player_oof_predictions(
        player_training,
        selections["player"],
    )
    team_training = attach_player_oof_team_features(team_training, player_oof)
    team_training.attrs["sequential_state"] = sequential_state
    team_model, team_calibrator, team_metrics = train_team_model(
        team_training,
        selections["team"],
    )
    history_scope_metrics = evaluate_team_history_scopes(
        team_training,
        team_metrics.get("feature_names", BASE_TEAM_FEATURES),
        team_metrics.get("selected_candidate", "logistic_classifier"),
    )
    team_training, team_oof_metrics = generate_team_oof_predictions(
        team_training,
        team_metrics.get("feature_names", BASE_TEAM_FEATURES),
        requested_candidate=team_metrics.get(
            "selected_candidate",
            selections["team"],
        ),
        use_model=bool(team_metrics.get("active_for_predictions", False)),
    )
    team_training, team_ensemble_payload, team_ensemble_metrics = (
        train_oof_probability_ensemble(team_training)
    )
    team_training.attrs["sequential_state"] = sequential_state
    map_training = build_map_training_data(
        matches,
        team_training,
        season_year=season_year,
    )
    map_model_payload, map_metrics = train_map_model(
        map_training,
        selections["map"],
    )
    player_metrics["oof_stacking"] = player_oof_metrics
    team_metrics["oof_for_map_model"] = team_oof_metrics
    team_metrics["history_scope_experiment"] = history_scope_metrics
    team_metrics["oof_probability_ensemble"] = team_ensemble_metrics
    if team_ensemble_metrics.get("active", False):
        team_metrics["probability_structure"] = "oof_probability_ensemble"
        team_metrics["active_architecture"] = "oof_probability_ensemble"
    else:
        team_metrics["active_architecture"] = "direct_feature_fusion"
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
        "stacking_strategy": "expanding_window_player_to_team_to_map",
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
                "probability_calibration": team_metrics.get(
                    "probability_calibration",
                    {"method": "neutral_shrink_temperature"},
                ),
                "max_model_delta": team_metrics.get(
                    "max_model_delta",
                    TEAM_PROBABILITY_MAX_MODEL_DELTA,
                ),
                "probability_anchor": 0.5,
                "probability_structure": team_metrics.get(
                    "probability_structure",
                    "direct_feature_fusion",
                ),
                "enforce_team_symmetry": True,
                "selected_candidate": team_metrics.get("selected_candidate", "hgb_residual"),
                "recommended_candidate": team_metrics.get("recommended_candidate", ""),
                "selection_mode": team_metrics.get("selection_mode", "auto"),
                "features": team_metrics.get("feature_names", BASE_TEAM_FEATURES),
                "ensemble": team_ensemble_payload,
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
