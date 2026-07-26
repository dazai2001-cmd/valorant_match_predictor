from __future__ import annotations

import hashlib
import json
import threading
from itertools import combinations
from pathlib import Path

import pandas as pd

from ..config import (
    MATCHES_CSV,
    NEWS_CSV,
    ROSTERS_CSV,
    TEAM_MODEL_PATH,
    TEAM_RANKINGS_CSV,
    TEAM_RANKINGS_META_PATH,
    TEAMS_CSV,
)
from ..data_quality import enrich_match_metadata, filter_training_ready_matches
from ..features.form_calculations import (
    build_player_profiles,
    build_player_profiles_from_rosters,
    calculate_team_form_from_rows,
    clean_match_data,
    filter_curated_competition_history,
    probable_lineup,
)
from ..features.team_context import (
    build_sequential_team_context,
    prepare_team_state_for_prediction,
    team_context_from_prepared_state,
)
from ..team_registry import (
    VCT_TIER1_SEASON,
    active_tier1_registry,
    filter_registry_tier1_matchups,
    load_team_registry,
    registry_team_pages,
)
from ..training.model_inference import load_model_payload, score_team_win_probability
from ..training.train_models import (
    TEAM_FEATURES,
    dataset_fingerprint,
    player_stack_features_from_summaries,
    reverse_team_feature_row,
    team_feature_row,
    team_rows_from_cleaned,
)
from ..vlr_client import canonicalize_match_dataframe
from .predict_match import lineup_identities, summarize_team
from .predict_player_ratings import apply_news, load_csv


RANKING_VERSION = "model-round-robin-v1"
RANKING_REGIONS = ("VCT Americas", "VCT EMEA", "VCT Pacific", "VCT China")
_CACHE_LOCK = threading.RLock()


def _source_signature() -> str:
    digest = hashlib.sha256(RANKING_VERSION.encode("utf-8"))
    for source in [
        MATCHES_CSV,
        ROSTERS_CSV,
        NEWS_CSV,
        TEAMS_CSV,
        TEAM_MODEL_PATH,
    ]:
        path = Path(source)
        digest.update(path.name.encode("utf-8"))
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


def _read_cached_rankings(signature: str, allow_stale: bool = False) -> dict | None:
    try:
        meta = json.loads(Path(TEAM_RANKINGS_META_PATH).read_text(encoding="utf-8"))
        rankings = pd.read_csv(TEAM_RANKINGS_CSV, low_memory=False)
    except (FileNotFoundError, OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
        return None
    if rankings.empty or (not allow_stale and meta.get("source_signature") != signature):
        return None
    return {**meta, "rows": rankings.to_dict("records"), "stale": meta.get("source_signature") != signature}


def _write_rankings_cache(rankings: pd.DataFrame, meta: dict) -> None:
    csv_path = Path(TEAM_RANKINGS_CSV)
    meta_path = Path(TEAM_RANKINGS_META_PATH)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = csv_path.with_suffix(".csv.tmp")
    meta_temp = meta_path.with_suffix(".json.tmp")
    rankings.to_csv(csv_temp, index=False)
    meta_temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    csv_temp.replace(csv_path)
    meta_temp.replace(meta_path)


def aggregate_pairwise_rankings(
    team_regions: dict[str, str],
    matchup_probabilities: list[dict],
    team_details: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Average each team's neutral-series probabilities against the full field."""
    team_details = team_details or {}
    probabilities = {team: [] for team in team_regions}
    for matchup in matchup_probabilities:
        team = matchup["team"]
        opponent = matchup["opponent"]
        probability = min(1.0, max(0.0, float(matchup["probability"])))
        probabilities.setdefault(team, []).append(probability)
        probabilities.setdefault(opponent, []).append(1.0 - probability)

    rows = []
    for team, region in team_regions.items():
        values = probabilities.get(team, [])
        detail = team_details.get(team, {})
        rows.append(
            {
                "team": team,
                "region": region,
                "power_score": 100.0 * sum(values) / len(values) if values else 50.0,
                "favored_against": sum(value > 0.5 for value in values),
                "opponents_rated": len(values),
                "recent_win_rate": float(detail.get("recent_win_rate", 0.5)),
                "lineup_certainty": float(detail.get("lineup_certainty", 0.0)),
                "evidence": float(detail.get("evidence", 0.0)),
                "matches": int(detail.get("matches", 0)),
            }
        )

    output = pd.DataFrame(rows).sort_values(
        ["power_score", "evidence", "team"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    output["global_rank"] = range(1, len(output) + 1)
    output["regional_rank"] = (
        output.groupby("region", sort=False).cumcount() + 1
    )
    return output[
        [
            "global_rank",
            "regional_rank",
            "team",
            "region",
            "power_score",
            "favored_against",
            "opponents_rated",
            "recent_win_rate",
            "lineup_certainty",
            "evidence",
            "matches",
        ]
    ]


def _neutral_team_features() -> dict:
    features = {name: 0.0 for name in TEAM_FEATURES}
    features["h2h_win_rate"] = 0.5
    features["elo_probability"] = 0.5
    return features


def _ranking_profiles(
    matches: pd.DataFrame,
    teams: list[str],
    rosters: pd.DataFrame,
) -> pd.DataFrame:
    if not rosters.empty:
        profiles = build_player_profiles_from_rosters(
            matches,
            teams,
            rosters,
            recent_maps=10,
        )
    else:
        profiles = build_player_profiles(matches, teams, recent_maps=10)
    if profiles.empty:
        return pd.DataFrame(columns=["team", "player", "predicted_rating"])
    return apply_news(profiles, load_csv(NEWS_CSV), recent_days=45)


def build_team_rankings(
    season_year: int = VCT_TIER1_SEASON,
    source_signature: str | None = None,
    progress=None,
) -> dict:
    registry = load_team_registry()
    active_registry = active_tier1_registry(season_year=season_year)
    if active_registry.empty:
        raise ValueError(f"No active Tier 1 teams are configured for {season_year}.")
    active_registry = active_registry.drop_duplicates("team", keep="last")
    team_regions = {
        str(row["team"]): str(row.get("region") or row.get("league") or "Unassigned")
        for _, row in active_registry.iterrows()
    }
    teams = sorted(team_regions, key=str.casefold)

    raw_matches = canonicalize_match_dataframe(
        load_csv(MATCHES_CSV),
        registry_team_pages(season_year=None),
    )
    raw_matches = enrich_match_metadata(
        raw_matches,
        registry=registry,
        rosters=load_csv(ROSTERS_CSV),
    )
    raw_matches = filter_curated_competition_history(
        filter_registry_tier1_matchups(raw_matches)
    )
    raw_matches = filter_training_ready_matches(raw_matches)
    if raw_matches.empty:
        raise ValueError("No training-ready match history is available for rankings.")
    fingerprint = dataset_fingerprint(raw_matches)
    matches = clean_match_data(raw_matches)
    if matches["match_date_sort"].notna().any():
        matches = matches[matches["match_date_sort"].dt.year <= season_year].copy()
    team_maps = team_rows_from_cleaned(matches)
    if team_maps.empty:
        raise ValueError("No team-level match history is available for rankings.")

    profiles = _ranking_profiles(matches, teams, load_csv(ROSTERS_CSV))
    summaries = {
        team: summarize_team(
            team,
            profiles,
            matches,
            team_form=calculate_team_form_from_rows(team_maps, team),
        )
        for team in teams
    }
    lineups = {
        team: lineup_identities(probable_lineup(profiles, team, lineup_size=5))
        for team in teams
    }

    model_payload = load_model_payload(TEAM_MODEL_PATH)
    model_metadata = (model_payload or {}).get("metadata", {})
    if (
        model_payload is not None
        and model_payload.get("sequential_state") is not None
        and model_metadata.get("dataset_fingerprint") == fingerprint
    ):
        sequential_state = model_payload["sequential_state"]
    else:
        _, sequential_state = build_sequential_team_context(matches, team_maps)

    latest_match = team_maps["match_date_sort"].dropna().max()
    now = pd.Timestamp.now(tz="UTC")
    as_of_date = max(now, latest_match + pd.Timedelta(days=1)) if pd.notna(latest_match) else now
    prepared_state = prepare_team_state_for_prediction(
        sequential_state,
        lineups_by_team=lineups,
        as_of_date=as_of_date,
    )
    numeric_match_ids = pd.to_numeric(team_maps.get("match_id"), errors="coerce")
    target = pd.Series(
        {
            "match_date_sort": as_of_date,
            "match_id": numeric_match_ids.max() + 1 if numeric_match_ids.notna().any() else None,
        }
    )

    if model_payload is None:
        model_payload = {
            "features": TEAM_FEATURES,
            "model_kind": "baseline",
            "metadata": {"team_metrics": {"selected_candidate": "elo_baseline"}},
        }

    matchup_probabilities = []
    pairings = list(combinations(teams, 2))
    for index, (team, opponent) in enumerate(pairings, start=1):
        features = team_feature_row(team_maps, target, team, opponent)
        reverse_features = team_feature_row(team_maps, target, opponent, team)
        features = dict(features or _neutral_team_features())
        reverse_features = dict(reverse_features or _neutral_team_features())

        context = team_context_from_prepared_state(
            prepared_state,
            team,
            opponent,
            lineup_a=lineups.get(team, set()),
            lineup_b=lineups.get(opponent, set()),
            as_of_date=as_of_date,
        )
        map_probabilities = context.pop("map_probabilities", {})
        player_context = player_stack_features_from_summaries(
            summaries[team],
            summaries[opponent],
        )
        features.update(context)
        features.update(player_context)
        reverse_features.update(
            reverse_team_feature_row({**context, **player_context})
        )
        probability, _ = score_team_win_probability(
            model_payload,
            features,
            reverse_features=reverse_features,
            map_probabilities=map_probabilities,
            best_of=3,
        )
        matchup_probabilities.append(
            {"team": team, "opponent": opponent, "probability": probability}
        )
        if progress and (index == 1 or index % 25 == 0 or index == len(pairings)):
            progress(
                "rankings",
                index,
                len(pairings),
                f"Scoring neutral Bo3 matchup {index} of {len(pairings)}",
            )

    team_details = {}
    for team in teams:
        summary = summaries[team]
        history = team_maps[team_maps["team"] == team]
        match_count = int(history["match_key"].nunique())
        history_reliability = min(1.0, match_count / 20.0)
        freshness = float(prepared_state["elo_freshness"].get(team, 0.0))
        lineup_certainty = float(summary.get("lineup_certainty", 0.0))
        player_reliability = float(summary.get("data_reliability", 0.0))
        model_reliability = float(
            model_payload.get("metadata", {})
            .get("team_metrics", {})
            .get("model_reliability", 0.0)
        )
        evidence = (
            0.35 * history_reliability
            + 0.20 * freshness
            + 0.20 * lineup_certainty
            + 0.15 * player_reliability
            + 0.10 * model_reliability
        )
        team_details[team] = {
            "recent_win_rate": summary.get("recent_win_rate", 0.5),
            "lineup_certainty": lineup_certainty,
            "evidence": min(1.0, max(0.0, evidence)),
            "matches": match_count,
        }

    rankings = aggregate_pairwise_rankings(
        team_regions,
        matchup_probabilities,
        team_details=team_details,
    )
    metrics = model_payload.get("metadata", {}).get("team_metrics", {})
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    meta = {
        "source_signature": source_signature or _source_signature(),
        "ranking_version": RANKING_VERSION,
        "season_year": season_year,
        "generated_at": generated_at,
        "as_of_date": as_of_date.isoformat(),
        "team_count": len(rankings),
        "matchup_count": len(matchup_probabilities),
        "regions": list(RANKING_REGIONS),
        "model_candidate": metrics.get("selected_candidate", "elo_baseline"),
        "method": "Average neutral Bo3 win probability against every active Tier 1 team",
        "stale": False,
    }
    _write_rankings_cache(rankings, meta)
    return {**meta, "rows": rankings.to_dict("records")}


def team_rankings_payload(
    force: bool = False,
    allow_stale: bool = False,
    season_year: int = VCT_TIER1_SEASON,
) -> dict:
    signature = _source_signature()
    with _CACHE_LOCK:
        if not force:
            cached = _read_cached_rankings(signature, allow_stale=allow_stale)
            if cached is not None:
                return cached
        return build_team_rankings(
            season_year=season_year,
            source_signature=signature,
        )
