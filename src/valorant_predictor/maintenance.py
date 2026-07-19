from __future__ import annotations

import json
from pathlib import Path

from .config import (
    DATA_QUALITY_PATH,
    MATCHES_CSV,
    NEWS_CSV,
    ROSTERS_CSV,
    UPCOMING_MATCHES_CSV,
)
from .data_quality import data_quality_report, enrich_match_metadata
from .storage import sync_match_data
from .team_registry import (
    VCT_TIER1_SEASON,
    load_team_registry,
    registry_team_pages,
    seed_vct_tier1_teams,
)
from .training.train_models import train_models
from .vct_events import import_vct_season
from .vlr_client import (
    scrape_matches,
    scrape_news,
    scrape_rosters,
    scrape_upcoming_matches,
)


UPDATE_STAGES = 6
STAGE_UNITS = 1000


def _stage_progress(progress, stage_index: int):
    def report(step: str, current: int, total: int, message: str) -> None:
        fraction = max(0.0, min(1.0, float(current) / float(total))) if total else 0.0
        progress(
            step,
            stage_index * STAGE_UNITS + round(fraction * STAGE_UNITS),
            UPDATE_STAGES * STAGE_UNITS,
            message,
        )

    return report


def _stage_boundary(progress, stage_index: int, step: str, message: str) -> None:
    progress(
        step,
        stage_index * STAGE_UNITS,
        UPDATE_STAGES * STAGE_UNITS,
        message,
    )


def update_database_and_model(
    progress,
    history_season: int = 2025,
    model_selection: dict | None = None,
) -> dict:
    warnings = []
    _stage_boundary(
        progress,
        0,
        "vct_history",
        f"Updating official {history_season} VCT events",
    )
    matches, history_summary = import_vct_season(
        season_year=history_season,
        output_csv=MATCHES_CSV,
        progress=_stage_progress(progress, 0),
    )

    _stage_boundary(
        progress,
        1,
        "current_matches",
        f"Discovering new {VCT_TIER1_SEASON} Tier 1 results",
    )
    seed_vct_tier1_teams(resolve_missing=True)
    matches = scrape_matches(
        output_csv=MATCHES_CSV,
        limit_per_team=None,
        team_pages=registry_team_pages(season_year=VCT_TIER1_SEASON),
        season_year=VCT_TIER1_SEASON,
        min_matches_per_team=20,
    )
    current_summary = dict(matches.attrs.get("scrape_summary", {}))

    _stage_boundary(progress, 2, "rosters", "Updating current player rosters")
    rosters = scrape_rosters(
        output_csv=ROSTERS_CSV,
        team_pages=registry_team_pages(season_year=VCT_TIER1_SEASON),
    )
    roster_summary = dict(rosters.attrs.get("scrape_summary", {}))
    matches = enrich_match_metadata(
        matches,
        registry=load_team_registry(),
        rosters=rosters,
    )
    matches.to_csv(MATCHES_CSV, index=False)
    sync_match_data(matches, source=str(MATCHES_CSV))
    quality = data_quality_report(matches)
    Path(DATA_QUALITY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DATA_QUALITY_PATH).write_text(json.dumps({"source": quality}, indent=2), encoding="utf-8")

    _stage_boundary(progress, 3, "news", "Updating VLR roster and availability news")
    try:
        news = scrape_news(output_csv=NEWS_CSV, pages=3)
        news_rows = len(news)
    except Exception as exc:
        news_rows = 0
        warnings.append(f"News update failed: {exc}")

    _stage_boundary(progress, 4, "schedule", "Updating the upcoming Tier 1 schedule")
    try:
        upcoming = scrape_upcoming_matches(
            output_csv=UPCOMING_MATCHES_CSV,
            team_pages=registry_team_pages(season_year=VCT_TIER1_SEASON),
        )
        upcoming_rows = len(upcoming)
    except Exception as exc:
        upcoming_rows = 0
        warnings.append(f"Schedule update failed: {exc}")

    _stage_boundary(progress, 5, "training", "Training and evaluating candidate models")
    metrics = train_models(
        matches_csv=MATCHES_CSV,
        season_year=VCT_TIER1_SEASON,
        recent_days=60,
        fallback_years=None,
        min_player_history_maps=1,
        min_team_matches=20,
        force=False,
        model_selection=model_selection,
    )
    missing_history = int(history_summary.get("missing_matches", 0)) - int(
        history_summary.get("parsed_matches", 0)
    )
    message = "Database and model update completed."
    if missing_history > 0:
        message += f" {missing_history} historical matches still need player-map rows."
    if warnings:
        message += f" {len(warnings)} optional source update(s) failed."
    return {
        "message": message,
        "history": history_summary,
        "current": current_summary,
        "roster_rows": len(rosters),
        "roster_summary": roster_summary,
        "news_rows": news_rows,
        "upcoming_rows": upcoming_rows,
        "player_map_rows": len(matches),
        "data_quality": quality,
        "training_status": metrics.get("training_status"),
        "model_version": metrics.get("model_version"),
        "model_selection": metrics.get("model_selection", {}),
        "warnings": warnings,
    }


def train_and_evaluate(
    progress,
    model_selection: dict | None = None,
) -> dict:
    progress("training", 0, 1, "Training and evaluating candidate models")
    metrics = train_models(
        matches_csv=MATCHES_CSV,
        season_year=VCT_TIER1_SEASON,
        recent_days=60,
        fallback_years=None,
        min_player_history_maps=1,
        min_team_matches=20,
        force=True,
        model_selection=model_selection,
    )
    return {
        "message": "Training and held-out evaluation completed.",
        "training_status": metrics.get("training_status"),
        "model_version": metrics.get("model_version"),
        "model_selection": metrics.get("model_selection", {}),
        "player_metrics": metrics.get("player_metrics", {}),
        "team_metrics": metrics.get("team_metrics", {}),
        "map_metrics": metrics.get("map_metrics", {}),
    }
