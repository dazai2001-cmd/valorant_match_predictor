from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from valorant_predictor.config import (
    MATCH_COVERAGE_CSV,
    MATCHES_CSV,
    MATCH_PREDICTION_CSV,
    NEWS_CSV,
    PREDICTIONS_CSV,
    ROSTERS_CSV,
    UPCOMING_MATCHES_CSV,
)
from valorant_predictor.prediction.predict_match import predict_match
from valorant_predictor.prediction.predict_player_ratings import load_csv
from valorant_predictor.prediction.team_rankings import team_rankings_payload
from valorant_predictor.jobs import job_manager
from valorant_predictor.maintenance import train_and_evaluate, update_database_and_model
from valorant_predictor.model_selection import (
    MODEL_CHOICES,
    load_model_selection,
    normalize_model_selection,
)
from valorant_predictor.team_registry import (
    VCT_TIER1_SEASON,
    active_tier1_team_names,
    load_team_registry,
    registry_team_pages,
    seed_vct_tier1_teams,
)
from valorant_predictor.training.train_models import METRICS_PATH, train_models
from valorant_predictor.vlr_client import (
    match_coverage_report,
    scrape_matches,
    scrape_news,
    scrape_rosters,
    scrape_upcoming_matches,
)


app = Flask(__name__)


def bool_form(name: str) -> bool:
    return request.form.get(name) == "on"


def int_form(name: str, default: int | None = None) -> int | None:
    value = request.form.get(name, "").strip()
    if not value:
        return default
    return int(value)


def available_teams() -> list[str]:
    teams = active_tier1_team_names()
    if teams:
        return teams

    registry = load_team_registry()
    if not registry.empty:
        return sorted(registry["team"].dropna().astype(str).unique())
    return []


def available_maps() -> list[str]:
    matches = load_csv(MATCHES_CSV)
    if matches.empty or "map_name" not in matches.columns:
        return []
    scoped = matches[[column for column in ["match_id", "map_id", "map_name"] if column in matches]].copy()
    scoped["map_name"] = scoped["map_name"].fillna("").astype(str).str.strip()
    scoped = scoped[scoped["map_name"] != ""]
    scoped = scoped.drop_duplicates(
        subset=[column for column in ["match_id", "map_id"] if column in scoped.columns]
    )
    return sorted(scoped["map_name"].unique(), key=str.casefold)


def round_records(df: pd.DataFrame, digits: int = 3) -> list[dict]:
    if df.empty:
        return []
    output = df.copy()
    for col in output.select_dtypes(include=["float", "float64"]).columns:
        output[col] = output[col].round(digits)
    output = output.astype(object).where(pd.notna(output), None)
    return output.to_dict("records")


def load_last_prediction() -> dict:
    summary_df = load_csv(MATCH_PREDICTION_CSV)
    players_df = load_csv(PREDICTIONS_CSV)
    if summary_df.empty:
        return {}

    summary = summary_df.iloc[-1].to_dict()
    team_rows = []
    for prefix in ["team1", "team2"]:
        name = summary.get(prefix)
        if not name:
            continue
        team_rows.append(
            {
                "team": name,
                "strength": summary.get(f"{prefix}_strength"),
                "avg_player_rating": summary.get(f"{prefix}_avg_player_rating"),
                "recent_win_rate": summary.get(f"{prefix}_recent_win_rate"),
                "avg_news_adjustment": summary.get(f"{prefix}_avg_news_adjustment"),
                "avg_roster_uncertainty": summary.get(f"{prefix}_avg_roster_uncertainty"),
                "lineup_reliability": summary.get(f"{prefix}_lineup_reliability"),
                "data_reliability": summary.get(f"{prefix}_data_reliability"),
                "lineup_certainty": summary.get(f"{prefix}_lineup_certainty"),
                "roster_continuity": summary.get(f"{prefix}_roster_continuity"),
                "map_pool_edge": summary.get(f"{prefix}_map_pool_edge"),
                "elo_edge": summary.get(f"{prefix}_elo_edge"),
                "avg_player_uncertainty": summary.get(f"{prefix}_avg_player_uncertainty"),
            }
        )

    return {
        "summary": summary,
        "team_rows": team_rows,
        "players": round_records(players_df),
    }


def metrics_payload() -> dict:
    path = Path(METRICS_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def requested_model_selection() -> dict[str, str]:
    return normalize_model_selection(
        {
            "player": request.form.get("player_model", "auto"),
            "team": request.form.get("team_model", "auto"),
            "map": request.form.get("map_model", "auto"),
        }
    )


def model_reports(metrics: dict) -> list[dict]:
    def first_recorded(*values):
        return next((value for value in values if value is not None), None)

    def improvement(metric, baseline):
        if metric is None or baseline in (None, 0):
            return None
        return (float(baseline) - float(metric)) / float(baseline)

    reports = []
    definitions = [
        ("player", "Player rating", "MAE", "mae", "baseline_mae", "last_10_baseline"),
        ("team", "Match winner", "Log loss", "log_loss", "elo_log_loss", "elo_baseline"),
        ("map", "Map winner", "Log loss", "log_loss", "baseline_log_loss", "map_form_baseline"),
    ]
    for task, title, metric_label, test_key, baseline_key, baseline_id in definitions:
        task_metrics = metrics.get(f"{task}_metrics", {}) if metrics else {}
        candidates = task_metrics.get("candidate_calibration", {})
        rows = []
        reference_values = next(iter(candidates.values()), {})
        if task == "player":
            validation_key = "mae"
            rolling_key = "rolling_mae_mean"
            rolling_std_key = "rolling_mae_std"
            baseline_validation = first_recorded(
                task_metrics.get("validation_baseline_mae"),
                reference_values.get("validation_baseline_mae"),
            )
            baseline_rolling = first_recorded(
                reference_values.get("rolling_baseline_mae_mean"),
                task_metrics.get("rolling_baseline_mae_mean"),
            )
            baseline_rolling_std = first_recorded(
                reference_values.get("rolling_baseline_mae_std"),
                task_metrics.get("rolling_baseline_mae_std"),
            )
            baseline_validation_accuracy = None
            baseline_rolling_accuracy = None
            baseline_validation_r2 = reference_values.get(
                "validation_baseline_r2"
            )
            baseline_validation_brier = None
            baseline_rolling_accuracy_std = None
            baseline_rolling_brier = None
            baseline_rolling_brier_std = None
            baseline_test_r2 = task_metrics.get("baseline_r2")
            baseline_test_brier = None
        else:
            validation_key = "log_loss"
            rolling_key = "rolling_log_loss_mean"
            rolling_std_key = "rolling_log_loss_std"
            baseline_validation = first_recorded(
                reference_values.get("validation_baseline_log_loss"),
                task_metrics.get("latest_development_reference_log_loss")
                if task == "team"
                else None,
            )
            baseline_rolling = first_recorded(
                reference_values.get("rolling_baseline_log_loss_mean"),
                reference_values.get("rolling_reference_log_loss_mean"),
                task_metrics.get("rolling_reference_log_loss_mean"),
            )
            baseline_rolling_std = reference_values.get(
                "rolling_baseline_log_loss_std"
            )
            baseline_validation_accuracy = reference_values.get(
                "validation_baseline_accuracy"
            )
            baseline_rolling_accuracy = reference_values.get(
                "rolling_baseline_accuracy_mean"
            )
            baseline_validation_r2 = None
            baseline_validation_brier = reference_values.get(
                "validation_baseline_brier"
            )
            baseline_rolling_accuracy_std = reference_values.get(
                "rolling_baseline_accuracy_std"
            )
            baseline_rolling_brier = reference_values.get(
                "rolling_baseline_brier_mean"
            )
            baseline_rolling_brier_std = reference_values.get(
                "rolling_baseline_brier_std"
            )
            baseline_test_r2 = None
            baseline_test_brier = task_metrics.get(
                "elo_brier_score" if task == "team" else "baseline_brier_score"
            )
        baseline_test = task_metrics.get(baseline_key)
        rows.append(
            {
                "id": baseline_id,
                "name": MODEL_CHOICES[task].get(baseline_id, baseline_id),
                "validation_metric": baseline_validation,
                "validation_accuracy": baseline_validation_accuracy,
                "validation_r2": baseline_validation_r2,
                "validation_brier": baseline_validation_brier,
                "rolling_metric": baseline_rolling,
                "rolling_variation": baseline_rolling_std,
                "rolling_accuracy": baseline_rolling_accuracy,
                "rolling_accuracy_variation": baseline_rolling_accuracy_std,
                "rolling_brier": baseline_rolling_brier,
                "rolling_brier_variation": baseline_rolling_brier_std,
                "rolling_folds": None,
                "rolling_wins": None,
                "improvement_vs_baseline": 0.0,
                "selection_metric": None,
                "test_metric": baseline_test,
                "test_accuracy": task_metrics.get(
                    "elo_accuracy" if task == "team" else "baseline_accuracy"
                ),
                "test_r2": baseline_test_r2,
                "test_brier": baseline_test_brier,
            }
        )
        for candidate_id, values in candidates.items():
            evaluated = candidate_id == task_metrics.get("selected_candidate")
            rolling_metric = values.get(rolling_key)
            candidate_baseline = first_recorded(
                values.get(
                    "rolling_baseline_mae_mean"
                    if task == "player"
                    else "rolling_baseline_log_loss_mean"
                ),
                values.get(
                    "rolling_baseline_mae_mean"
                    if task == "player"
                    else "rolling_reference_log_loss_mean"
                ),
                baseline_rolling,
            )
            rows.append(
                {
                    "id": candidate_id,
                    "name": MODEL_CHOICES[task].get(candidate_id, candidate_id),
                    "validation_metric": values.get(validation_key),
                    "validation_accuracy": values.get("accuracy"),
                    "validation_r2": values.get("r2"),
                    "validation_brier": values.get("brier_score"),
                    "rolling_metric": rolling_metric,
                    "rolling_variation": values.get(rolling_std_key),
                    "rolling_accuracy": values.get("rolling_accuracy_mean"),
                    "rolling_accuracy_variation": values.get(
                        "rolling_accuracy_std"
                    ),
                    "rolling_brier": values.get("rolling_brier_mean"),
                    "rolling_brier_variation": values.get("rolling_brier_std"),
                    "rolling_folds": values.get("rolling_folds"),
                    "rolling_wins": values.get("rolling_wins"),
                    "improvement_vs_baseline": improvement(
                        rolling_metric,
                        candidate_baseline,
                    ),
                    "selection_metric": (
                        rolling_metric
                        if task == "player"
                        else values.get("selection_log_loss")
                    ),
                    "test_metric": first_recorded(
                        values.get(
                            "test_mae" if task == "player" else "test_log_loss"
                        ),
                        task_metrics.get(test_key) if evaluated else None,
                    ),
                    "test_accuracy": first_recorded(
                        values.get("test_accuracy"),
                        task_metrics.get("accuracy")
                        if evaluated and task != "player"
                        else None,
                    ),
                    "test_r2": first_recorded(
                        values.get("test_r2"),
                        task_metrics.get("r2") if evaluated and task == "player" else None,
                    ),
                    "test_brier": first_recorded(
                        values.get("test_brier_score"),
                        task_metrics.get("brier_score")
                        if evaluated and task != "player"
                        else None,
                    ),
                }
            )
        evaluated_id = task_metrics.get("selected_candidate", "")
        recommended_id = task_metrics.get("recommended_candidate", evaluated_id)
        active_for_predictions = task_metrics.get(
            "active_for_predictions",
            task_metrics.get("enabled", True),
        )
        if task == "player":
            active_id = (
                baseline_id
                if evaluated_id == baseline_id
                or float(task_metrics.get("correction_weight", 0.0) or 0.0) <= 0.0
                else evaluated_id
            )
        else:
            active_id = evaluated_id if active_for_predictions else baseline_id
        passed_safeguard = bool(
            task_metrics.get(
                "enabled",
                task == "player"
                and float(task_metrics.get("skill_vs_baseline", 0.0) or 0.0) > 0.0,
            )
        )
        reports.append(
            {
                "task": task,
                "title": title,
                "metric_label": metric_label,
                "rows": rows,
                "active": active_id,
                "evaluated": evaluated_id,
                "recommended": recommended_id,
                "selection_mode": task_metrics.get("selection_mode", "auto"),
                "passed_safeguard": passed_safeguard,
                "candidate_rejected": bool(
                    evaluated_id
                    and active_id != evaluated_id
                    and task_metrics.get("selection_mode", "auto") == "auto"
                ),
                "test_rows": task_metrics.get("test_rows"),
                "test_matches": task_metrics.get("test_matches"),
                "test_start": task_metrics.get("test_start", ""),
                "interval_coverage": task_metrics.get("interval_test_coverage") if task == "player" else None,
                "skill_vs_baseline": task_metrics.get("skill_vs_baseline") if task == "player" else None,
                "brier_score": task_metrics.get("brier_score") if task != "player" else None,
                "reliability": task_metrics.get("model_reliability"),
                "model_blend_weight": task_metrics.get("model_blend_weight") if task != "player" else None,
                "probability_temperature": task_metrics.get("probability_temperature") if task != "player" else None,
                "max_model_delta": task_metrics.get("max_model_delta") if task != "player" else None,
                "probability_structure": task_metrics.get("probability_structure", ""),
                "calibration_method": task_metrics.get("calibration", ""),
                "feature_set": task_metrics.get("selected_feature_set", ""),
                "stacked_accuracy_delta": task_metrics.get("stacked_holdout_accuracy_delta"),
                "stacked_log_loss_delta": task_metrics.get("stacked_holdout_log_loss_delta"),
                "oof_coverage": (
                    task_metrics.get("player_oof_stack_coverage")
                    if task == "team"
                    else task_metrics.get("team_anchor_oof_coverage")
                    if task == "map"
                    else task_metrics.get("oof_stacking", {}).get("coverage")
                ),
                "rolling_folds": task_metrics.get("rolling_backtest_folds"),
                "rolling_wins": task_metrics.get("rolling_wins"),
                "rolling_passed": task_metrics.get("rolling_safeguard_passed"),
                "latest_window_passed": task_metrics.get(
                    "latest_development_safeguard_passed"
                ),
                "bootstrap": task_metrics.get("bootstrap_95pct", {}),
                "selective_accuracy": task_metrics.get("selective_accuracy", []),
                "region_log_loss_delta": task_metrics.get(
                    "region_holdout_log_loss_delta"
                ),
                "ensemble": task_metrics.get("oof_probability_ensemble", {})
                if task == "team"
                else {},
                "history_scope": task_metrics.get("history_scope_experiment", {})
                if task == "team"
                else {},
                "active_architecture": task_metrics.get("active_architecture", ""),
            }
        )
    return reports


@lru_cache(maxsize=8)
def _computed_match_coverage(
    matches_mtime_ns: int,
    min_matches: int,
    season_year: int,
) -> pd.DataFrame:
    matches = load_csv(MATCHES_CSV)
    return match_coverage_report(
        matches,
        registry_team_pages(include_missing=True),
        min_matches_per_team=min_matches,
        season_year=season_year,
    )


def match_coverage_payload(
    min_matches: int = 20,
    season_year: int = VCT_TIER1_SEASON,
) -> dict:
    path = Path(MATCHES_CSV)
    matches_mtime_ns = path.stat().st_mtime_ns if path.exists() else 0
    coverage = _computed_match_coverage(matches_mtime_ns, min_matches, season_year)
    if coverage.empty:
        coverage = load_csv(MATCH_COVERAGE_CSV)
    if coverage.empty:
        return {}

    below = coverage[coverage["status"].astype(str) != "ok"].copy()
    return {
        "target": min_matches,
        "season_year": season_year,
        "teams": len(coverage),
        "teams_at_target": int((coverage["status"].astype(str) == "ok").sum()),
        "teams_below_target": len(below),
        "teams_with_legacy_data": int((coverage.get("legacy_matches", 0) > 0).sum())
        if "legacy_matches" in coverage.columns
        else 0,
        "lowest": round_records(below.sort_values(["matches", "team"]).head(8)),
    }


def roster_coverage_payload(target_players: int = 5) -> dict:
    teams = available_teams()
    if not teams:
        return {}

    rosters = load_csv(ROSTERS_CSV)
    counts = {team: 0 for team in teams}
    if not rosters.empty and {"team", "player"}.issubset(rosters.columns):
        scoped = rosters[rosters["team"].isin(teams)].copy()
        if "is_staff" in scoped.columns:
            scoped = scoped[~scoped["is_staff"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if "is_active_player" in scoped.columns:
            scoped = scoped[scoped["is_active_player"].astype(str).str.lower().isin(["true", "1", "yes"])]

        grouped = scoped.groupby("team")["player"].nunique()
        counts.update({team: int(count) for team, count in grouped.items()})

    below = [
        {"team": team, "players": count, "target_players": target_players}
        for team, count in sorted(counts.items())
        if count < target_players
    ]
    return {
        "target_players": target_players,
        "teams": len(teams),
        "teams_at_target": sum(1 for count in counts.values() if count >= target_players),
        "teams_below_target": len(below),
        "lowest": below[:8],
    }


def upcoming_matches_payload(limit: int = 16) -> list[dict]:
    upcoming = load_csv(UPCOMING_MATCHES_CSV)
    if upcoming.empty:
        return []
    output = upcoming.copy()
    output["match_date_sort"] = pd.to_datetime(output.get("match_date"), utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    output = output[
        output["match_date_sort"].isna()
        | (output["match_date_sort"] >= now - pd.Timedelta(hours=4))
    ].sort_values("match_date_sort", na_position="last")
    output["display_date"] = output["match_date_sort"].dt.strftime("%a %d %b, %H:%M UTC")
    return round_records(output.head(limit))


def rankings_context(job_running: bool = False) -> dict:
    try:
        payload = team_rankings_payload(allow_stale=job_running)
        generated_at = pd.to_datetime(
            payload.get("generated_at"),
            utc=True,
            errors="coerce",
        )
        payload["generated_display"] = (
            generated_at.strftime("%d %b %Y, %H:%M UTC")
            if pd.notna(generated_at)
            else ""
        )
        return payload
    except Exception as exc:
        return {"rows": [], "regions": [], "error": str(exc)}


def base_context(**extra) -> dict:
    teams = available_teams()
    default_team1 = request.form.get("team1") or (teams[0] if teams else "")
    default_team2 = request.form.get("team2") or ("Sentinels" if "Sentinels" in teams else (teams[1] if len(teams) > 1 else ""))
    metrics = metrics_payload()
    job = job_manager.snapshot()
    job_running = job.get("status") in {"queued", "running"}
    context = {
        "teams": teams,
        "map_options": available_maps(),
        "selected": {
            "team1": default_team1,
            "team2": default_team2,
            "best_of": request.form.get("best_of", "3"),
            "season_year": request.form.get("season_year", "2026"),
            "recent_maps": request.form.get("recent_maps", "10"),
            "recent_news_days": request.form.get("recent_news_days", "45"),
            "maps": [request.form.get(f"map_{index}", "") for index in range(1, 6)],
            "map_picks": [
                request.form.get(f"map_pick_{index}", "")
                for index in range(1, 6)
            ],
        },
        "last_prediction": load_last_prediction(),
        "metrics": metrics,
        "model_choices": MODEL_CHOICES,
        "model_selection": load_model_selection(),
        "model_reports": model_reports(metrics),
        "data_quality": metrics.get("data_quality", {}) if metrics else {},
        "job": job,
        "job_running": job_running,
        "match_coverage": match_coverage_payload(),
        "roster_coverage": roster_coverage_payload(),
        "upcoming_matches": upcoming_matches_payload(),
        "rankings": rankings_context(job_running=job_running),
    }
    context.update(extra)
    return context


@app.get("/")
def index():
    return render_template("index.html", **base_context())


@app.post("/predict")
def predict_route():
    try:
        team1 = request.form["team1"]
        team2 = request.form["team2"]
        season_year = int_form("season_year")
        recent_maps = int_form("recent_maps", 10) or 10
        recent_news_days = int_form("recent_news_days", 45) or 45
        best_of = int_form("best_of", 3) or 3
        map_values = [request.form.get(f"map_{index}", "").strip() for index in range(1, best_of + 1)]
        selected_maps = [value for value in map_values if value]
        selected_picks = [
            request.form.get(f"map_pick_{index}", "").strip().lower()
            for index in range(1, best_of + 1)
        ]
        if selected_maps and len(selected_maps) != best_of:
            raise ValueError(f"Choose all {best_of} maps for a Bo{best_of}, or leave every map on Auto.")
        if len(set(selected_maps)) != len(selected_maps):
            raise ValueError("Each selected map must be unique.")
        if any(selected_picks) and not selected_maps:
            raise ValueError("Choose all maps before assigning pick ownership.")

        if team1 == team2:
            raise ValueError("Choose two different teams.")

        if bool_form("refresh_matches"):
            seed_vct_tier1_teams(resolve_missing=True)
            scrape_matches(
                output_csv=MATCHES_CSV,
                limit_per_team=None,
                team_pages=registry_team_pages(),
                season_year=season_year,
                min_matches_per_team=20,
            )
        if bool_form("refresh_news"):
            scrape_news(output_csv=NEWS_CSV, pages=int_form("news_pages", 3) or 3)
        if bool_form("refresh_rosters"):
            seed_vct_tier1_teams(resolve_missing=True)
            scrape_rosters(output_csv=ROSTERS_CSV, team_pages=registry_team_pages())

        summary, players, team_summaries = predict_match(
            team1=team1,
            team2=team2,
            best_of=best_of,
            recent_maps=recent_maps,
            recent_news_days=recent_news_days,
            season_year=season_year,
            recent_days=None,
            use_trained_models=bool_form("use_trained_models"),
            match_date=request.form.get("match_date") or None,
            event_name=request.form.get("event_name", ""),
            event_stage=request.form.get("event_stage", ""),
            current_patch=request.form.get("patch", ""),
            selected_maps=selected_maps,
            selected_picks=selected_picks,
        )
        result = {
            "summary": summary,
            "team_rows": round_records(team_summaries),
            "players": round_records(players),
        }
        return render_template("index.html", **base_context(result=result, active_tab="predict"))
    except Exception as exc:
        return render_template("index.html", **base_context(error=str(exc), active_tab="predict")), 400


@app.post("/train")
def train_route():
    selections = requested_model_selection()
    job_manager.start(
        "train_and_evaluate",
        lambda progress: train_and_evaluate(
            progress,
            model_selection=selections,
        )
    )
    return redirect(url_for("index"))


@app.post("/jobs/database-update")
def database_update_job_route():
    selections = requested_model_selection()
    job_manager.start(
        "database_update",
        lambda progress: update_database_and_model(
            progress,
            history_season=2025,
            model_selection=selections,
        ),
    )
    return redirect(url_for("index"))


@app.get("/api/job")
def job_status_route():
    return jsonify(job_manager.snapshot())


@app.post("/teams/sync")
def sync_teams_route():
    try:
        registry, meta = seed_vct_tier1_teams(season_year=VCT_TIER1_SEASON, resolve_missing=True)
        unresolved = meta.get("unresolved", [])
        message = (
            f"Seeded {meta['seeded_count']} VCT Tier 1 teams for {meta['season_year']}. "
            f"{meta['resolved_count']} have VLR match URLs."
        )
        if unresolved:
            message += f" {len(unresolved)} still need VLR ID resolution."
        if meta.get("resolution_error"):
            message += " VLR search was not reachable, so resolution stopped early."
        return render_template(
            "index.html",
            **base_context(success=message, team_seed_status=meta, active_tab="teams"),
        )
    except Exception as exc:
        return render_template("index.html", **base_context(error=str(exc), active_tab="teams")), 400


@app.post("/data/update/matches")
def update_matches_route():
    try:
        seed_vct_tier1_teams(resolve_missing=True)
        matches = scrape_matches(
            output_csv=MATCHES_CSV,
            limit_per_team=None,
            team_pages=registry_team_pages(),
            min_matches_per_team=20,
            season_year=VCT_TIER1_SEASON,
        )
        scrape_summary = matches.attrs.get("scrape_summary", {})
        coverage = match_coverage_report(
            matches,
            registry_team_pages(include_missing=True),
            min_matches_per_team=20,
            season_year=VCT_TIER1_SEASON,
        )
        teams_at_target = int((coverage["status"].astype(str) == "ok").sum()) if not coverage.empty else 0
        message = (
            f"Parsed {scrape_summary.get('parsed_matches', 0)} new matches and reused "
            f"{scrape_summary.get('reused_matches', 0)} existing matches. "
            f"Stored {len(matches)} player-map rows; {teams_at_target} / {len(coverage)} teams "
            f"have 20+ {VCT_TIER1_SEASON} games."
        )
        return render_template("index.html", **base_context(success=message, active_tab="teams"))
    except Exception as exc:
        return render_template("index.html", **base_context(error=str(exc), active_tab="teams")), 400


@app.post("/data/update/rosters")
def update_rosters_route():
    try:
        seed_vct_tier1_teams(resolve_missing=True)
        rosters = scrape_rosters(output_csv=ROSTERS_CSV, team_pages=registry_team_pages())
        coverage = roster_coverage_payload()
        message = (
            f"Updated roster/player names with {len(rosters)} rows. "
            f"{coverage.get('teams_at_target', 0)} / {coverage.get('teams', 0)} teams have "
            f"{coverage.get('target_players', 5)}+ active players."
        )
        return render_template("index.html", **base_context(success=message, active_tab="teams"))
    except Exception as exc:
        return render_template("index.html", **base_context(error=str(exc), active_tab="teams")), 400


@app.post("/data/update/upcoming")
def update_upcoming_route():
    try:
        upcoming = scrape_upcoming_matches(
            output_csv=UPCOMING_MATCHES_CSV,
            team_pages=registry_team_pages(),
        )
        message = f"Updated {len(upcoming)} upcoming Tier 1 matches from VLR."
        return render_template(
            "index.html",
            **base_context(success=message, active_tab="upcoming"),
        )
    except Exception as exc:
        return render_template(
            "index.html",
            **base_context(error=str(exc), active_tab="upcoming"),
        ), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
