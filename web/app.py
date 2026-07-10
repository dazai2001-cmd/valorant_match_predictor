from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from valorant_predictor.config import (
    MATCHES_CSV,
    MATCH_PREDICTION_CSV,
    NEWS_CSV,
    PREDICTIONS_CSV,
    ROSTERS_CSV,
)
from valorant_predictor.prediction.predict_match import predict_match
from valorant_predictor.prediction.predict_player_ratings import load_csv
from valorant_predictor.team_registry import (
    VCT_TIER1_SEASON,
    active_tier1_registry,
    active_tier1_team_names,
    load_team_registry,
    registry_team_pages,
    seed_vct_tier1_teams,
)
from valorant_predictor.training.train_models import METRICS_PATH, train_models
from valorant_predictor.vlr_client import scrape_matches, scrape_news, scrape_rosters


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


def round_records(df: pd.DataFrame, digits: int = 3) -> list[dict]:
    if df.empty:
        return []
    output = df.copy()
    for col in output.select_dtypes(include=["float", "float64"]).columns:
        output[col] = output[col].round(digits)
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
    return pd.read_json(path, typ="series").to_dict()


def base_context(**extra) -> dict:
    teams = available_teams()
    default_team1 = request.form.get("team1") or (teams[0] if teams else "")
    default_team2 = request.form.get("team2") or ("Sentinels" if "Sentinels" in teams else (teams[1] if len(teams) > 1 else ""))
    context = {
        "teams": teams,
        "selected": {
            "team1": default_team1,
            "team2": default_team2,
            "best_of": request.form.get("best_of", "3"),
            "season_year": request.form.get("season_year", "2026"),
            "recent_days": request.form.get("recent_days", ""),
            "recent_maps": request.form.get("recent_maps", "10"),
            "recent_news_days": request.form.get("recent_news_days", "45"),
            "limit_per_team": request.form.get("limit_per_team", "75"),
            "news_pages": request.form.get("news_pages", "3"),
            "use_trained_models": bool_form("use_trained_models"),
            "refresh_matches": bool_form("refresh_matches"),
            "refresh_news": bool_form("refresh_news"),
            "refresh_rosters": bool_form("refresh_rosters"),
        },
        "last_prediction": load_last_prediction(),
        "metrics": metrics_payload(),
        "team_registry": round_records(active_tier1_registry()),
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
        recent_days = int_form("recent_days")
        recent_maps = int_form("recent_maps", 10) or 10
        recent_news_days = int_form("recent_news_days", 45) or 45
        best_of = int_form("best_of", 3) or 3

        if team1 == team2:
            raise ValueError("Choose two different teams.")

        if bool_form("refresh_matches"):
            scrape_matches(
                output_csv=MATCHES_CSV,
                limit_per_team=int_form("limit_per_team", 75) or 75,
                team_pages=registry_team_pages(),
                season_year=season_year,
                recent_days=recent_days,
            )
        if bool_form("refresh_news"):
            scrape_news(output_csv=NEWS_CSV, pages=int_form("news_pages", 3) or 3)
        if bool_form("refresh_rosters"):
            scrape_rosters(output_csv=ROSTERS_CSV, team_pages=registry_team_pages())

        summary, players, team_summaries = predict_match(
            team1=team1,
            team2=team2,
            best_of=best_of,
            recent_maps=recent_maps,
            recent_news_days=recent_news_days,
            season_year=season_year,
            recent_days=recent_days,
            use_trained_models=bool_form("use_trained_models"),
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
    try:
        season_year = int_form("train_season_year", 2026)
        recent_days = int_form("train_recent_days", 60) or 60
        limit_per_team = int_form("train_limit_per_team", 75) or 75
        fallback_years = int_form("fallback_years", 1) or 1
        min_history = int_form("min_player_history_maps", 3) or 3

        if bool_form("train_refresh_matches"):
            scrape_matches(
                output_csv=MATCHES_CSV,
                limit_per_team=limit_per_team,
                team_pages=registry_team_pages(),
                season_year=season_year,
            )

        metrics = train_models(
            matches_csv=MATCHES_CSV,
            season_year=season_year,
            recent_days=recent_days,
            fallback_years=fallback_years,
            min_player_history_maps=min_history,
        )
        return render_template("index.html", **base_context(train_result=metrics, active_tab="train"))
    except Exception as exc:
        return render_template("index.html", **base_context(error=str(exc), active_tab="train")), 400


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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
