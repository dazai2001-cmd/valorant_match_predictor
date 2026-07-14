import argparse
from datetime import datetime

import pandas as pd

from ..config import MATCHES_CSV, NEWS_CSV, PREDICTIONS_CSV, ROSTERS_CSV
from ..features.form_calculations import build_player_profiles as build_form_profiles
from ..features.form_calculations import build_player_profiles_from_rosters
from ..features.form_calculations import clean_match_data
from ..features.form_calculations import filter_curated_competition_history
from ..features.news_impact import player_news_adjustments
from ..team_registry import filter_registry_tier1_matchups, registry_team_pages
from ..training.model_inference import apply_player_model
from ..vlr_client import canonicalize_match_dataframe, scrape_matches, scrape_news, scrape_rosters


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    return clean_match_data(df)


def rating_from_stats(group: pd.DataFrame) -> dict:
    deaths = group["deaths"].replace(0, 1)
    kd_ratio = (group["kills"] / deaths).mean()
    avg_acs = group["acs"].mean()
    assists = group["assists"].mean()
    win_rate = group["is_winner"].astype(float).mean() if "is_winner" in group else 0.5

    acs_component = (avg_acs - 200.0) / 250.0
    kd_component = kd_ratio - 1.0
    assist_component = (assists - 5.0) / 35.0
    win_component = win_rate - 0.5

    stats_rating = 1.00
    stats_rating += 0.34 * acs_component
    stats_rating += 0.28 * kd_component
    stats_rating += 0.08 * assist_component
    stats_rating += 0.10 * win_component

    avg_vlr_rating = None
    if "vlr_rating" in group.columns and group["vlr_rating"].notna().any():
        avg_vlr_rating = group["vlr_rating"].mean()
        rating = (0.65 * avg_vlr_rating) + (0.35 * stats_rating)
    else:
        rating = stats_rating

    return {
        "base_rating": max(0.55, min(1.55, rating)),
        "avg_vlr_rating": avg_vlr_rating,
        "avg_acs": avg_acs,
        "kd_ratio": kd_ratio,
        "avg_assists": assists,
        "win_rate": win_rate,
        "recent_maps": len(group),
    }


def build_player_profiles(matches: pd.DataFrame, teams: list[str], recent_maps: int) -> pd.DataFrame:
    return build_form_profiles(matches, teams, recent_maps=recent_maps)


def apply_news(profiles: pd.DataFrame, news: pd.DataFrame, recent_days: int) -> pd.DataFrame:
    if profiles.empty:
        return profiles

    adjustments = player_news_adjustments(
        news,
        profiles[["team", "player"]],
        recent_days=recent_days,
        reference_date=datetime.utcnow(),
    )

    output = profiles.copy()
    output["news_adjustment"] = 0.0
    output["news_availability_adjustment"] = 0.0
    output["news_uncertainty_adjustment"] = 0.0
    output["news_reasons"] = ""
    output["news_urls"] = ""

    for index, row in output.iterrows():
        payload = adjustments.get((row["team"], row["player"]))
        if not payload:
            continue

        output.at[index, "news_adjustment"] = payload["adjustment"]
        output.at[index, "news_availability_adjustment"] = payload.get("availability_adjustment", 0.0)
        output.at[index, "news_uncertainty_adjustment"] = payload.get("uncertainty_adjustment", 0.0)
        reasons = []
        urls = []
        for reason in payload["reasons"][:5]:
            labels = ", ".join(reason["labels"]) if reason["labels"] else "news"
            reasons.append(f"{reason['title']} ({labels})")
            if reason["url"]:
                urls.append(reason["url"])
        output.at[index, "news_reasons"] = " | ".join(reasons)
        output.at[index, "news_urls"] = " | ".join(dict.fromkeys(urls))

    base_roster_uncertainty = pd.to_numeric(
        output["roster_uncertainty"] if "roster_uncertainty" in output.columns else 0.0,
        errors="coerce",
    )
    if not isinstance(base_roster_uncertainty, pd.Series):
        base_roster_uncertainty = pd.Series(base_roster_uncertainty, index=output.index, dtype=float)
    output["combined_roster_uncertainty"] = (
        base_roster_uncertainty.fillna(0.0) + output["news_uncertainty_adjustment"]
    ).clip(lower=0.0, upper=0.45)
    output["availability_probability"] = (
        1.0 + output["news_availability_adjustment"]
    ).clip(lower=0.05, upper=1.0)
    output["lineup_certainty"] = (
        (1.0 - output["combined_roster_uncertainty"])
        * output["availability_probability"]
    ).clip(lower=0.0, upper=1.0)
    if "data_reliability" in output.columns:
        output["reliability"] = output["data_reliability"] * output["lineup_certainty"]

    output["predicted_rating"] = output["base_rating"] * (1.0 + output["news_adjustment"])
    output["predicted_rating"] = output["predicted_rating"].clip(lower=0.45, upper=1.70)
    def numeric_column(name: str, fallback: float) -> pd.Series:
        if name not in output.columns:
            return pd.Series(fallback, index=output.index, dtype=float)
        return pd.to_numeric(output[name], errors="coerce").fillna(fallback)

    observed_std = numeric_column("rating_std", 0.18)
    model_std = numeric_column("trained_residual_std", 0.16)
    data_reliability = numeric_column(
        "data_reliability" if "data_reliability" in output.columns else "reliability",
        0.0,
    )
    roster_uncertainty = numeric_column("combined_roster_uncertainty", 0.0)
    combined_std = (0.55 * observed_std.pow(2) + 0.45 * model_std.pow(2)).pow(0.5)
    heuristic_std = (
        combined_std * (1.0 + 0.50 * (1.0 - data_reliability))
        + (0.12 * roster_uncertainty)
        + (0.08 * output["news_adjustment"].abs())
    ).clip(lower=0.08, upper=0.45)
    if {"trained_rating_low", "trained_rating_high"}.issubset(output.columns):
        news_multiplier = 1.0 + output["news_adjustment"]
        uncertainty_expansion = 0.10 * (
            roster_uncertainty + (1.0 - data_reliability)
        )
        output["predicted_rating_low"] = (
            output["trained_rating_low"] * news_multiplier - uncertainty_expansion
        ).clip(lower=0.35, upper=1.90)
        output["predicted_rating_high"] = (
            output["trained_rating_high"] * news_multiplier + uncertainty_expansion
        ).clip(lower=0.35, upper=1.90)
        interval_std = (
            output["predicted_rating_high"] - output["predicted_rating_low"]
        ) / 2.563
        output["predicted_rating_std"] = interval_std.clip(lower=0.08, upper=0.45)
    else:
        output["predicted_rating_std"] = heuristic_std
        interval_width = 1.282 * output["predicted_rating_std"]
        output["predicted_rating_low"] = (
            output["predicted_rating"] - interval_width
        ).clip(lower=0.35, upper=1.90)
        output["predicted_rating_high"] = (
            output["predicted_rating"] + interval_width
        ).clip(lower=0.35, upper=1.90)
    return output.sort_values(["team", "predicted_rating"], ascending=[True, False])


def predict_player_ratings(
    team1: str,
    team2: str,
    matches_csv: str = MATCHES_CSV,
    news_csv: str = NEWS_CSV,
    rosters_csv: str = ROSTERS_CSV,
    output_csv: str = PREDICTIONS_CSV,
    recent_maps: int = 10,
    recent_news_days: int = 45,
    use_rosters: bool = True,
    season_year: int | None = None,
    recent_days: int | None = None,
    use_trained_models: bool = False,
    prepared_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if prepared_matches is None:
        raw_matches = canonicalize_match_dataframe(
            load_csv(matches_csv),
            registry_team_pages(season_year=None),
        )
        matches = clean_matches(
            filter_registry_tier1_matchups(
                filter_curated_competition_history(raw_matches)
            )
        )
        if season_year is not None and matches["match_date_sort"].notna().any():
            matches = matches[matches["match_date_sort"].dt.year <= season_year].copy()
    else:
        matches = prepared_matches.copy()
    if matches.empty:
        raise ValueError(
            "No usable match data found. Run: python scripts\\scrape.py --matches"
        )

    rosters = load_csv(rosters_csv) if use_rosters else pd.DataFrame()
    if use_rosters and not rosters.empty:
        profiles = build_player_profiles_from_rosters(matches, [team1, team2], rosters, recent_maps)
    else:
        profiles = build_player_profiles(matches, [team1, team2], recent_maps)
    if profiles.empty:
        raise ValueError("No players found for those teams in the match CSV.")

    if use_trained_models:
        profiles = apply_player_model(profiles, matches, team1, team2)

    news = load_csv(news_csv)
    predictions = apply_news(profiles, news, recent_news_days)
    predictions.to_csv(output_csv, index=False)
    from ..storage import sync_dataframe

    sync_dataframe("player_predictions", predictions, source=str(output_csv))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Valorant player ratings with VLR news impact.")
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--matches-csv", default=MATCHES_CSV)
    parser.add_argument("--news-csv", default=NEWS_CSV)
    parser.add_argument("--rosters-csv", default=ROSTERS_CSV)
    parser.add_argument("--output-csv", default=PREDICTIONS_CSV)
    parser.add_argument("--recent-maps", type=int, default=10)
    parser.add_argument("--season-year", type=int)
    parser.add_argument("--recent-days", type=int)
    parser.add_argument("--recent-news-days", type=int, default=45)
    parser.add_argument("--refresh-news", action="store_true")
    parser.add_argument("--news-pages", type=int, default=3)
    parser.add_argument("--refresh-matches", action="store_true")
    parser.add_argument("--refresh-rosters", action="store_true")
    parser.add_argument("--ignore-rosters", action="store_true")
    parser.add_argument("--use-trained-models", action="store_true")
    parser.add_argument("--limit-per-team", type=int, default=0)
    parser.add_argument("--min-team-matches", type=int, default=20)
    args = parser.parse_args()

    if args.refresh_matches:
        scrape_matches(
            output_csv=args.matches_csv,
            limit_per_team=args.limit_per_team or None,
            team_pages=registry_team_pages(),
            season_year=args.season_year,
            recent_days=args.recent_days,
            min_matches_per_team=args.min_team_matches,
        )
    if args.refresh_news:
        scrape_news(output_csv=args.news_csv, pages=args.news_pages)
    if args.refresh_rosters:
        scrape_rosters(output_csv=args.rosters_csv, team_pages=registry_team_pages())

    predictions = predict_player_ratings(
        team1=args.team1,
        team2=args.team2,
        matches_csv=args.matches_csv,
        news_csv=args.news_csv,
        rosters_csv=args.rosters_csv,
        output_csv=args.output_csv,
        recent_maps=args.recent_maps,
        recent_news_days=args.recent_news_days,
        use_rosters=not args.ignore_rosters,
        season_year=args.season_year,
        recent_days=args.recent_days,
        use_trained_models=args.use_trained_models,
    )

    display_cols = [
        "team",
        "player",
        "base_rating",
        "news_adjustment",
        "predicted_rating",
        "recent_maps",
        "avg_acs",
        "kd_ratio",
        "avg_vlr_rating",
        "trained_base_rating",
        "roster_uncertainty",
        "is_new_to_team",
    ]
    existing_display_cols = [col for col in display_cols if col in predictions.columns]
    print(predictions[existing_display_cols].round(3).to_string(index=False))
    print(f"\nSaved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
