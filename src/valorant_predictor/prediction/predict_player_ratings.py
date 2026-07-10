import argparse
from datetime import datetime

import pandas as pd

from ..config import MATCHES_CSV, NEWS_CSV, PREDICTIONS_CSV, ROSTERS_CSV
from ..features.form_calculations import build_player_profiles as build_form_profiles
from ..features.form_calculations import build_player_profiles_from_rosters
from ..features.form_calculations import clean_match_data
from ..features.form_calculations import filter_matches_by_time
from ..features.news_impact import player_news_adjustments
from ..team_registry import registry_team_pages
from ..training.model_inference import apply_player_model
from ..vlr_client import scrape_matches, scrape_news, scrape_rosters


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
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
    output["news_reasons"] = ""
    output["news_urls"] = ""

    for index, row in output.iterrows():
        payload = adjustments.get((row["team"], row["player"]))
        if not payload:
            continue

        output.at[index, "news_adjustment"] = payload["adjustment"]
        reasons = []
        urls = []
        for reason in payload["reasons"][:5]:
            labels = ", ".join(reason["labels"]) if reason["labels"] else "news"
            reasons.append(f"{reason['title']} ({labels})")
            if reason["url"]:
                urls.append(reason["url"])
        output.at[index, "news_reasons"] = " | ".join(reasons)
        output.at[index, "news_urls"] = " | ".join(dict.fromkeys(urls))

    output["predicted_rating"] = output["base_rating"] * (1.0 + output["news_adjustment"])
    output["predicted_rating"] = output["predicted_rating"].clip(lower=0.45, upper=1.70)
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
) -> pd.DataFrame:
    matches = clean_matches(load_csv(matches_csv))
    if season_year is not None or recent_days is not None:
        matches = filter_matches_by_time(matches, season_year=season_year, recent_days=recent_days)
    if matches.empty:
        raise ValueError(
            "No usable match data found. Run: python scripts\\scrape.py --matches --limit-per-team 25"
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
    parser.add_argument("--limit-per-team", type=int, default=25)
    args = parser.parse_args()

    if args.refresh_matches:
        scrape_matches(
            output_csv=args.matches_csv,
            limit_per_team=args.limit_per_team,
            team_pages=registry_team_pages(),
            season_year=args.season_year,
            recent_days=args.recent_days,
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
