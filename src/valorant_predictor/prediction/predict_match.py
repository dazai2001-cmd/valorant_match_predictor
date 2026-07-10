import argparse

import pandas as pd

from ..config import MATCHES_CSV, MATCH_PREDICTION_CSV, NEWS_CSV, PREDICTIONS_CSV, ROSTERS_CSV
from ..features.form_calculations import (
    best_of_probability,
    calculate_team_form,
    clean_match_data,
    filter_matches_by_time,
    logistic_probability,
    probable_lineup,
)
from ..team_registry import registry_team_pages
from ..training.model_inference import predict_team_win_probability
from ..vlr_client import scrape_matches, scrape_news, scrape_rosters
from .predict_player_ratings import load_csv, predict_player_ratings


def summarize_team(team: str, player_predictions: pd.DataFrame, matches: pd.DataFrame) -> dict:
    lineup = probable_lineup(player_predictions, team, lineup_size=5)
    team_form = calculate_team_form(matches, team)

    if lineup.empty:
        avg_rating = 1.0
        top_rating = 1.0
        bottom_rating = 1.0
        rating_spread = 0.0
        news_adjustment = 0.0
        roster_uncertainty = 0.0
    else:
        avg_rating = float(lineup["predicted_rating"].mean())
        top_rating = float(lineup["predicted_rating"].max())
        bottom_rating = float(lineup["predicted_rating"].min())
        rating_spread = top_rating - bottom_rating
        news_adjustment = float(lineup["news_adjustment"].mean()) if "news_adjustment" in lineup else 0.0
        roster_uncertainty = (
            float(lineup["roster_uncertainty"].mean()) if "roster_uncertainty" in lineup else 0.0
        )
    lineup_recent_maps = float(lineup["recent_maps"].mean()) if not lineup.empty else 0.0
    lineup_reliability = float(lineup["reliability"].mean()) if "reliability" in lineup and not lineup.empty else 0.0

    strength = avg_rating
    strength += team_form["team_form_adjustment"]
    strength += 0.03 * (top_rating - avg_rating)
    strength -= 0.04 * max(0.0, avg_rating - bottom_rating)
    strength += 0.03 * news_adjustment

    return {
        "team": team,
        "strength": strength,
        "avg_player_rating": avg_rating,
        "top_player_rating": top_rating,
        "bottom_player_rating": bottom_rating,
        "rating_spread": rating_spread,
        "avg_news_adjustment": news_adjustment,
        "avg_roster_uncertainty": roster_uncertainty,
        "lineup_recent_maps": lineup_recent_maps,
        "lineup_reliability": lineup_reliability,
        **team_form,
    }


def mark_lineup_usage(player_predictions: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    output = player_predictions.copy()
    output["used_in_lineup"] = False
    for team in teams:
        lineup = probable_lineup(output, team, lineup_size=5)
        output.loc[lineup.index, "used_in_lineup"] = True
    return output


def confidence_factor(team1_summary: dict, team2_summary: dict) -> float:
    team_sample = min(team1_summary["recent_team_maps"], team2_summary["recent_team_maps"])
    lineup_sample = min(team1_summary["lineup_recent_maps"], team2_summary["lineup_recent_maps"])
    lineup_reliability = min(team1_summary["lineup_reliability"], team2_summary["lineup_reliability"])
    roster_uncertainty = max(
        team1_summary.get("avg_roster_uncertainty", 0.0),
        team2_summary.get("avg_roster_uncertainty", 0.0),
    )

    team_confidence = min(1.0, team_sample / 8.0)
    lineup_confidence = min(1.0, lineup_sample / 10.0)
    raw_confidence = min(team_confidence, lineup_confidence, lineup_reliability)
    confidence = 0.35 + (0.65 * raw_confidence)
    return confidence * (1.0 - min(0.30, roster_uncertainty))


def prediction_reasons(team1_summary: dict, team2_summary: dict) -> list[str]:
    reasons = []
    rating_diff = team1_summary["avg_player_rating"] - team2_summary["avg_player_rating"]
    form_diff = team1_summary["recent_win_rate"] - team2_summary["recent_win_rate"]
    news_diff = team1_summary["avg_news_adjustment"] - team2_summary["avg_news_adjustment"]
    roster_uncertainty = max(
        team1_summary.get("avg_roster_uncertainty", 0.0),
        team2_summary.get("avg_roster_uncertainty", 0.0),
    )

    if abs(rating_diff) >= 0.03:
        leader = team1_summary["team"] if rating_diff > 0 else team2_summary["team"]
        reasons.append(f"{leader} has the stronger projected five-player rating.")
    if abs(form_diff) >= 0.10:
        leader = team1_summary["team"] if form_diff > 0 else team2_summary["team"]
        reasons.append(f"{leader} has better recent win form.")
    if abs(news_diff) >= 0.03:
        leader = team1_summary["team"] if news_diff > 0 else team2_summary["team"]
        reasons.append(f"{leader} has the better news/availability signal.")
    if roster_uncertainty >= 0.05:
        reasons.append("Roster uncertainty is dampening confidence, not directly lowering team strength.")

    if not reasons:
        reasons.append("The teams grade closely, so this prediction is low-confidence.")
    return reasons


def predict_match(
    team1: str,
    team2: str,
    matches_csv: str = MATCHES_CSV,
    news_csv: str = NEWS_CSV,
    rosters_csv: str = ROSTERS_CSV,
    player_output_csv: str = PREDICTIONS_CSV,
    match_output_csv: str = MATCH_PREDICTION_CSV,
    recent_maps: int = 10,
    recent_news_days: int = 45,
    best_of: int = 3,
    use_rosters: bool = True,
    season_year: int | None = None,
    recent_days: int | None = None,
    use_trained_models: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    matches = clean_match_data(load_csv(matches_csv))
    if season_year is not None or recent_days is not None:
        matches = filter_matches_by_time(matches, season_year=season_year, recent_days=recent_days)
    if matches.empty:
        raise ValueError("No usable match data found. Run: python scripts\\scrape.py --matches --limit-per-team 25")

    players = predict_player_ratings(
        team1=team1,
        team2=team2,
        matches_csv=matches_csv,
        news_csv=news_csv,
        rosters_csv=rosters_csv,
        output_csv=player_output_csv,
        recent_maps=recent_maps,
        recent_news_days=recent_news_days,
        use_rosters=use_rosters,
        season_year=season_year,
        recent_days=recent_days,
        use_trained_models=use_trained_models,
    )
    players = mark_lineup_usage(players, [team1, team2])
    players.to_csv(player_output_csv, index=False)

    team1_summary = summarize_team(team1, players, matches)
    team2_summary = summarize_team(team2, players, matches)
    diff = team1_summary["strength"] - team2_summary["strength"]
    confidence = confidence_factor(team1_summary, team2_summary)
    adjusted_diff = diff * confidence
    map_probability = logistic_probability(adjusted_diff, scale=5.0)
    match_probability = best_of_probability(map_probability, best_of=best_of)
    trained_team_probability = None
    if use_trained_models:
        trained_team_probability, _ = predict_team_win_probability(matches, team1, team2)
        if trained_team_probability is not None:
            trained_match_probability = best_of_probability(trained_team_probability, best_of=best_of)
            match_probability = (0.65 * trained_match_probability) + (0.35 * match_probability)
            map_probability = (0.65 * trained_team_probability) + (0.35 * map_probability)

    winner = team1 if match_probability >= 0.5 else team2
    winner_probability = match_probability if winner == team1 else 1.0 - match_probability

    summary = {
        "team1": team1,
        "team2": team2,
        "predicted_winner": winner,
        "winner_probability": winner_probability,
        "team1_win_probability": match_probability,
        "team2_win_probability": 1.0 - match_probability,
        "team1_map_probability": map_probability,
        "team2_map_probability": 1.0 - map_probability,
        "best_of": best_of,
        "strength_diff": diff,
        "adjusted_strength_diff": adjusted_diff,
        "confidence_factor": confidence,
        "trained_team1_map_probability": trained_team_probability,
        "reasons": " | ".join(prediction_reasons(team1_summary, team2_summary)),
    }

    summary_df = pd.DataFrame([{**summary, **{f"team1_{k}": v for k, v in team1_summary.items() if k != "team"}}])
    for key, value in team2_summary.items():
        if key != "team":
            summary_df[f"team2_{key}"] = value
    summary_df.to_csv(match_output_csv, index=False)

    return summary, players, pd.DataFrame([team1_summary, team2_summary])


def print_prediction(summary: dict, players: pd.DataFrame, team_summaries: pd.DataFrame) -> None:
    print(f"Match: {summary['team1']} vs {summary['team2']} (Bo{summary['best_of']})")
    print(f"Predicted winner: {summary['predicted_winner']} ({summary['winner_probability']:.1%})")
    print()
    print("Win probability")
    print(f"{summary['team1']}: {summary['team1_win_probability']:.1%}")
    print(f"{summary['team2']}: {summary['team2_win_probability']:.1%}")
    print()

    display_summary = team_summaries[
        [
            "team",
            "strength",
            "avg_player_rating",
            "recent_win_rate",
            "team_form_adjustment",
            "avg_news_adjustment",
            "avg_roster_uncertainty",
            "lineup_reliability",
            "recent_team_maps",
        ]
    ].copy()
    print("Team strength")
    print(display_summary.round(3).to_string(index=False))
    print()

    print("Key reasons")
    for reason in summary["reasons"].split(" | "):
        print(f"- {reason}")
    print()

    player_cols = [
        "team",
        "player",
        "predicted_rating",
        "base_rating",
        "news_adjustment",
        "recent_maps",
        "roster_uncertainty",
        "is_new_to_team",
        "used_in_lineup",
    ]
    player_cols = [col for col in player_cols if col in players.columns]
    print("Predicted player ratings")
    print(players[player_cols].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Valorant match winner from player form and VLR news.")
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--matches-csv", default=MATCHES_CSV)
    parser.add_argument("--news-csv", default=NEWS_CSV)
    parser.add_argument("--rosters-csv", default=ROSTERS_CSV)
    parser.add_argument("--player-output-csv", default=PREDICTIONS_CSV)
    parser.add_argument("--match-output-csv", default=MATCH_PREDICTION_CSV)
    parser.add_argument("--recent-maps", type=int, default=10)
    parser.add_argument("--season-year", type=int)
    parser.add_argument("--recent-days", type=int)
    parser.add_argument("--recent-news-days", type=int, default=45)
    parser.add_argument("--best-of", type=int, default=3)
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

    summary, players, team_summaries = predict_match(
        team1=args.team1,
        team2=args.team2,
        matches_csv=args.matches_csv,
        news_csv=args.news_csv,
        rosters_csv=args.rosters_csv,
        player_output_csv=args.player_output_csv,
        match_output_csv=args.match_output_csv,
        recent_maps=args.recent_maps,
        recent_news_days=args.recent_news_days,
        best_of=args.best_of,
        use_rosters=not args.ignore_rosters,
        season_year=args.season_year,
        recent_days=args.recent_days,
        use_trained_models=args.use_trained_models,
    )
    print_prediction(summary, players, team_summaries)
    print()
    print(f"Saved match prediction to {args.match_output_csv}")
    print(f"Saved player predictions to {args.player_output_csv}")


if __name__ == "__main__":
    main()
