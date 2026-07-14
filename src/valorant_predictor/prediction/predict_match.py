import argparse

import pandas as pd

from ..config import MATCHES_CSV, MATCH_PREDICTION_CSV, NEWS_CSV, PREDICTIONS_CSV, ROSTERS_CSV
from ..features.form_calculations import (
    calculate_team_form,
    clean_match_data,
    filter_curated_competition_history,
    logistic_probability,
    normalize_map_name,
    probable_lineup,
)
from ..features.team_context import current_team_context, normalize_lineup_identity
from ..team_registry import filter_registry_tier1_matchups, registry_team_pages
from ..training.model_inference import (
    cached_team_context,
    predict_map_win_probabilities,
    predict_team_win_probability,
)
from ..training.train_models import dataset_fingerprint, team_rows_from_cleaned
from ..vlr_client import canonicalize_match_dataframe, scrape_matches, scrape_news, scrape_rosters
from .predict_player_ratings import load_csv, predict_player_ratings
from .simulation import simulate_series, stable_seed


def blend_map_model_probabilities(
    baseline_probabilities: dict[str, float],
    learned_probabilities: dict[str, float],
    reliability: float,
) -> tuple[dict[str, float], float]:
    weight = min(0.80, max(0.0, 0.75 * float(reliability)))
    blended = dict(baseline_probabilities)
    for map_name, learned_probability in learned_probabilities.items():
        baseline_probability = float(blended.get(map_name, 0.5))
        blended[map_name] = (
            (1.0 - weight) * baseline_probability
            + weight * float(learned_probability)
        )
    return blended, weight


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
        player_uncertainty = 0.18
        data_reliability = 0.0
        lineup_certainty = 0.0
    else:
        avg_rating = float(lineup["predicted_rating"].mean())
        top_rating = float(lineup["predicted_rating"].max())
        bottom_rating = float(lineup["predicted_rating"].min())
        rating_spread = top_rating - bottom_rating
        news_adjustment = float(lineup["news_adjustment"].mean()) if "news_adjustment" in lineup else 0.0
        roster_uncertainty = (
            float(lineup["roster_uncertainty"].mean()) if "roster_uncertainty" in lineup else 0.0
        )
        player_uncertainty = float(lineup["predicted_rating_std"].mean()) if "predicted_rating_std" in lineup else 0.18
        data_reliability = float(lineup["data_reliability"].mean()) if "data_reliability" in lineup else 0.0
        lineup_certainty = float(lineup["lineup_certainty"].mean()) if "lineup_certainty" in lineup else 1.0 - roster_uncertainty
    lineup_recent_maps = float(lineup["maps_60d"].mean()) if "maps_60d" in lineup and not lineup.empty else (float(lineup["recent_maps"].mean()) if not lineup.empty else 0.0)
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
        "avg_player_uncertainty": player_uncertainty,
        "data_reliability": data_reliability,
        "lineup_certainty": lineup_certainty,
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


def lineup_identities(lineup: pd.DataFrame) -> set[str]:
    return {
        normalize_lineup_identity(row.get("player"), row.get("player_id"))
        for _, row in lineup.iterrows()
    }


def confidence_components(
    team1_summary: dict,
    team2_summary: dict,
    model_reliability: float,
    map_pool_reliability: float,
    map_selection_reliability: float,
) -> dict:
    team_sample = min(team1_summary["recent_team_maps"], team2_summary["recent_team_maps"])
    team_form_reliability = min(1.0, team_sample / 10.0)
    data_reliability = min(
        team1_summary.get("data_reliability", 0.0),
        team2_summary.get("data_reliability", 0.0),
    )
    lineup_certainty = min(
        team1_summary.get("lineup_certainty", 0.0),
        team2_summary.get("lineup_certainty", 0.0),
    )
    quality = (
        0.30 * data_reliability
        + 0.20 * lineup_certainty
        + 0.15 * team_form_reliability
        + 0.10 * max(0.0, min(1.0, map_pool_reliability))
        + 0.10 * max(0.0, min(1.0, map_selection_reliability))
        + 0.15 * max(0.0, min(1.0, model_reliability))
    )
    return {
        "data_reliability": data_reliability,
        "lineup_certainty": lineup_certainty,
        "team_form_reliability": team_form_reliability,
        "map_pool_reliability": map_pool_reliability,
        "map_selection_reliability": map_selection_reliability,
        "model_reliability": model_reliability,
        "quality": quality,
    }


def prediction_reasons(team1_summary: dict, team2_summary: dict) -> list[str]:
    reasons = []
    rating_diff = team1_summary["avg_player_rating"] - team2_summary["avg_player_rating"]
    form_diff = team1_summary["recent_win_rate"] - team2_summary["recent_win_rate"]
    news_diff = team1_summary["avg_news_adjustment"] - team2_summary["avg_news_adjustment"]
    elo_diff = team1_summary.get("elo_edge", 0.0) - team2_summary.get("elo_edge", 0.0)
    map_pool_diff = team1_summary.get("map_pool_edge", 0.0) - team2_summary.get("map_pool_edge", 0.0)
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
    if abs(elo_diff) >= 0.20:
        leader = team1_summary["team"] if elo_diff > 0 else team2_summary["team"]
        reasons.append(f"{leader} has the stronger opponent-adjusted Elo baseline.")
    if abs(map_pool_diff) >= 0.10 and min(
        team1_summary.get("map_pool_reliability", 0.0),
        team2_summary.get("map_pool_reliability", 0.0),
    ) >= 0.35:
        leader = team1_summary["team"] if map_pool_diff > 0 else team2_summary["team"]
        reasons.append(f"{leader} has the stronger recent map-pool matchup.")
    if roster_uncertainty >= 0.05:
        reasons.append(
            "Roster uncertainty keeps the forecast closer to 50/50; "
            "it does not directly lower team strength."
        )

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
    match_date: str | None = None,
    event_name: str = "",
    event_stage: str = "",
    current_patch: str = "",
    selected_maps: list[str] | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    selected_maps = [normalize_map_name(name) for name in (selected_maps or []) if normalize_map_name(name)]
    if selected_maps and len(selected_maps) != best_of:
        raise ValueError(f"Choose exactly {best_of} unique maps for a Bo{best_of}, or leave all maps on Auto.")
    if len(set(selected_maps)) != len(selected_maps):
        raise ValueError("Each selected map must be unique.")
    raw_matches = canonicalize_match_dataframe(
        load_csv(matches_csv),
        registry_team_pages(season_year=None),
    )
    raw_matches = filter_registry_tier1_matchups(
        filter_curated_competition_history(raw_matches)
    )
    matches_fingerprint = dataset_fingerprint(raw_matches)
    matches = clean_match_data(raw_matches)
    if season_year is not None and matches["match_date_sort"].notna().any():
        matches = matches[matches["match_date_sort"].dt.year <= season_year].copy()
    if matches.empty:
        raise ValueError("No usable match data found. Run: python scripts\\scrape.py --matches")

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
        prepared_matches=matches,
    )
    players = mark_lineup_usage(players, [team1, team2])
    players.to_csv(player_output_csv, index=False)

    team1_summary = summarize_team(team1, players, matches)
    team2_summary = summarize_team(team2, players, matches)
    lineup1 = lineup_identities(probable_lineup(players, team1, lineup_size=5))
    lineup2 = lineup_identities(probable_lineup(players, team2, lineup_size=5))
    team_matches = team_rows_from_cleaned(matches)
    prediction_date = pd.to_datetime(match_date, utc=True, errors="coerce")
    if pd.isna(prediction_date):
        prediction_date = (
            team_matches["match_date_sort"].max() + pd.Timedelta(days=1)
            if team_matches["match_date_sort"].notna().any()
            else pd.Timestamp.now(tz="UTC")
        )
    team_context = cached_team_context(
        team1,
        team2,
        lineup1,
        lineup2,
        matches_fingerprint,
        as_of_date=prediction_date,
        current_patch=current_patch,
    )
    if team_context is None:
        team_context = current_team_context(
            matches,
            team_matches,
            team1,
            team2,
            lineup_a=lineup1,
            lineup_b=lineup2,
            as_of_date=prediction_date,
            current_patch=current_patch,
        )
    diff = team1_summary["strength"] - team2_summary["strength"]
    trained_team_probability = None
    trained_features = None
    if use_trained_models:
        trained_team_probability, trained_features = predict_team_win_probability(
            matches,
            team1,
            team2,
            lineup1=lineup1,
            lineup2=lineup2,
            team_maps=team_matches,
            sequential_context=team_context,
            target_date=prediction_date,
            current_patch=current_patch,
        )
        if trained_features:
            team_context = {**team_context, **trained_features}

    model_reliability = float(team_context.get("model_reliability", 0.0))
    map_pool_reliability = float(team_context.get("map_pool_reliability", 0.0))
    map_selection_reliability = (
        1.0
        if selected_maps
        else float(team_context.get("map_selection_reliability", 0.0))
    )
    confidence = confidence_components(
        team1_summary,
        team2_summary,
        model_reliability,
        map_pool_reliability,
        map_selection_reliability,
    )
    raw_formula_map_probability = logistic_probability(diff, scale=5.0)
    evidence_shrink = 0.45 + (0.55 * confidence["quality"])
    formula_probability = 0.5 + (raw_formula_map_probability - 0.5) * evidence_shrink
    elo_probability = float(team_context.get("elo_probability", 0.5))
    elo_reliability = float(team_context.get("elo_reliability", 0.0)) * float(
        team_context.get("minimum_elo_freshness", 1.0)
    )
    elo_weight = 0.25 + (0.35 * max(0.0, min(1.0, elo_reliability)))
    map_probability = (
        elo_weight * elo_probability
        + (1.0 - elo_weight) * formula_probability
    )
    if trained_team_probability is not None:
        if bool(team_context.get("active_model_is_baseline")):
            map_probability = trained_team_probability
        else:
            model_weight = (
                0.80
                if bool(team_context.get("manual_model_override"))
                else min(0.80, max(0.0, 0.75 * model_reliability))
            )
            map_probability = model_weight * trained_team_probability + (1.0 - model_weight) * map_probability

    contextual_map_probabilities = team_context.get("map_probabilities", {})
    map_weight = 0.45 * map_pool_reliability
    if contextual_map_probabilities:
        simulation_map_probabilities = {
            map_name: (1.0 - map_weight) * map_probability + map_weight * float(probability)
            for map_name, probability in contextual_map_probabilities.items()
        }
    else:
        simulation_map_probabilities = {"Overall": map_probability}

    estimated_order = [
        normalize_map_name(name)
        for name in team_context.get("likely_map_order", [])
        if normalize_map_name(name)
    ][:best_of]
    resolved_map_order = selected_maps or estimated_order
    map_selection_source = "selected" if selected_maps else "estimated"
    for map_name in resolved_map_order:
        simulation_map_probabilities.setdefault(map_name, map_probability)

    map_model_metadata = {
        "map_model_used": False,
        "map_model_reliability": 0.0,
        "map_model_candidate": "",
        "map_model_deployment_weight": 0.0,
    }
    if use_trained_models and resolved_map_order:
        learned_map_probabilities, map_model_metadata = predict_map_win_probabilities(
            team_context,
            resolved_map_order,
        )
        if map_model_metadata.get("map_model_used"):
            simulation_map_probabilities, deployment_weight = (
                blend_map_model_probabilities(
                    simulation_map_probabilities,
                    learned_map_probabilities,
                    map_model_metadata.get("map_model_reliability", 0.0),
                )
            )
            map_model_metadata["map_model_deployment_weight"] = deployment_weight

    performance_volatility = 1.8 * (
        team1_summary.get("avg_player_uncertainty", 0.18)
        + team2_summary.get("avg_player_uncertainty", 0.18)
    ) / 2.0
    simulation = simulate_series(
        simulation_map_probabilities,
        best_of=best_of,
        performance_volatility=performance_volatility,
        seed=stable_seed(team1, team2, str(best_of), str(season_year)),
        map_order=resolved_map_order,
        veto_uncertainty=(
            0.0
            if selected_maps
            else 0.55 * (1.0 - float(team_context.get("map_selection_reliability", 0.0)))
        ),
    )
    match_probability = float(simulation["team1_win_probability"])
    decisiveness = abs(match_probability - 0.5) * 2.0
    prediction_confidence = confidence["quality"] * (0.40 + 0.60 * decisiveness)
    adjusted_diff = diff * confidence["quality"]

    team1_summary.update(
        {
            "elo_edge": team_context.get("elo_diff", 0.0),
            "map_pool_edge": team_context.get("map_pool_win_rate_diff", 0.0),
            "map_pool_reliability": map_pool_reliability,
            "roster_continuity": team_context.get("team_roster_continuity", team1_summary.get("lineup_certainty", 0.0)),
        }
    )
    team2_summary.update(
        {
            "elo_edge": -float(team_context.get("elo_diff", 0.0)),
            "map_pool_edge": -float(team_context.get("map_pool_win_rate_diff", 0.0)),
            "map_pool_reliability": map_pool_reliability,
            "roster_continuity": team_context.get("opponent_roster_continuity", team2_summary.get("lineup_certainty", 0.0)),
        }
    )

    winner = team1 if match_probability >= 0.5 else team2
    winner_probability = match_probability if winner == team1 else 1.0 - match_probability

    reasons = prediction_reasons(team1_summary, team2_summary)
    ordered_map_probabilities = [
        float(simulation["map_probabilities"].get(map_name, map_probability))
        for map_name in simulation.get("likely_map_order", [])
    ]
    if ordered_map_probabilities:
        average_selected_map_probability = float(
            sum(ordered_map_probabilities) / len(ordered_map_probabilities)
        )
        if abs(average_selected_map_probability - 0.5) >= 0.03:
            map_leader = team1 if average_selected_map_probability > 0.5 else team2
            map_label = "selected" if selected_maps else "estimated"
            reasons.append(f"{map_leader} has the stronger outlook across the {map_label} maps.")

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
        "match_date": prediction_date.isoformat(),
        "event_name": event_name,
        "event_stage": event_stage,
        "patch": current_patch,
        "strength_diff": diff,
        "adjusted_strength_diff": adjusted_diff,
        "confidence_factor": confidence["quality"],
        "prediction_confidence": prediction_confidence,
        "data_reliability": confidence["data_reliability"],
        "lineup_certainty": confidence["lineup_certainty"],
        "team_form_reliability": confidence["team_form_reliability"],
        "map_pool_reliability": confidence["map_pool_reliability"],
        "map_selection_reliability": confidence["map_selection_reliability"],
        "model_reliability": confidence["model_reliability"],
        "map_model_reliability": map_model_metadata.get("map_model_reliability", 0.0),
        "map_model_used": map_model_metadata.get("map_model_used", False),
        "map_model_candidate": map_model_metadata.get("map_model_candidate", ""),
        "map_model_deployment_weight": map_model_metadata.get(
            "map_model_deployment_weight",
            0.0,
        ),
        "map_selection_source": map_selection_source,
        "elo_probability": elo_probability,
        "elo_reliability": elo_reliability,
        "trained_team1_map_probability": trained_team_probability,
        "team1_probability_low": simulation["probability_low"],
        "team1_probability_high": simulation["probability_high"],
        "likely_score": simulation["likely_score"],
        "simulation_count": simulation["simulations"],
        "map_probabilities": " | ".join(
            f"{name}: {probability:.3f}"
            for name, probability in simulation["map_probabilities"].items()
        ),
        "likely_map_order": " | ".join(simulation.get("likely_map_order", [])),
        "reasons": " | ".join(reasons),
    }
    for index, map_name in enumerate(simulation.get("likely_map_order", []), start=1):
        probability = float(simulation["map_probabilities"].get(map_name, map_probability))
        summary[f"map_{index}_name"] = map_name
        summary[f"map_{index}_team1_win_probability"] = probability
        summary[f"map_{index}_team2_win_probability"] = 1.0 - probability
        summary[f"map_{index}_predicted_winner"] = team1 if probability >= 0.5 else team2

    summary_df = pd.DataFrame([{**summary, **{f"team1_{k}": v for k, v in team1_summary.items() if k != "team"}}])
    for key, value in team2_summary.items():
        if key != "team":
            summary_df[f"team2_{key}"] = value
    summary_df.to_csv(match_output_csv, index=False)
    from ..storage import sync_dataframe

    sync_dataframe("match_predictions", summary_df, source=str(match_output_csv))

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
    parser.add_argument(
        "--maps",
        nargs="*",
        default=[],
        help="Ordered maps; provide exactly the Bo count, for example --maps Haven Breeze Lotus.",
    )
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
        selected_maps=args.maps,
    )
    print_prediction(summary, players, team_summaries)
    print()
    print(f"Saved match prediction to {args.match_output_csv}")
    print(f"Saved player predictions to {args.player_output_csv}")


if __name__ == "__main__":
    main()
