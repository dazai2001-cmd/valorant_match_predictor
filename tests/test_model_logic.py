import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valorant_predictor.features.form_calculations import (
    filter_curated_competition_history,
    player_form_from_group,
)
from valorant_predictor.features.news_impact import player_news_adjustments, structure_news_events
from valorant_predictor.features.team_context import build_sequential_team_context
from valorant_predictor.model_selection import normalize_model_selection
from valorant_predictor.prediction.predict_match import blend_map_model_probabilities
from valorant_predictor.prediction.simulation import simulate_series
from valorant_predictor.storage import load_table, sync_dataframe
from valorant_predictor.team_registry import (
    filter_registry_tier1_matchups,
    normalize_registry,
)
from valorant_predictor.training.train_models import (
    add_training_weights,
    apply_probability_blend_temperature,
    grouped_chronological_split,
    grouped_chronological_three_way_split,
    rolling_origin_splits,
)


class ModelLogicTests(unittest.TestCase):
    def test_curated_history_excludes_older_unknown_competitions(self):
        frame = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "match_date": "2024-06-01",
                    "competition_tier": "unknown",
                    "event_tier": "unknown",
                },
                {
                    "match_id": 2,
                    "match_date": "2025-02-01",
                    "competition_tier": "tier1",
                    "event_tier": "tier1",
                },
                {
                    "match_id": 3,
                    "match_date": "2025-03-01",
                    "competition_tier": "unknown",
                    "event_tier": "unknown",
                },
                {
                    "match_id": 4,
                    "match_date": "2026-01-01",
                    "competition_tier": "unknown",
                    "event_tier": "unknown",
                },
            ]
        )

        filtered = filter_curated_competition_history(
            frame,
            minimum_classified_matches=1,
        )

        self.assertEqual(set(filtered["match_id"]), {2, 4})

    def test_registry_keeps_the_same_team_in_multiple_seasons(self):
        registry = normalize_registry(
            pd.DataFrame(
                [
                    {"team": "Example", "team_id": "7", "season_year": "2025"},
                    {"team": "Example", "team_id": "7", "season_year": "2026"},
                ]
            )
        )

        self.assertEqual(len(registry), 2)
        self.assertEqual(set(registry["season_year"]), {"2025", "2026"})

    def test_unclassified_match_requires_two_registered_tier1_teams(self):
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "match_date": "2026-02-01",
                    "season_year": 2026,
                    "team": "Team A",
                    "opponent": "Team B",
                    "competition_tier": "unknown",
                    "event_tier": "unknown",
                },
                {
                    "match_id": 2,
                    "match_date": "2026-02-02",
                    "season_year": 2026,
                    "team": "Team A",
                    "opponent": "Tier 2 Team",
                    "competition_tier": "unknown",
                    "event_tier": "unknown",
                },
                {
                    "match_id": 3,
                    "match_date": "2025-10-01",
                    "season_year": 2025,
                    "team": "Promoted Team",
                    "opponent": "Tier 2 Team",
                    "competition_tier": "promotion",
                    "event_tier": "promotion",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "teams.csv"
            pd.DataFrame(
                [
                    {"team": "Team A", "tier": "tier1", "active": "true", "season_year": 2026},
                    {"team": "Team B", "tier": "tier1", "active": "true", "season_year": 2026},
                ]
            ).to_csv(registry_path, index=False)
            filtered = filter_registry_tier1_matchups(matches, path=registry_path)

        self.assertEqual(set(filtered["match_id"]), {1, 3})

    def test_model_selection_rejects_unknown_candidates(self):
        selection = normalize_model_selection(
            {"player": "extra_trees", "team": "not-a-model", "map": "map_form_baseline"}
        )

        self.assertEqual(selection["player"], "extra_trees")
        self.assertEqual(selection["team"], "auto")
        self.assertEqual(selection["map"], "map_form_baseline")

    def test_probability_blend_temperature_preserves_team_symmetry(self):
        forward = apply_probability_blend_temperature(
            [0.72, 0.41],
            [0.60, 0.48],
            blend_weight=0.35,
            temperature=0.75,
        )
        reverse = apply_probability_blend_temperature(
            [0.28, 0.59],
            [0.40, 0.52],
            blend_weight=0.35,
            temperature=0.75,
        )

        self.assertTrue((abs(forward + reverse - 1.0) < 1e-10).all())

    def test_map_model_is_deployed_in_proportion_to_measured_reliability(self):
        blended, weight = blend_map_model_probabilities(
            {"Haven": 0.42},
            {"Haven": 0.20},
            reliability=0.01,
        )

        self.assertAlmostEqual(weight, 0.0075)
        self.assertGreater(blended["Haven"], 0.41)
        self.assertLess(blended["Haven"], 0.42)

    def test_reliability_uses_uncapped_60_day_evidence(self):
        dates = pd.date_range("2026-06-01", periods=30, freq="D", tz="UTC")
        group = pd.DataFrame(
            {
                "match_date_sort": dates,
                "match_id": range(1, 31),
                "map_number": 1,
                "map_id": range(101, 131),
                "rating_for_model": [1.0 + (index % 5) * 0.02 for index in range(30)],
                "vlr_rating": [1.0] * 30,
                "acs": [210] * 30,
                "kills": [18] * 30,
                "deaths": [15] * 30,
                "assists": [6] * 30,
                "is_winner": [True] * 30,
                "agents": ["Sova"] * 30,
            }
        )

        profile = player_form_from_group(group, global_rating=1.0, recent_maps=10, reference_date=dates.max())

        self.assertEqual(profile["recent_maps"], 10)
        self.assertEqual(profile["maps_60d"], 30)
        self.assertGreater(profile["data_reliability"], 0.70)

    def test_grouped_split_keeps_match_rows_together(self):
        rows = []
        for match_id in range(1, 13):
            for _ in range(4):
                rows.append(
                    {
                        "match_key": f"id:{match_id}",
                        "target_date": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=match_id),
                        "target_match_id": match_id,
                    }
                )
        frame = pd.DataFrame(rows)

        train_mask, validation_mask = grouped_chronological_split(frame)
        train_matches = set(frame.loc[train_mask, "match_key"])
        validation_matches = set(frame.loc[validation_mask, "match_key"])

        self.assertFalse(train_matches & validation_matches)
        self.assertLess(frame.loc[train_mask, "target_match_id"].max(), frame.loc[validation_mask, "target_match_id"].min())

    def test_three_way_split_is_grouped_and_chronological(self):
        frame = pd.DataFrame(
            [
                {
                    "match_key": f"id:{match_id}",
                    "target_date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=match_id),
                    "target_match_id": match_id,
                }
                for match_id in range(1, 31)
                for _ in range(2)
            ]
        )
        train, calibration, test = grouped_chronological_three_way_split(frame)
        train_ids = set(frame.loc[train, "target_match_id"])
        calibration_ids = set(frame.loc[calibration, "target_match_id"])
        test_ids = set(frame.loc[test, "target_match_id"])

        self.assertFalse(train_ids & calibration_ids)
        self.assertFalse(calibration_ids & test_ids)
        self.assertLess(max(train_ids), min(calibration_ids))
        self.assertLess(max(calibration_ids), min(test_ids))

    def test_training_weights_keep_old_rows_but_reduce_their_influence(self):
        frame = pd.DataFrame(
            [
                {"target_date": "2025-01-01", "match_importance": 1.0},
                {"target_date": "2026-01-01", "match_importance": 1.0},
            ]
        )
        weighted = add_training_weights(frame)

        self.assertEqual(len(weighted), 2)
        self.assertGreater(weighted.iloc[1]["training_weight"], weighted.iloc[0]["training_weight"])

    def test_rolling_origin_splits_move_forward_without_match_leakage(self):
        frame = pd.DataFrame(
            [
                {
                    "match_key": f"id:{match_id}",
                    "target_date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=match_id),
                    "target_match_id": match_id,
                }
                for match_id in range(1, 61)
                for _ in range(3)
            ]
        )

        splits = rolling_origin_splits(frame, folds=3)

        self.assertEqual(len(splits), 3)
        previous_test_start = 0
        for train, calibration, test in splits:
            train_ids = set(frame.loc[train, "target_match_id"])
            calibration_ids = set(frame.loc[calibration, "target_match_id"])
            test_ids = set(frame.loc[test, "target_match_id"])
            self.assertFalse(train_ids & calibration_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(calibration_ids & test_ids)
            self.assertLess(max(train_ids), min(calibration_ids))
            self.assertLess(max(calibration_ids), min(test_ids))
            self.assertGreater(min(test_ids), previous_test_start)
            previous_test_start = min(test_ids)

    def test_sequential_context_uses_only_prior_results(self):
        team_matches = pd.DataFrame(
            [
                {"match_key": "id:1", "match_id": 1, "match_date_sort": pd.Timestamp("2026-01-01", tz="UTC"), "team": "A", "opponent": "B", "team_win": 1.0, "score_margin": 1.0},
                {"match_key": "id:1", "match_id": 1, "match_date_sort": pd.Timestamp("2026-01-01", tz="UTC"), "team": "B", "opponent": "A", "team_win": 0.0, "score_margin": -1.0},
                {"match_key": "id:2", "match_id": 2, "match_date_sort": pd.Timestamp("2026-02-01", tz="UTC"), "team": "A", "opponent": "B", "team_win": 0.0, "score_margin": -1.0},
                {"match_key": "id:2", "match_id": 2, "match_date_sort": pd.Timestamp("2026-02-01", tz="UTC"), "team": "B", "opponent": "A", "team_win": 1.0, "score_margin": 1.0},
            ]
        )
        cleaned_rows = []
        for match_key, match_id, team, opponent, score, opp_score in [
            ("id:1", 1, "A", "B", 13, 8),
            ("id:1", 1, "B", "A", 8, 13),
            ("id:2", 2, "A", "B", 9, 13),
            ("id:2", 2, "B", "A", 13, 9),
        ]:
            for player_index in range(5):
                cleaned_rows.append(
                    {
                        "match_key": match_key,
                        "match_id": match_id,
                        "map_id": match_id * 10,
                        "map_name": "Ascent",
                        "team": team,
                        "opponent": opponent,
                        "player": f"{team}{player_index}",
                        "map_team_score": score,
                        "map_opp_score": opp_score,
                        "is_winner": score > opp_score,
                        "rating_for_model": 1.1 if score > opp_score else 0.9,
                    }
                )
        context, _ = build_sequential_team_context(pd.DataFrame(cleaned_rows), team_matches)
        second_match_a = context[(context["match_key"] == "id:2") & (context["team"] == "A")].iloc[0]

        self.assertGreater(second_match_a["elo_diff"], 0.0)
        self.assertGreater(second_match_a["map_pool_win_rate_diff"], 0.0)
        self.assertLess(second_match_a["minimum_elo_freshness"], 1.0)

    def test_news_events_separate_rating_and_roster_uncertainty(self):
        news = pd.DataFrame(
            [
                {"url": "https://example.test/1", "title": "A signs NewPlayer", "summary": "NewPlayer joins A", "published": "July 10, 2026"},
                {"url": "https://example.test/2", "title": "Star injured", "summary": "A Star has a wrist injury", "published": "July 10, 2026"},
            ]
        )
        players = pd.DataFrame([{"team": "A", "player": "Star"}, {"team": "A", "player": "NewPlayer"}])
        events = structure_news_events(news, reference_date=pd.Timestamp("2026-07-11").to_pydatetime())
        adjustments = player_news_adjustments(news, players, reference_date=pd.Timestamp("2026-07-11").to_pydatetime())

        self.assertIn("roster_join", set(events["event_type"]))
        self.assertLess(adjustments[("A", "Star")]["adjustment"], 0.0)
        self.assertGreater(adjustments[("A", "NewPlayer")]["uncertainty_adjustment"], 0.0)

    def test_accented_player_name_does_not_match_roster_substring(self):
        news = pd.DataFrame(
            [
                {
                    "title": "FURIA completes Stage 2 roster with Shyy",
                    "summary": "The Brazilian team has finalized its active five.",
                    "published": "July 1, 2026",
                    "url": "https://example.test/furia-roster",
                }
            ]
        )
        players = pd.DataFrame([{"team": "BBL Esports", "player": "Rosé"}])

        adjustments = player_news_adjustments(
            news,
            players,
            reference_date=pd.Timestamp("2026-07-10").to_pydatetime(),
        )

        self.assertNotIn(("BBL Esports", "Rosé"), adjustments)

    def test_monte_carlo_uses_map_probabilities(self):
        result = simulate_series(
            {"Ascent": 0.62, "Haven": 0.58, "Lotus": 0.60},
            best_of=3,
            performance_volatility=0.15,
            simulations=2000,
            scenario_count=300,
            seed=11,
        )

        self.assertGreater(result["team1_win_probability"], 0.5)
        self.assertLess(result["probability_low"], result["probability_high"])
        self.assertEqual(len(result["likely_map_order"]), 3)

    def test_sqlite_store_round_trips_dataframe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.sqlite3"
            frame = pd.DataFrame([{"match_id": 1, "team": "A"}])
            sync_dataframe("matches", frame, source="test", path=path)
            loaded = load_table("matches", path=path)

            self.assertEqual(loaded.to_dict("records"), frame.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
