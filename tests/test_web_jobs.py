import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from valorant_predictor.jobs import JobManager
from web.app import app, model_reports


class WebJobTests(unittest.TestCase):
    def test_index_renders_global_and_regional_power_rankings(self):
        ranking_payload = {
            "rows": [
                {
                    "global_rank": 1,
                    "regional_rank": 1,
                    "team": "Example Team",
                    "region": "VCT EMEA",
                    "power_score": 63.4,
                    "favored_against": 30,
                    "opponents_rated": 47,
                    "recent_win_rate": 0.6,
                    "evidence": 0.8,
                }
            ],
            "regions": ["VCT Americas", "VCT EMEA", "VCT Pacific", "VCT China"],
            "team_count": 48,
            "matchup_count": 1128,
            "generated_display": "22 Jul 2026, 17:00 UTC",
            "stale": False,
        }
        client = app.test_client()
        with patch("web.app.rankings_context", return_value=ranking_payload):
            response = client.get("/")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Team Power Rankings", body)
        self.assertIn('data-ranking-region="VCT EMEA"', body)
        self.assertIn("Example Team", body)

    def test_model_report_exposes_every_candidate_metric(self):
        reports = model_reports(
            {
                "team_metrics": {
                    "selected_candidate": "logistic_classifier",
                    "recommended_candidate": "logistic_classifier",
                    "enabled": True,
                    "elo_log_loss": 0.69,
                    "elo_brier_score": 0.248,
                    "elo_accuracy": 0.53,
                    "candidate_calibration": {
                        "logistic_classifier": {
                            "log_loss": 0.64,
                            "brier_score": 0.225,
                            "accuracy": 0.63,
                            "rolling_log_loss_mean": 0.65,
                            "rolling_log_loss_std": 0.01,
                            "rolling_brier_mean": 0.23,
                            "rolling_brier_std": 0.008,
                            "rolling_accuracy_mean": 0.61,
                            "rolling_accuracy_std": 0.02,
                            "rolling_folds": 3,
                            "rolling_wins": 3,
                            "rolling_baseline_log_loss_mean": 0.68,
                            "selection_log_loss": 0.6435,
                            "test_log_loss": 0.66,
                            "test_brier_score": 0.235,
                            "test_accuracy": 0.60,
                        },
                        "hgb_classifier": {
                            "log_loss": 0.65,
                            "brier_score": 0.23,
                            "accuracy": 0.61,
                            "rolling_log_loss_mean": 0.66,
                            "rolling_log_loss_std": 0.02,
                            "rolling_folds": 3,
                            "rolling_wins": 2,
                            "rolling_baseline_log_loss_mean": 0.68,
                            "selection_log_loss": 0.6535,
                            "test_log_loss": 0.67,
                            "test_brier_score": 0.24,
                            "test_accuracy": 0.58,
                        },
                    },
                }
            }
        )

        team_report = next(report for report in reports if report["task"] == "team")
        hgb = next(row for row in team_report["rows"] if row["id"] == "hgb_classifier")

        self.assertEqual(hgb["test_metric"], 0.67)
        self.assertEqual(hgb["test_brier"], 0.24)
        self.assertEqual(hgb["test_accuracy"], 0.58)
        self.assertEqual(hgb["rolling_variation"], 0.02)
        self.assertAlmostEqual(hgb["improvement_vs_baseline"], (0.68 - 0.66) / 0.68)

    def test_train_route_queues_selected_models(self):
        client = app.test_client()
        with patch("web.app.job_manager.start") as start:
            response = client.post(
                "/train",
                data={
                    "player_model": "extra_trees",
                    "team_model": "hgb_classifier",
                    "map_model": "map_form_baseline",
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(start.call_args.args[0], "train_and_evaluate")
        self.assertTrue(callable(start.call_args.args[1]))

    def test_database_route_queues_the_combined_update(self):
        client = app.test_client()
        with patch("web.app.job_manager.start") as start:
            response = client.post(
                "/jobs/database-update",
                data={"player_model": "auto", "team_model": "auto", "map_model": "auto"},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(start.call_args.args[0], "database_update")

    def test_job_manager_persists_completion_and_result_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = JobManager(Path(temp_dir) / "job.json")

            def target(progress):
                progress("sample", 1, 2, "Halfway")
                return {"message": "Finished sample job", "rows": 4}

            _, started = manager.start("sample", target)
            deadline = time.time() + 3
            while manager.snapshot().get("status") not in {"completed", "failed"}:
                if time.time() >= deadline:
                    self.fail("Background job did not finish")
                time.sleep(0.01)
            state = manager.snapshot()
            manager.executor.shutdown(wait=True)

        self.assertTrue(started)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["message"], "Finished sample job")
        self.assertEqual(state["result"]["rows"], 4)


if __name__ == "__main__":
    unittest.main()
