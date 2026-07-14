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
from web.app import app


class WebJobTests(unittest.TestCase):
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
