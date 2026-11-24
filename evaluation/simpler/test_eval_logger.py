import json
import tempfile
import unittest
from pathlib import Path

from evaluation.simpler.eval_logger import EvalLogger


class EvalLoggerTaskSummaryTest(unittest.TestCase):
    def test_finalize_updates_task_summary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cases = [(3, False, 60), (1, True, 17), (2, True, 25)]
            for seed, success, steps in cases:
                logger = EvalLogger(root, task_name="test_task", seed=seed)
                logger.finalize(
                    metrics={
                        "task_name": "test_task",
                        "seed": seed,
                        "success": success,
                        "episode_length": steps,
                        "terminated": success,
                        "truncated": not success,
                    },
                    reference_info={},
                )

            summary_path = root / "test_task" / "task_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["num_episodes"], 3)
            self.assertEqual(summary["num_successes"], 2)
            self.assertAlmostEqual(summary["success_rate"], 2 / 3)
            self.assertEqual(summary["evaluated_seeds"], [1, 2, 3])
            self.assertEqual(summary["success_seeds"], [1, 2])
            self.assertEqual(
                summary["successful_episodes"],
                [{"seed": 1, "episode_length": 17}, {"seed": 2, "episode_length": 25}],
            )
            self.assertEqual(summary["success_steps_by_seed"], {"001": 17, "002": 25})
            self.assertEqual(summary["failed_seeds"], [3])
            self.assertEqual(summary["terminated_count"], 2)
            self.assertEqual(summary["truncated_count"], 1)
            self.assertEqual(summary["exception_count"], 0)


if __name__ == "__main__":
    unittest.main()
