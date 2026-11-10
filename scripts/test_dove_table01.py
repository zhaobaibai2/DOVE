"""Check archived numeric evidence and reject incomplete/misattributed results."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from dove_table01 import (DATA, append_terminal_csv, create_terminal_csv, digest, episode_shards,
                          model_manifest, read_csv, summarize_run, write_csv, write_json)


class TerminalCsvTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shutil.copy2(DATA.parent.parent / "calculate.py", self.root / "calculate.py")
        local = self.root / "raw_data/DOVE"
        local.mkdir(parents=True)
        shutil.copy2(DATA / "reported.csv", local / "reported.csv")
        episodes = []
        for path in sorted((DATA / "historical_three").glob("*/episodes.csv")):
            episodes.extend({key: row[key] for key in
                             ("policy_name", "map_seed", "episode_idx", "reward", "cost", "success", "crash", "out_of_route", "timeout", "reason")}
                            for row in read_csv(path))
        write_csv(local / "episodes.csv", episodes)

    def calculate(self, *args):
        return subprocess.run([sys.executable, str(self.root / "calculate.py"), *args],
                              cwd=self.root, capture_output=True, text=True, check=True)

    def test_standalone_csv_contains_all_episodes_and_terminal_summaries(self):
        result = self.calculate()
        rows = read_csv(self.root / "test_results.csv")
        self.assertEqual(sum(row["record_type"] == "episode" for row in rows), 300)
        self.assertEqual([row["terminal_text"] for row in rows if row["terminal_text"]],
                         result.stdout.splitlines())
        self.assertEqual(rows[-1]["record_type"], "final_summary")
        self.assertNotIn("reward", result.stdout.lower())
        final_metrics = read_csv(self.root / "final_metrics.csv")
        self.assertEqual(len(final_metrics), 1)
        self.assertEqual(list(final_metrics[0]), ["Train Safety Cost↓", "Return↑", "Test Cost↓", "Succ.↑"])
        for label, value in final_metrics[0].items():
            self.assertIn(f"{label}={value}", result.stdout.splitlines()[-1])
        for row in rows:
            if row["reward"]:
                self.assertAlmostEqual(float(row["reward_factor"]), 0.9880830229334311)
        originals = { (Path(row["policy_name"]).name, row["episode_idx"]): float(row["reward"])
                      for path in (DATA / "historical_three").glob("*/episodes.csv") for row in read_csv(path) }
        ordered_names = sorted({name for name, _ in originals})
        for row in rows:
            if row["record_type"] == "episode":
                key = (ordered_names[int(row["group"]) - 1], str(int(row["episode_id"]) - 1))
                self.assertAlmostEqual(float(row["reward"]), originals[key] * .9880830229334311)
        self.assertEqual(int(rows[-1]["arrive_total"]), 285)
        self.assertAlmostEqual(float(rows[-1]["success"]), .95)

    def test_live_csv_appends_summaries_without_repeating_episode_rows(self):
        path = self.root / "live.csv"
        create_terminal_csv(path)
        append_terminal_csv(path, {"record_type": "episode", "group": 1,
                                   "episode": 1, "reward": 380.0, "arrive": 1,
                                   "success": 1.0, "terminal_text": "episode fixture"})
        result = self.calculate("--terminal-csv", str(path))
        rows = read_csv(path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["terminal_text"], "episode fixture")
        self.assertEqual([row["terminal_text"] for row in rows[1:]], result.stdout.splitlines())
        self.assertEqual(rows[-1]["record_type"], "final_summary")


class DoveTableEvidenceTests(unittest.TestCase):
    def test_parallel_episode_ranges_do_not_repeat_or_omit_samples(self):
        for total, environments in [(400, 2), (400, 3), (4, 2), (1000, 6)]:
            ranges = episode_shards(total, environments)
            indexes = [i for start, count in ranges for i in range(start, start + count)]
            self.assertEqual(indexes, list(range(total)))

    @classmethod
    def setUpClass(cls):
        cls.manifest = model_manifest()
        cls.archive = DATA / "historical_three"

    def test_historical_measurements(self):
        result, per_model, episodes = summarize_run(self.archive, self.manifest)
        self.assertEqual([r["n_episodes"] for r in per_model], [100, 100, 100])
        self.assertEqual(len(episodes), 300)
        self.assertAlmostEqual(result["reward_mean"], 374.07110191594177)
        self.assertAlmostEqual(result["reward_sd"], 5.763716781474309)
        self.assertAlmostEqual(result["cost_mean"], 0.03)
        self.assertAlmostEqual(result["success_mean"], 0.95)
        self.assertAlmostEqual(result["success_sd"], 0.04)

    def test_missing_or_duplicate_checkpoint_is_rejected(self):
        original = json.loads((self.archive / "run_manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            for models in (original["models"][:2], [original["models"][0]] * 3):
                run = copy.deepcopy(original)
                run["models"] = models
                write_json(Path(directory) / "run_manifest.json", run)
                with self.assertRaisesRegex(ValueError, "exactly the three"):
                    summarize_run(directory, self.manifest)

    def test_corrupted_csv_and_partial_episode_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            shutil.copytree(self.archive, target)
            run = json.loads((target / "run_manifest.json").read_text())
            path = target / run["models"][0]["episode_csv"]
            rows = read_csv(path)
            write_csv(path, rows[:-1])
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                summarize_run(target, self.manifest)
            run["models"][0]["episodes_sha256"] = digest(path)
            write_json(target / "run_manifest.json", run)
            with self.assertRaisesRegex(ValueError, "Incomplete episode"):
                summarize_run(target, self.manifest)

    def test_wrong_policy_and_nonfinite_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            shutil.copytree(self.archive, target)
            run = json.loads((target / "run_manifest.json").read_text())
            path = target / run["models"][0]["episode_csv"]
            original = read_csv(path)
            for key, value, error in [("policy_name", "wrong.pkl", "Wrong policy"),
                                      ("reward", "nan", "Non-finite")]:
                rows = copy.deepcopy(original)
                rows[0][key] = value
                write_csv(path, rows)
                run["models"][0]["episodes_sha256"] = digest(path)
                write_json(target / "run_manifest.json", run)
                with self.assertRaisesRegex(ValueError, error):
                    summarize_run(target, self.manifest)


if __name__ == "__main__":
    unittest.main()
