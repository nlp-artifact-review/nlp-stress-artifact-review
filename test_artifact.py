import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reviewer_results"


class ArtifactIntegrityTests(unittest.TestCase):
    def test_matched_design_and_results_are_complete(self):
        design = json.loads((RESULTS / "reviewer_run_design.json").read_text())
        self.assertEqual(len(design["seeds"]), 10)
        self.assertEqual(design["effective_batch"], 8)
        self.assertEqual(design["epochs_max"], 5)
        rows = pd.read_csv(RESULTS / "reviewer_matched_test_by_seed.csv")
        self.assertEqual(len(rows), 60)
        self.assertEqual(len(rows.drop_duplicates(["backbone", "method", "seed"])), 60)
        self.assertEqual(len(list((RESULTS / "seed_logits").glob("*.npz"))), 60)

    def test_npz_contains_only_logits_and_binary_labels(self):
        expected_keys = {
            "validation_logits", "validation_labels", "test_logits", "test_labels"
        }
        for path in (RESULTS / "seed_logits").glob("*.npz"):
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), expected_keys)
                self.assertTrue(all(arrays[key].dtype.kind in "fiub"
                                    for key in expected_keys))

    def test_public_notebook_has_no_saved_execution_output(self):
        notebook = json.loads((ROOT / "Dreaddit_experiments.ipynb").read_text())
        self.assertGreaterEqual(len(notebook["cells"]), 167)
        self.assertNotIn("widgets", notebook["metadata"])
        self.assertNotIn("Krawczyk", json.dumps(notebook))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs"), [])
                self.assertIsNone(cell.get("execution_count"))

    def test_cpu_verification_command(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "verify_matched_results.py")],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        self.assertIn("PASS: 60 archived model test predictions", result.stdout)
        self.assertIn("holm_adjusted_p", result.stdout)


if __name__ == "__main__":
    unittest.main()
