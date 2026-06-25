# -*- coding: utf-8 -*-
"""Модульные тесты программы SimpleAnalysis."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from simple_analysis import (
    HIGH,
    LOW,
    N_VALUES,
    build_dataframe,
    clean_series,
    compute_statistics,
    generate_series,
    round_to_hundreds,
    run_analysis,
)


class SimpleAnalysisTests(unittest.TestCase):
    def test_generated_series_has_required_size_type_and_range(self) -> None:
        series = generate_series()
        self.assertEqual(len(series), N_VALUES)
        self.assertTrue(pd.api.types.is_integer_dtype(series.dtype))
        self.assertGreaterEqual(int(series.min()), LOW)
        self.assertLessEqual(int(series.max()), HIGH)

    def test_cleaning_removes_invalid_values_and_keeps_valid_values(self) -> None:
        raw = pd.Series([0, "15", "abc", None, 10_001, -10_001, -7])
        self.assertEqual(clean_series(raw).tolist(), [0, 15, -7])

    def test_statistics_are_correct_for_control_sample(self) -> None:
        series = pd.Series([-100, 0, 100, 100])
        statistics = compute_statistics(series)
        self.assertEqual(statistics["minimum"], -100)
        self.assertEqual(statistics["maximum"], 100)
        self.assertEqual(statistics["total_sum"], 100)
        self.assertEqual(statistics["repeated_values"], 1)
        self.assertEqual(statistics["extra_repeats"], 1)
        self.assertAlmostEqual(statistics["std_dev"], 82.9156198, places=6)

    def test_rounding_uses_mathematical_rule(self) -> None:
        self.assertEqual(round_to_hundreds(149), 100)
        self.assertEqual(round_to_hundreds(150), 200)
        self.assertEqual(round_to_hundreds(-149), -100)
        self.assertEqual(round_to_hundreds(-150), -200)

    def test_dataframe_contains_correct_sorted_columns(self) -> None:
        frame = build_dataframe(pd.Series([3, -1, 2], name="values"))
        self.assertEqual(frame["sorted_asc"].tolist(), [-1, 2, 3])
        self.assertEqual(frame["sorted_desc"].tolist(), [3, 2, -1])

    def test_full_scenario_creates_all_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_analysis(Path(temp_dir), show_plots=False)
            expected = [result["csv_path"], result["txt_path"], result["results_path"], *result["plots"]]
            for path in expected:
                self.assertTrue(path.exists(), f"Не создан файл {path}")
                self.assertGreater(path.stat().st_size, 0, f"Пустой файл {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
