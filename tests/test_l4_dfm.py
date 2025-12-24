"""
Unit tests for L4 DFM Analyzer.
Verifies threshold logic based on manufacturing_data.yaml (or defaults).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.core.dfm.analyzer import DFMAnalyzer


class TestL4DFM(unittest.TestCase):
    def setUp(self):
        # Mock configuration to ensure tests are deterministic
        # regardless of external yaml changes.
        self.mock_config = {
            "dfm_thresholds": {
                "min_wall_thickness_mm": 1.0,
                "max_slenderness_ratio": 8.0,
                "max_stock_removal_ratio": 0.80,
            }
        }

        with patch("src.core.dfm.analyzer.yaml.safe_load", return_value=self.mock_config):
            with patch("builtins.open", unittest.mock.mock_open(read_data="data")):
                self.analyzer = DFMAnalyzer()

    def test_thin_wall_detection(self):
        """Test detection of thin walls."""
        features = {
            "thin_walls_detected": True,
            "min_thickness_estimate": 0.5,  # Below 1.0 threshold
            "stock_removal_ratio": 0.5,
            "aspect_ratio_max_min": 2.0,
        }
        result = self.analyzer.analyze(features, "housing")

        # Should have THIN_WALL issue
        codes = [i["code"] for i in result["issues"]]
        self.assertIn("THIN_WALL", codes)
        self.assertLess(result["dfm_score"], 100)

    def test_high_waste_detection(self):
        """Test material removal ratio warning."""
        features = {
            "thin_walls_detected": False,
            "stock_removal_ratio": 0.9,  # Above 0.80 threshold
            "aspect_ratio_max_min": 2.0,
        }
        result = self.analyzer.analyze(features, "block")

        codes = [i["code"] for i in result["issues"]]
        self.assertIn("HIGH_WASTE", codes)

    def test_slender_shaft(self):
        """Test L/D ratio for shafts."""
        features = {
            "thin_walls_detected": False,
            "stock_removal_ratio": 0.5,
            "aspect_ratio_max_min": 12.0,  # Above 8.0 threshold
        }
        result = self.analyzer.analyze(features, "shaft")

        codes = [i["code"] for i in result["issues"]]
        self.assertIn("SLENDER_PART", codes)

    def test_perfect_part(self):
        """Test a part with no issues."""
        features = {
            "thin_walls_detected": False,
            "stock_removal_ratio": 0.5,
            "aspect_ratio_max_min": 2.0,
        }
        result = self.analyzer.analyze(features, "bracket")

        self.assertEqual(len(result["issues"]), 0)
        self.assertEqual(result["dfm_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
