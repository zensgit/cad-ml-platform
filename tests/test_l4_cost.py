"""
Unit tests for L4 Cost Estimator.
Verifies calculation logic against known inputs.
"""

import unittest
from unittest.mock import patch

from src.core.cost.estimator import CostEstimator


class TestL4Cost(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.mock_config = {
            "materials": {
                "steel": {"price_per_kg_usd": 2.0, "density_kg_per_m3": 8000},
                "unknown": {"price_per_kg_usd": 3.0, "density_kg_per_m3": 7850},
            },
            "machine_hourly_rates": {"cnc_milling": 100.0},
        }

        with patch("src.core.cost.estimator.yaml.safe_load", return_value=self.mock_config):
            with patch("builtins.open", unittest.mock.mock_open(read_data="data")):
                self.estimator = CostEstimator()

    def test_cost_calculation_steel_cube(self):
        """
        Test logic for a simple steel cube.
        100mm x 100mm x 100mm
        Volume = 1,000,000 mm^3 = 0.001 m^3
        Density = 8000 kg/m^3
        Weight = 8.0 kg
        """
        features_3d = {"volume": 1000000.0, "stock_removal_ratio": 0.5}  # 50% waste
        process_rec = {"process": "cnc_milling", "method": "3_axis"}

        result = self.estimator.estimate(features_3d, process_rec, material="steel", batch_size=1)

        # 1. Material Verification
        # Final Weight = 8.0 kg
        # Removal 0.5 -> Stock = 8.0 / (1 - 0.5) = 16.0 kg
        # Mat Cost = 16.0 * $2.0 = $32.0
        mat_cost = result["breakdown"]["material_cost"]
        self.assertAlmostEqual(mat_cost, 32.0, delta=0.1)

        # 2. Machining Verification
        # Rate = $100/hr
        # Setup = 0.5 hr / 1 = 0.5 hr
        # Removal Vol = 16kg - 8kg = 8kg steel
        # 8kg steel = 0.001 m^3 = 1,000,000 mm^3
        # Removal Rate heuristic (in code) = 100 cm^3/min = 100,000 mm^3/min
        # Removal Time = 1,000,000 / 100,000 = 10 min = 0.166 hrs
        # Total Time = 0.5 + 0.166 = 0.666 hrs
        # Cost = 0.666 * 100 = $66.6
        mach_cost = result["breakdown"]["machining_cost"]
        self.assertTrue(50.0 < mach_cost < 80.0)  # Heuristics might vary slightly

        # Total
        self.assertAlmostEqual(result["total_unit_cost"], mat_cost + mach_cost, delta=0.1)

    def test_batch_scaling(self):
        """Test that setup cost amortizes with batch size."""
        features = {"volume": 1000, "stock_removal_ratio": 0.1}
        proc = {"process": "cnc_milling"}

        res_1 = self.estimator.estimate(features, proc, batch_size=1)
        res_10 = self.estimator.estimate(features, proc, batch_size=10)

        # Setup cost per unit should drop by 10x
        setup_1 = res_1["breakdown"]["setup_amortized"]
        setup_10 = res_10["breakdown"]["setup_amortized"]

        self.assertAlmostEqual(setup_1 / 10, setup_10, delta=0.5)


if __name__ == "__main__":
    unittest.main()
