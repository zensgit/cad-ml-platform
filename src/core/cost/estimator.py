import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config path
CONFIG_PATH = os.getenv("MANUFACTURING_CONFIG_PATH", "config/manufacturing_data.yaml")


class CostEstimator:
    """
    Parametric Cost Estimation Engine.
    Loads manufacturing data (material prices, machine rates) from a YAML config file.
    """

    def __init__(self):
        self.materials_config: Dict[str, Any] = {}
        self.machine_rates_config: Dict[str, float] = {}
        self._load_config()

    def _load_config(self):
        """Loads manufacturing data from the YAML config file."""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.materials_config = config.get("materials", {})
                self.machine_rates_config = config.get("machine_hourly_rates", {})
            logger.info("Successfully loaded manufacturing data from %s", CONFIG_PATH)
        except FileNotFoundError:
            logger.warning(
                "Config file not found at %s. Using fallback hardcoded defaults.",
                CONFIG_PATH,
            )
            self._set_default_hardcoded_config()
        except yaml.YAMLError as e:
            logger.error(
                "Error parsing YAML config at %s: %s. Using fallback hardcoded defaults.",
                CONFIG_PATH,
                e,
            )
            self._set_default_hardcoded_config()
        except Exception as e:
            logger.error(
                "Unexpected error loading config from %s: %s. Using fallback hardcoded defaults.",
                CONFIG_PATH,
                e,
            )
            self._set_default_hardcoded_config()

    def _set_default_hardcoded_config(self):
        """Sets hardcoded default values if config loading fails."""
        self.materials_config = {
            "steel": {
                "price_per_kg_usd": 2.5,
                "density_kg_per_m3": 7850,
            },
            "aluminum": {
                "price_per_kg_usd": 4.0,
                "density_kg_per_m3": 2700,
            },
            "titanium": {
                "price_per_kg_usd": 40.0,
                "density_kg_per_m3": 4500,
            },
            "plastic": {
                "price_per_kg_usd": 15.0,
                "density_kg_per_m3": 1200,
            },
            "unknown": {
                "price_per_kg_usd": 3.0,
                "density_kg_per_m3": 7850,
            },
        }
        self.machine_rates_config = {
            "cnc_milling": 60.0,
            "cnc_lathe": 50.0,
            "5_axis": 120.0,
            "additive_manufacturing": 40.0,
            "general_machining": 55.0,
        }

    def estimate(
        self,
        features_3d: Dict[str, Any],
        process_rec: Dict[str, Any],
        material: str = "steel",
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate estimated cost.
        """
        mat_key = self._resolve_material(material)
        mat_data = self.materials_config.get(mat_key, self.materials_config["unknown"])
        mat_price = mat_data["price_per_kg_usd"]
        density = mat_data["density_kg_per_m3"]

        # 1. Material Cost
        volume_mm3 = features_3d.get("volume", 0)
        weight_kg = (volume_mm3 * 1e-9) * density  # Convert mm^3 to m^3

        # Add waste factor (Stock Removal)
        removal_ratio = features_3d.get("stock_removal_ratio", 0.0)
        stock_weight_kg = (
            weight_kg / (1.0 - min(0.9, removal_ratio)) if removal_ratio < 1.0 else weight_kg * 1.2
        )

        material_cost = stock_weight_kg * mat_price

        # 2. Machining Cost
        process_type = process_rec.get("process", "general_machining")
        rate = self.machine_rates_config.get(process_type, 55.0)

        # Heuristic Time Estimation
        base_setup_hrs = 0.5

        removal_vol_mm3 = stock_weight_kg / density * 1e9 - volume_mm3
        removal_hours = (removal_vol_mm3 / 1000) / (100 * 60)  # Rough estimate

        complexity_mult = 1.0
        if process_rec.get("method") == "5_axis":
            complexity_mult = 2.5

        machining_hours = (base_setup_hrs / batch_size) + (removal_hours * complexity_mult)
        machining_cost = machining_hours * rate

        total_cost = material_cost + machining_cost

        return {
            "total_unit_cost": round(total_cost, 2),
            "breakdown": {
                "material_cost": round(material_cost, 2),
                "machining_cost": round(machining_cost, 2),
                "setup_amortized": round((base_setup_hrs * rate) / batch_size, 2),
            },
            "parameters": {
                "stock_weight_kg": round(stock_weight_kg, 3),
                "est_cycle_time_min": round(machining_hours * 60, 1),
                "material_rate": mat_price,
                "machine_rate": rate,
            },
            "currency": "USD",
        }

    def _resolve_material(self, mat: str) -> str:
        """Resolves raw material string to a canonical key for config lookup."""
        mat_lower = mat.lower()
        for key in self.materials_config:
            if key in mat_lower:
                return key
        return "unknown"


# Singleton
_estimator = CostEstimator()


def get_cost_estimator() -> CostEstimator:
    return _estimator
