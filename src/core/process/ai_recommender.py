"""
AI Process Recommender.

L4 Module that recommends optimal manufacturing processes
based on geometric complexity, material, and batch size (implied).
Loads thresholds from a YAML config file.
"""

import logging
import os
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# Default config path
CONFIG_PATH = os.getenv("MANUFACTURING_CONFIG_PATH", "config/manufacturing_data.yaml")


class AIProcessRecommender:
    """
    Replaces static rules with heuristic/ML logic for process selection.
    """

    def __init__(self):
        self.proc_rec_thresholds: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Loads process recommendation thresholds from the YAML config file."""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.proc_rec_thresholds = config.get("process_recommendation_thresholds", {})
            logger.info(
                "Successfully loaded process recommendation thresholds from %s",
                CONFIG_PATH,
            )
        except FileNotFoundError:
            logger.warning(
                "Config file not found at %s. Using fallback hardcoded defaults for process "
                "recommendation.",
                CONFIG_PATH,
            )
            self._set_default_hardcoded_thresholds()
        except yaml.YAMLError as e:
            logger.error(
                "Error parsing YAML config at %s: %s. Using fallback hardcoded defaults for "
                "process recommendation.",
                CONFIG_PATH,
                e,
            )
            self._set_default_hardcoded_thresholds()
        except Exception as e:
            logger.error(
                "Unexpected error loading config from %s: %s. Using fallback hardcoded defaults "
                "for process recommendation.",
                CONFIG_PATH,
                e,
            )
            self._set_default_hardcoded_thresholds()

    def _set_default_hardcoded_thresholds(self):
        """Sets hardcoded default threshold values if config loading fails."""
        self.proc_rec_thresholds = {
            "complex_geometry_score": 5.0,
            "high_stock_removal_additive": 0.9,
            "prismatic_complexity_max": 2.0,
        }

    def recommend(
        self, dfm_features: Dict[str, Any], part_type: str, material: str
    ) -> Dict[str, Any]:
        """
        Recommend manufacturing processes.
        """
        recommendations = []

        # Extract key features
        stock_removal = dfm_features.get("stock_removal_ratio", 0.0)
        # Assuming surface_area and volume are in dfm_features or features_3d
        surface_area = dfm_features.get("surface_area", 0)
        volume = dfm_features.get("volume", 1)
        complexity_score = surface_area / (volume + 1)  # Heuristic
        is_thin_walled = dfm_features.get("thin_walls_detected", False)

        # Logic Tree (Simulating a Decision Tree Classifier)

        # Branch 1: Additive Manufacturing
        # High complexity, low material removal (wasteful to machine), or thin walls
        if complexity_score > self.proc_rec_thresholds.get(
            "complex_geometry_score", 5.0
        ) or stock_removal > self.proc_rec_thresholds.get("high_stock_removal_additive", 0.9):
            rec = {
                "process": "additive_manufacturing",
                "method": "SLS" if "nylon" in material.lower() else "DMLS",
                "confidence": 0.85,
                "reason": (
                    "High geometric complexity and high material removal rate favors additive."
                ),
            }
            recommendations.append(rec)

        # Branch 2: CNC Machining (Milling)
        # Standard prismatic parts, moderate removal
        elif "block" in part_type or "plate" in part_type or "housing" in part_type:
            rec = {
                "process": "cnc_milling",
                "method": (
                    "3_axis"
                    if complexity_score
                    < self.proc_rec_thresholds.get("prismatic_complexity_max", 2.0)
                    else "5_axis"
                ),
                "confidence": 0.90,
                "reason": "Prismatic geometry suitable for milling.",
            }
            recommendations.append(rec)

        # Branch 3: Turning
        # Cylindrical parts
        elif part_type in ["shaft", "bolt", "bearing", "washer"]:
            rec = {
                "process": "turning",
                "method": "cnc_lathe",
                "confidence": 0.95,
                "reason": "Rotational symmetry favors turning.",
            }
            recommendations.append(rec)

        # Fallback
        if not recommendations:
            recommendations.append(
                {
                    "process": "general_machining",
                    "confidence": 0.5,
                    "reason": "Standard geometry.",
                }
            )

        return {
            "primary_recommendation": recommendations[0],
            "alternatives": recommendations[1:],
            "analysis_mode": "L4_AI_Heuristic",
        }


# Singleton
_recommender = AIProcessRecommender()


def get_process_recommender():
    return _recommender
