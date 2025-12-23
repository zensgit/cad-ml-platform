"""
DFM (Design for Manufacturability) Analyzer.

L4 Module that uses geometric features to assess manufacturability
and identify potential production issues.
"""

import logging
import os
import yaml
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Default config path
CONFIG_PATH = os.getenv("MANUFACTURING_CONFIG_PATH", "config/manufacturing_data.yaml")

class DFMAnalyzer:
    """
    AI-assisted DFM Analysis Engine.
    Loads DFM thresholds from a YAML config file.
    """

    def __init__(self):
        self.dfm_thresholds: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Loads DFM thresholds from the YAML config file."""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.dfm_thresholds = config.get("dfm_thresholds", {})
            logger.info("Successfully loaded DFM thresholds from %s", CONFIG_PATH)
        except FileNotFoundError:
            logger.warning(
                "Config file not found at %s. Using fallback hardcoded defaults for DFM.",
                CONFIG_PATH,
            )
            self._set_default_hardcoded_thresholds()
        except yaml.YAMLError as e:
            logger.error(
                "Error parsing YAML config at %s: %s. Using fallback hardcoded defaults for DFM.",
                CONFIG_PATH,
                e,
            )
            self._set_default_hardcoded_thresholds()
        except Exception as e:
            logger.error(
                "Unexpected error loading config from %s: %s. Using fallback hardcoded defaults "
                "for DFM.",
                CONFIG_PATH,
                e,
            )
            self._set_default_hardcoded_thresholds()

    def _set_default_hardcoded_thresholds(self):
        """Sets hardcoded default threshold values if config loading fails."""
        self.dfm_thresholds = {
            "min_wall_thickness_mm": 0.8,
            "max_slenderness_ratio": 10.0,
            "max_stock_removal_ratio": 0.85,
        }

    def analyze(self, dfm_features: Dict[str, Any], part_type: str) -> Dict[str, Any]:
        """
        Analyze features and return DFM report.
        """
        issues = []
        score = 100.0

        # 1. Thin Wall Check
        if dfm_features.get("thin_walls_detected"):
            min_thk = dfm_features.get("min_thickness_estimate", 0)
            threshold = self.dfm_thresholds.get("min_wall_thickness_mm", 0.8)
            if min_thk < threshold:
                issues.append(
                    {
                        "severity": "high",
                        "code": "THIN_WALL",
                        "message": (
                            f"Potential thin walls detected (~{min_thk:.2f}mm). "
                            f"Below threshold ({threshold:.2f}mm). "
                            "May cause warping or printing failures."
                        ),
                    }
                )
                score -= 20

        # 2. Material Removal Check (CNC)
        removal_ratio = dfm_features.get("stock_removal_ratio", 0)
        threshold = self.dfm_thresholds.get("max_stock_removal_ratio", 0.85)
        if removal_ratio > threshold:
            issues.append(
                {
                    "severity": "medium",
                    "code": "HIGH_WASTE",
                    "message": (
                        f"Requires removing {removal_ratio*100:.1f}% of stock material. "
                        f"Above threshold ({threshold*100:.1f}%). Consider casting or "
                        "additive manufacturing."
                    ),
                }
            )
            score -= 10

        # 3. Aspect Ratio Check (Lathe/Turning)
        ar = dfm_features.get("aspect_ratio_max_min", 0)
        threshold = self.dfm_thresholds.get("max_slenderness_ratio", 10.0)
        if part_type == "shaft" and ar > threshold:
            issues.append(
                {
                    "severity": "medium",
                    "code": "SLENDER_PART",
                    "message": (
                        f"Part is very slender (L/D = {ar:.1f}). Above threshold "
                        f"({threshold:.1f}). Risk of vibration/deflection during turning."
                    ),
                }
            )
            score -= 15

        if score > 80:
            manufacturability = "high"
        elif score > 50:
            manufacturability = "medium"
        else:
            manufacturability = "low"

        return {
            "dfm_score": max(0.0, score),
            "issues": issues,
            "manufacturability": manufacturability,
        }

# Singleton
_dfm = DFMAnalyzer()

def get_dfm_analyzer():
    return _dfm
