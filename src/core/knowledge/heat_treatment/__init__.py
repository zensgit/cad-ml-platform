"""
Heat Treatment Knowledge Module.

Provides heat treatment parameters, process recommendations, and
material-specific treatment guidance for manufacturing.

Reference Standards:
- ASM Handbook Volume 4 - Heat Treating
- ISO 4545 - Metallic materials - Hardness testing
- GB/T 230 - Metallic materials - Hardness testing
"""

from .processes import (
    HeatTreatmentProcess,
    HeatTreatmentParameters,
    get_heat_treatment_parameters,
    get_quench_media,
    calculate_holding_time,
    recommend_process_for_hardness,
    HEAT_TREATMENT_DATABASE,
)
from .hardening import (
    HardenabilityClass,
    get_hardenability,
    get_tempering_temperature,
    calculate_hardness_after_tempering,
    recommend_quench_for_section,
    HARDENING_DATABASE,
)
from .annealing import (
    AnnealingType,
    get_annealing_parameters,
    get_stress_relief_parameters,
    calculate_annealing_cycle_time,
    recommend_annealing_for_purpose,
    ANNEALING_DATABASE,
)

__all__ = [
    # Heat treatment processes
    "HeatTreatmentProcess",
    "HeatTreatmentParameters",
    "get_heat_treatment_parameters",
    "get_quench_media",
    "calculate_holding_time",
    "recommend_process_for_hardness",
    "HEAT_TREATMENT_DATABASE",
    # Hardening
    "HardenabilityClass",
    "get_hardenability",
    "get_tempering_temperature",
    "calculate_hardness_after_tempering",
    "recommend_quench_for_section",
    "HARDENING_DATABASE",
    # Annealing
    "AnnealingType",
    "get_annealing_parameters",
    "get_stress_relief_parameters",
    "calculate_annealing_cycle_time",
    "recommend_annealing_for_purpose",
    "ANNEALING_DATABASE",
]
