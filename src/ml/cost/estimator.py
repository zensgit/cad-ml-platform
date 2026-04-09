"""Manufacturing cost estimator.

Loads a YAML cost model and produces itemised cost breakdowns for parts
described by geometric features (bounding volume, entity count, complexity).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.ml.cost.models import CostBreakdown, CostEstimateRequest, CostEstimateResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration embedded as fallback so the estimator works even when
# the YAML file is missing or incomplete.
# ---------------------------------------------------------------------------

_DEFAULT_MATERIALS: Dict[str, Dict[str, float]] = {
    "steel": {"price_per_kg": 6.5, "density_kg_per_m3": 7850, "machinability": 0.6},
    "stainless_steel": {"price_per_kg": 22.0, "density_kg_per_m3": 7930, "machinability": 0.4},
    "aluminum": {"price_per_kg": 28.0, "density_kg_per_m3": 2700, "machinability": 0.85},
    "titanium": {"price_per_kg": 280.0, "density_kg_per_m3": 4500, "machinability": 0.25},
    "plastic_abs": {"price_per_kg": 18.0, "density_kg_per_m3": 1040, "machinability": 0.95},
}

_DEFAULT_MACHINES: Dict[str, Dict[str, float]] = {
    "cnc_3axis": {"hourly_rate": 80, "setup_minutes": 30},
    "cnc_5axis": {"hourly_rate": 200, "setup_minutes": 60},
    "cnc_lathe": {"hourly_rate": 60, "setup_minutes": 20},
    "wire_edm": {"hourly_rate": 120, "setup_minutes": 45},
    "grinding": {"hourly_rate": 100, "setup_minutes": 25},
}

_DEFAULT_TOLERANCE: Dict[str, float] = {
    "IT6": 2.0, "IT7": 1.5, "IT8": 1.0, "IT9": 0.8,
    "IT10": 0.6, "IT11": 0.5, "IT12": 0.4,
}

_DEFAULT_SURFACE: Dict[str, float] = {
    "Ra0.8": 2.5, "Ra1.6": 1.8, "Ra3.2": 1.0, "Ra6.3": 0.7, "Ra12.5": 0.5,
}

_DEFAULT_SETUP_BASE_COST: float = 200.0
_DEFAULT_OVERHEAD_RATE: float = 0.15

# Volume thresholds (mm^3) for process-route heuristic
_SMALL_VOLUME_THRESHOLD = 50_000.0       # 50 cm^3
_MEDIUM_VOLUME_THRESHOLD = 500_000.0     # 500 cm^3

# Materials that are best suited for specific primary processes
_LATHE_PREFERRED_MATERIALS = {"steel", "stainless_steel", "aluminum"}
_EDM_PREFERRED_MATERIALS = {"titanium", "stainless_steel"}


class CostEstimator:
    """Rule-based manufacturing cost estimator backed by a YAML configuration.

    Usage::

        estimator = CostEstimator()
        response = estimator.estimate(CostEstimateRequest(
            material="aluminum",
            bounding_volume_mm3=12000,
            entity_count=45,
        ))
        print(response.estimate.total, response.estimate.currency)
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config_path: str = "config/cost_model.yaml") -> None:
        self._config: Dict[str, Any] = {}
        self._materials: Dict[str, Dict[str, float]] = dict(_DEFAULT_MATERIALS)
        self._machines: Dict[str, Dict[str, float]] = dict(_DEFAULT_MACHINES)
        self._tolerance_factor: Dict[str, float] = dict(_DEFAULT_TOLERANCE)
        self._surface_factor: Dict[str, float] = dict(_DEFAULT_SURFACE)
        self._setup_base_cost: float = _DEFAULT_SETUP_BASE_COST
        self._overhead_rate: float = _DEFAULT_OVERHEAD_RATE

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load YAML configuration, falling back to embedded defaults."""
        path = Path(config_path)
        if not path.is_file():
            logger.warning(
                "Cost model config not found at %s; using built-in defaults", config_path
            )
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._config = yaml.safe_load(fh) or {}
        except Exception:
            logger.exception("Failed to parse cost model config at %s", config_path)
            return

        # Materials
        raw_materials = self._config.get("materials")
        if isinstance(raw_materials, dict):
            for key, props in raw_materials.items():
                if isinstance(props, dict):
                    self._materials[key] = {
                        "price_per_kg": float(props.get("price_per_kg", 0)),
                        "density_kg_per_m3": float(props.get("density_kg_per_m3", 0)),
                        "machinability": float(props.get("machinability", 0.5)),
                    }

        # Machines
        raw_machines = self._config.get("machines")
        if isinstance(raw_machines, dict):
            for key, props in raw_machines.items():
                if isinstance(props, dict):
                    self._machines[key] = {
                        "hourly_rate": float(props.get("hourly_rate", 0)),
                        "setup_minutes": float(props.get("setup_minutes", 0)),
                    }

        # Scalar parameters
        if "setup_base_cost" in self._config:
            self._setup_base_cost = float(self._config["setup_base_cost"])
        if "overhead_rate" in self._config:
            self._overhead_rate = float(self._config["overhead_rate"])

        # Tolerance / surface lookup tables
        raw_tol = self._config.get("tolerance_factor")
        if isinstance(raw_tol, dict):
            self._tolerance_factor = {k: float(v) for k, v in raw_tol.items()}
        raw_surf = self._config.get("surface_factor")
        if isinstance(raw_surf, dict):
            self._surface_factor = {k: float(v) for k, v in raw_surf.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, request: CostEstimateRequest) -> CostEstimateResponse:
        """Produce a full cost estimate for the given request."""
        reasoning: List[str] = []

        # 1. Resolve material properties
        mat_key = request.material
        mat = self._materials.get(mat_key)
        if mat is None:
            mat = self._materials.get("steel", _DEFAULT_MATERIALS["steel"])
            reasoning.append(
                f"Unknown material '{mat_key}'; falling back to steel defaults."
            )
            mat_key = "steel"

        density = mat["density_kg_per_m3"]
        price_per_kg = mat["price_per_kg"]
        machinability = mat["machinability"]

        # 2. Complexity
        if request.complexity_score is not None:
            complexity = max(0.0, min(1.0, request.complexity_score))
            reasoning.append(
                f"Using provided complexity score: {complexity:.2f}."
            )
        else:
            complexity, complexity_reasons = self._calculate_complexity(
                request.entity_count, request.bounding_volume_mm3
            )
            reasoning.extend(complexity_reasons)

        # 3. Material cost
        #    volume_m3 = bounding_volume_mm3 * 1e-9
        #    waste_factor increases with complexity (more scrap)
        waste_factor = 1.1 + 0.4 * complexity  # range [1.1, 1.5]
        volume_m3 = request.bounding_volume_mm3 * 1e-9
        mass_kg = volume_m3 * density
        material_cost = mass_kg * price_per_kg * waste_factor
        reasoning.append(
            f"Material: {mass_kg:.4f} kg * {price_per_kg} CNY/kg "
            f"* {waste_factor:.2f} waste factor = {material_cost:.2f} CNY."
        )

        # 4. Process route
        process_route = self._determine_process_route(mat_key, request.bounding_volume_mm3)
        reasoning.append(f"Process route: {', '.join(process_route)}.")

        # 5. Machining time & cost
        primary_process = process_route[0] if process_route else "cnc_3axis"
        machining_time_hrs = self._calculate_machining_time(
            complexity, primary_process, machinability
        )

        # Apply tolerance and surface factors
        tol_factor = self._tolerance_factor.get(request.tolerance_grade, 1.0)
        surf_factor = self._surface_factor.get(request.surface_finish, 1.0)
        adjusted_time = machining_time_hrs * tol_factor * surf_factor
        reasoning.append(
            f"Machining time: {machining_time_hrs:.2f} h "
            f"* tolerance factor {tol_factor} "
            f"* surface factor {surf_factor} "
            f"= {adjusted_time:.2f} h."
        )

        machine_cfg = self._machines.get(primary_process, self._machines.get("cnc_3axis", _DEFAULT_MACHINES["cnc_3axis"]))
        hourly_rate = machine_cfg["hourly_rate"]
        machining_cost = adjusted_time * hourly_rate
        reasoning.append(
            f"Machining cost: {adjusted_time:.2f} h * {hourly_rate} CNY/h "
            f"= {machining_cost:.2f} CNY."
        )

        # 6. Setup cost (amortised across batch)
        setup_cost = self._setup_base_cost / max(request.batch_size, 1)
        reasoning.append(
            f"Setup cost: {self._setup_base_cost} CNY / {request.batch_size} "
            f"= {setup_cost:.2f} CNY per part."
        )

        # 7. Overhead
        subtotal = material_cost + machining_cost + setup_cost
        overhead = subtotal * self._overhead_rate
        reasoning.append(
            f"Overhead: ({material_cost:.2f} + {machining_cost:.2f} + "
            f"{setup_cost:.2f}) * {self._overhead_rate} = {overhead:.2f} CNY."
        )

        # 8. Total
        total = subtotal + overhead

        estimate = CostBreakdown(
            material_cost=round(material_cost, 2),
            machining_cost=round(machining_cost, 2),
            setup_cost=round(setup_cost, 2),
            overhead=round(overhead, 2),
            total=round(total, 2),
        )

        # 9. Optimistic / pessimistic
        optimistic = self._scale_breakdown(estimate, 0.8)
        pessimistic = self._scale_breakdown(estimate, 1.3)

        # 10. Confidence
        confidence = self._calculate_confidence(request)
        reasoning.append(f"Confidence: {confidence:.2f}.")

        return CostEstimateResponse(
            estimate=estimate,
            optimistic=optimistic,
            pessimistic=pessimistic,
            process_route=process_route,
            complexity_score=round(complexity, 4),
            confidence=confidence,
            reasoning=reasoning,
        )

    @property
    def materials(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of the loaded material properties."""
        return dict(self._materials)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_process_route(
        self, material: str, volume_mm3: float
    ) -> List[str]:
        """Select an ordered list of manufacturing processes.

        Heuristic rules:
        - Small volume + lathe-friendly material -> cnc_lathe, grinding
        - Titanium / stainless with tight features -> wire_edm, cnc_5axis
        - Medium complexity -> cnc_3axis
        - Large or complex -> cnc_5axis, grinding
        """
        route: List[str] = []

        if volume_mm3 <= _SMALL_VOLUME_THRESHOLD:
            if material in _LATHE_PREFERRED_MATERIALS:
                route.append("cnc_lathe")
            else:
                route.append("cnc_3axis")
        elif volume_mm3 <= _MEDIUM_VOLUME_THRESHOLD:
            route.append("cnc_3axis")
        else:
            route.append("cnc_5axis")

        # Secondary processes
        if material in _EDM_PREFERRED_MATERIALS and volume_mm3 > _SMALL_VOLUME_THRESHOLD:
            route.append("wire_edm")

        if volume_mm3 > _MEDIUM_VOLUME_THRESHOLD:
            route.append("grinding")

        return route

    def _calculate_machining_time(
        self,
        complexity: float,
        process_type: str,
        machinability: float,
    ) -> float:
        """Estimate machining hours from complexity and material properties.

        Base time grows with complexity and shrinks with machinability.
        Different processes have inherent speed multipliers.
        """
        # Base hours: 0.5 to 5 h depending on complexity
        base_hours = 0.5 + 4.5 * complexity

        # Process speed multiplier (slower processes take longer)
        process_multiplier = {
            "cnc_3axis": 1.0,
            "cnc_5axis": 1.4,
            "cnc_lathe": 0.8,
            "wire_edm": 1.6,
            "grinding": 0.6,
        }.get(process_type, 1.0)

        # Machinability inversely affects time
        safe_machinability = max(machinability, 0.1)
        time_hrs = base_hours * process_multiplier / safe_machinability

        return max(time_hrs, 0.1)  # floor at 6 minutes

    def _calculate_complexity(
        self,
        entity_count: int,
        bounding_volume_mm3: float,
        shape_entropy: Optional[float] = None,
    ) -> Tuple[float, List[str]]:
        """Derive a normalised complexity score (0-1) from geometric proxies.

        Factors:
        - Entity density (entities per cm^3 of bounding volume)
        - Absolute entity count (log-scaled)
        - Optional shape entropy
        """
        reasons: List[str] = []

        # Entity-count component (log-scaled, saturates around 1000 entities)
        if entity_count > 0:
            count_score = min(math.log10(entity_count + 1) / 3.0, 1.0)
        else:
            count_score = 0.0
        reasons.append(
            f"Entity count {entity_count} -> count score {count_score:.2f}."
        )

        # Density component
        if bounding_volume_mm3 > 0 and entity_count > 0:
            volume_cm3 = bounding_volume_mm3 / 1000.0
            density = entity_count / max(volume_cm3, 0.001)
            density_score = min(density / 10.0, 1.0)
        else:
            density_score = 0.0
        reasons.append(f"Density score: {density_score:.2f}.")

        # Entropy component (optional)
        if shape_entropy is not None:
            entropy_score = max(0.0, min(shape_entropy, 1.0))
            combined = 0.35 * count_score + 0.35 * density_score + 0.30 * entropy_score
            reasons.append(f"Shape entropy {entropy_score:.2f} included.")
        else:
            combined = 0.50 * count_score + 0.50 * density_score

        complexity = max(0.0, min(combined, 1.0))
        reasons.append(f"Final complexity: {complexity:.2f}.")
        return complexity, reasons

    @staticmethod
    def _scale_breakdown(base: CostBreakdown, factor: float) -> CostBreakdown:
        """Create a new breakdown by uniformly scaling every cost component."""
        return CostBreakdown(
            material_cost=round(base.material_cost * factor, 2),
            machining_cost=round(base.machining_cost * factor, 2),
            setup_cost=round(base.setup_cost * factor, 2),
            overhead=round(base.overhead * factor, 2),
            total=round(base.total * factor, 2),
        )

    @staticmethod
    def _calculate_confidence(request: CostEstimateRequest) -> float:
        """Compute confidence (0-1) based on how much information was provided."""
        conf = 0.30  # base confidence
        if request.bounding_volume_mm3 > 0:
            conf += 0.20
        if request.complexity_score is not None:
            conf += 0.20
        if request.tolerance_grade and request.tolerance_grade != "IT8":
            conf += 0.15
        elif request.tolerance_grade == "IT8":
            # Default grade contributes a small amount
            conf += 0.05
        if request.surface_finish and request.surface_finish != "Ra3.2":
            conf += 0.15
        elif request.surface_finish == "Ra3.2":
            conf += 0.05
        return min(round(conf, 2), 1.0)
