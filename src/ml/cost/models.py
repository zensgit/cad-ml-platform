"""Pydantic models for the manufacturing cost estimation module."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CostEstimateRequest(BaseModel):
    """Input parameters for a single cost estimate."""

    material: str = Field(
        default="steel",
        description="Material key matching config/cost_model.yaml materials section",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of identical parts in the production batch",
    )
    tolerance_grade: str = Field(
        default="IT8",
        description="ISO tolerance grade (IT6 through IT12)",
    )
    surface_finish: str = Field(
        default="Ra3.2",
        description="Surface roughness target (Ra0.8, Ra1.6, Ra3.2, Ra6.3, Ra12.5)",
    )
    bounding_volume_mm3: float = Field(
        default=0.0,
        ge=0.0,
        description="Bounding-box volume of the part in cubic millimetres",
    )
    entity_count: int = Field(
        default=0,
        ge=0,
        description="Number of geometric entities in the CAD model",
    )
    complexity_score: Optional[float] = Field(
        default=None,
        description="Pre-computed complexity score (0-1). Calculated automatically if omitted.",
    )


class CostBreakdown(BaseModel):
    """Itemised cost breakdown for a single part."""

    material_cost: float = Field(description="Raw material cost")
    machining_cost: float = Field(description="Machining / processing cost")
    setup_cost: float = Field(description="Setup cost amortised over batch")
    overhead: float = Field(description="Overhead surcharge")
    total: float = Field(description="Total estimated cost per part")
    currency: str = Field(default="CNY", description="Currency code")


class CostEstimateResponse(BaseModel):
    """Complete cost estimation result with confidence interval."""

    estimate: CostBreakdown = Field(description="Best-estimate cost breakdown")
    optimistic: CostBreakdown = Field(description="Lower-bound (optimistic) estimate")
    pessimistic: CostBreakdown = Field(description="Upper-bound (pessimistic) estimate")
    process_route: List[str] = Field(
        default_factory=list,
        description="Recommended manufacturing process sequence",
    )
    complexity_score: float = Field(
        default=0.0,
        description="Computed geometric complexity (0-1)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimation confidence (0-1) based on available input data",
    )
    reasoning: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations for each cost component",
    )
