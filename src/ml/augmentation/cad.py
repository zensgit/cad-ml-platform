"""
CAD-specific augmentations.

Provides augmentations tailored for CAD/DXF data.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CADAugmentation(ABC):
    """Base class for CAD-specific augmentations."""

    def __init__(self, p: float = 0.5):
        """
        Initialize augmentation.

        Args:
            p: Probability of applying
        """
        self.p = p

    def __call__(self, data: Any) -> Any:
        """Apply augmentation."""
        if random.random() > self.p:
            return data
        return self._apply(data)

    @abstractmethod
    def _apply(self, data: Any) -> Any:
        """Apply augmentation (internal)."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {"p": self.p}


class LayerShuffle(CADAugmentation):
    """
    Shuffle entity layer assignments.

    Simulates different layer naming conventions.
    """

    def __init__(
        self,
        shuffle_rate: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize layer shuffle.

        Args:
            shuffle_rate: Fraction of entities to shuffle
            p: Probability of applying
        """
        super().__init__(p)
        self.shuffle_rate = shuffle_rate

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        # Get all layers
        layers = list(set(e.get("layer", "0") for e in entities))
        if len(layers) < 2:
            return data

        # Shuffle some entities
        num_shuffle = max(1, int(len(entities) * self.shuffle_rate))
        indices = random.sample(range(len(entities)), min(num_shuffle, len(entities)))

        for idx in indices:
            current_layer = entities[idx].get("layer", "0")
            other_layers = [l for l in layers if l != current_layer]
            if other_layers:
                entities[idx]["layer"] = random.choice(other_layers)

        data["entities"] = entities
        return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "shuffle_rate": self.shuffle_rate}


class EntityDropout(CADAugmentation):
    """
    Randomly drop CAD entities.

    Simulates incomplete or simplified drawings.
    """

    def __init__(
        self,
        dropout_rate: float = 0.1,
        preserve_types: Optional[List[str]] = None,
        p: float = 0.5,
    ):
        """
        Initialize entity dropout.

        Args:
            dropout_rate: Fraction of entities to drop
            preserve_types: Entity types to never drop (e.g., ["LINE", "CIRCLE"])
            p: Probability of applying
        """
        super().__init__(p)
        self.dropout_rate = dropout_rate
        self.preserve_types = set(preserve_types) if preserve_types else set()

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        # Separate preserved and droppable entities
        preserved = []
        droppable = []

        for entity in entities:
            entity_type = entity.get("type", "").upper()
            if entity_type in self.preserve_types:
                preserved.append(entity)
            else:
                droppable.append(entity)

        # Drop some entities
        num_keep = max(1, int(len(droppable) * (1 - self.dropout_rate)))
        if num_keep < len(droppable):
            droppable = random.sample(droppable, num_keep)

        data["entities"] = preserved + droppable
        return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "dropout_rate": self.dropout_rate,
            "preserve_types": list(self.preserve_types),
        }


class TextPerturbation(CADAugmentation):
    """
    Perturb text content in CAD drawings.

    Simulates OCR errors or text variations.
    """

    def __init__(
        self,
        char_swap_rate: float = 0.05,
        case_change_rate: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize text perturbation.

        Args:
            char_swap_rate: Rate of character swaps
            case_change_rate: Rate of case changes
            p: Probability of applying
        """
        super().__init__(p)
        self.char_swap_rate = char_swap_rate
        self.case_change_rate = case_change_rate

        # Common OCR confusions
        self._confusions = {
            "0": "O", "O": "0",
            "1": "l", "l": "1",
            "I": "l", "l": "I",
            "5": "S", "S": "5",
            "8": "B", "B": "8",
            "2": "Z", "Z": "2",
        }

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        for entity in entities:
            if entity.get("type", "").upper() in ("TEXT", "MTEXT"):
                text = entity.get("text", "")
                if text:
                    entity["text"] = self._perturb_text(text)

        data["entities"] = entities
        return data

    def _perturb_text(self, text: str) -> str:
        """Perturb text content."""
        chars = list(text)

        for i, char in enumerate(chars):
            # Character swap
            if random.random() < self.char_swap_rate:
                if char in self._confusions:
                    chars[i] = self._confusions[char]

            # Case change
            if random.random() < self.case_change_rate:
                if char.isupper():
                    chars[i] = char.lower()
                elif char.islower():
                    chars[i] = char.upper()

        return "".join(chars)

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "char_swap_rate": self.char_swap_rate,
            "case_change_rate": self.case_change_rate,
        }


class DimensionNoise(CADAugmentation):
    """
    Add noise to dimension values.

    Simulates measurement variations.
    """

    def __init__(
        self,
        noise_percent: float = 0.02,
        p: float = 0.5,
    ):
        """
        Initialize dimension noise.

        Args:
            noise_percent: Maximum noise as percentage of value
            p: Probability of applying
        """
        super().__init__(p)
        self.noise_percent = noise_percent

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        for entity in entities:
            if entity.get("type", "").upper() == "DIMENSION":
                value = entity.get("measurement")
                if value is not None and isinstance(value, (int, float)):
                    noise = random.uniform(-self.noise_percent, self.noise_percent)
                    entity["measurement"] = value * (1 + noise)

            # Also perturb coordinate-based properties
            for key in ["length", "radius", "width", "height"]:
                if key in entity and isinstance(entity[key], (int, float)):
                    noise = random.uniform(-self.noise_percent, self.noise_percent)
                    entity[key] = entity[key] * (1 + noise)

        data["entities"] = entities
        return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "noise_percent": self.noise_percent}


class ColorJitter(CADAugmentation):
    """
    Jitter entity colors.

    Simulates different color schemes or display settings.
    """

    def __init__(
        self,
        color_change_rate: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize color jitter.

        Args:
            color_change_rate: Rate of color changes
            p: Probability of applying
        """
        super().__init__(p)
        self.color_change_rate = color_change_rate

        # Common CAD colors (AutoCAD color indices)
        self._colors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 256]  # Red, Yellow, Green, Cyan, Blue, Magenta, White, Gray, Light Gray, ByLayer

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        for entity in entities:
            if random.random() < self.color_change_rate:
                entity["color"] = random.choice(self._colors)

        data["entities"] = entities
        return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "color_change_rate": self.color_change_rate}


class LinetypeVariation(CADAugmentation):
    """
    Vary entity linetypes.

    Simulates different linetype configurations.
    """

    def __init__(
        self,
        variation_rate: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize linetype variation.

        Args:
            variation_rate: Rate of linetype changes
            p: Probability of applying
        """
        super().__init__(p)
        self.variation_rate = variation_rate

        # Common linetypes
        self._linetypes = [
            "Continuous", "DASHED", "HIDDEN", "CENTER",
            "PHANTOM", "DOT", "DASHDOT", "BORDER"
        ]

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        for entity in entities:
            if random.random() < self.variation_rate:
                entity["linetype"] = random.choice(self._linetypes)

        data["entities"] = entities
        return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "variation_rate": self.variation_rate}


class EntityTypeSimulation(CADAugmentation):
    """
    Simulate entity type variations.

    Converts between similar entity types (e.g., POLYLINE to LINE segments).
    """

    def __init__(
        self,
        conversion_rate: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize entity type simulation.

        Args:
            conversion_rate: Rate of type conversions
            p: Probability of applying
        """
        super().__init__(p)
        self.conversion_rate = conversion_rate

        # Convertible type pairs
        self._conversions = {
            "CIRCLE": "ARC",
            "ARC": "POLYLINE",
            "LWPOLYLINE": "POLYLINE",
        }

    def _apply(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        entities = data.get("entities", [])
        if not entities:
            return data

        for entity in entities:
            if random.random() < self.conversion_rate:
                entity_type = entity.get("type", "").upper()
                if entity_type in self._conversions:
                    entity["type"] = self._conversions[entity_type]
                    entity["converted_from"] = entity_type

        data["entities"] = entities
        return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "conversion_rate": self.conversion_rate}
