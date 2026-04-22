"""Base classes for storm data."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelVariant:
    """Base class representing a climate model and variant combination."""
    
    model: str
    variant: str
    scenarios: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"ModelVariant({self.model}, {self.variant}, scenarios={self.scenarios})"
    
    @property
    def model_variant_key(self) -> str:
        """Return a unique key for this model/variant combination."""
        return f"{self.model}_{self.variant}"