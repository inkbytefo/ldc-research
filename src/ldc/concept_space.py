"""Concept-graph latent representation.

A concept graph is a fixed set of K slots, each living in a product
manifold. Phase 1 uses pure Euclidean. Hyperbolic and toroidal components
are planned for Phase 5+ — the API here exposes hooks (the dataclass tracks
which slice of the latent dim is which manifold) so we don't have to break
contracts later.

Per `docs/yol-haritasi.md`:
    Phase 1: d_e=64 Euclidean, d_h=0
    Phase 5: d_e + 3-5 Poincare ball
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConceptSpaceConfig:
    num_slots: int = 8
    d_euclidean: int = 64
    d_hyperbolic: int = 0  # Phase 5+
    poincare_curvature: float = 1.0
    num_relations: int = 4

    @property
    def d_total(self) -> int:
        return self.d_euclidean + self.d_hyperbolic

    def __post_init__(self) -> None:
        if self.d_hyperbolic > 0:
            raise NotImplementedError(
                "Hyperbolic component is reserved for Phase 5. "
                "See docs/yol-haritasi.md section 7."
            )
        if self.d_total <= 0:
            raise ValueError("d_total must be > 0")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if self.num_relations <= 0:
            raise ValueError("num_relations must be > 0")


@dataclass
class ConceptGraph:
    """A batched concept graph.

    nodes : (B, K, d_total) — slot embeddings
    """

    nodes: torch.Tensor
    config: ConceptSpaceConfig = field(default_factory=ConceptSpaceConfig)

    @property
    def batch_size(self) -> int:
        return self.nodes.size(0)

    @property
    def shape(self) -> torch.Size:
        return self.nodes.shape

    def to(self, *args, **kwargs) -> "ConceptGraph":
        return ConceptGraph(nodes=self.nodes.to(*args, **kwargs), config=self.config)

    def detach(self) -> "ConceptGraph":
        return ConceptGraph(nodes=self.nodes.detach(), config=self.config)

    @classmethod
    def randn(
        cls,
        batch_size: int,
        config: ConceptSpaceConfig,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> "ConceptGraph":
        nodes = torch.randn(
            batch_size, config.num_slots, config.d_total, device=device, generator=generator
        )
        return cls(nodes=nodes, config=config)

    @classmethod
    def randn_like(cls, other: "ConceptGraph") -> "ConceptGraph":
        return cls(nodes=torch.randn_like(other.nodes), config=other.config)

    @classmethod
    def zeros(
        cls, batch_size: int, config: ConceptSpaceConfig, device: torch.device | str | None = None
    ) -> "ConceptGraph":
        nodes = torch.zeros(batch_size, config.num_slots, config.d_total, device=device)
        return cls(nodes=nodes, config=config)
