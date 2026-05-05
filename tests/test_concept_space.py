import pytest
import torch

from src.ldc.concept_space import ConceptGraph, ConceptSpaceConfig


def test_default_config_phase1_only_euclidean():
    cfg = ConceptSpaceConfig()
    assert cfg.d_hyperbolic == 0
    assert cfg.d_total == cfg.d_euclidean


def test_hyperbolic_disabled_in_phase1():
    with pytest.raises(NotImplementedError):
        ConceptSpaceConfig(d_hyperbolic=3)


def test_randn_concept_graph_shape():
    cfg = ConceptSpaceConfig(num_slots=4, d_euclidean=8)
    cg = ConceptGraph.randn(2, cfg)
    assert cg.shape == (2, 4, 8)
    assert cg.batch_size == 2


def test_zeros_and_randn_like():
    cfg = ConceptSpaceConfig(num_slots=4, d_euclidean=8)
    z = ConceptGraph.zeros(3, cfg)
    assert torch.equal(z.nodes, torch.zeros(3, 4, 8))
    r = ConceptGraph.randn_like(z)
    assert r.shape == z.shape


def test_validates_dims():
    with pytest.raises(ValueError):
        ConceptSpaceConfig(num_slots=0)
    with pytest.raises(ValueError):
        ConceptSpaceConfig(d_euclidean=0, d_hyperbolic=0)
