# HiCoRe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new compositional generalization architecture (HiCoRe) that combines hyperbolic-typed role embeddings, Tensor Product Representations, iterative refinement, and typed cross-attention decoder.

**Architecture:** Encoder produces fillers per token. A role-inference head assigns each token a discrete role (action/modifier/object/...) embedded in Poincaré ball. TPR binding sums outer products `Σ filler ⊗ role` into a structured tensor. An iterative refinement loop sharpens the bound tensor. A typed decoder generates output by querying roles to unbind fillers. Trained with `L_lm + β · L_role` (auxiliary role supervision from SCAN grammar).

**Tech Stack:** PyTorch 2.5+, geoopt (Poincaré ops), Hydra, pytest, existing `src/data` and `src/baselines/transformer.py` from LDC v2 codebase.

---

## File Structure

**New files (`src/hicore/`):**
- `__init__.py` — package marker
- `role_inventory.py` — Poincaré-ball role embeddings, projection, distance
- `tpr.py` — TPR bind / unbind operations
- `role_inference.py` — Gumbel-softmax classification head
- `filler_encoder.py` — Transformer encoder producing per-token fillers
- `refinement.py` — Iterative attention-based refinement of bound tensors
- `decoder.py` — Typed AR decoder with role-query unbinding cross-attention
- `model.py` — `HiCoReModel` + `HiCoReConfig` end-to-end

**New files (`tests/`):**
- `test_role_inventory.py`
- `test_tpr.py`
- `test_role_inference.py`
- `test_filler_encoder.py`
- `test_refinement.py`
- `test_typed_decoder.py`
- `test_hicore_model.py`

**New files (`src/data/`):**
- `role_tagger.py` — SCAN token → role label mapping (verb/modifier/direction/connector)

**New files (`experiments/configs/`):**
- `hicore_scan_simple.yaml`
- `hicore_scan_addjump.yaml`
- `hicore_scan_aroundright.yaml`

**Modified files:**
- `pyproject.toml` — add `geoopt` to `[project.dependencies]`
- `src/build.py` — add `hicore` branch to `build_model`

---

## Task 1: Project Setup

**Files:**
- Create: `src/hicore/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the package directory and marker**

```python
# src/hicore/__init__.py
```

(Empty file — just the package marker.)

- [ ] **Step 2: Add geoopt dependency**

Modify `pyproject.toml` `[project.dependencies]`. Add line: `"geoopt>=0.5.0",` to the existing dependencies array.

- [ ] **Step 3: Install dependency locally**

Run: `pip install -e .`
Expected: geoopt installed without errors. Verify with `python -c "import geoopt; print(geoopt.__version__)"`.

- [ ] **Step 4: Commit**

```bash
git add src/hicore/__init__.py pyproject.toml
git commit -m "feat(hicore): scaffold package + geoopt dependency"
```

---

## Task 2: Hyperbolic Role Inventory

A learnable inventory of K role embeddings on the Poincaré ball. Provides projection from Euclidean logits to Poincaré coordinates, distance computation, and role lookup.

**Files:**
- Create: `src/hicore/role_inventory.py`
- Test: `tests/test_role_inventory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_role_inventory.py
import pytest
import torch

from src.hicore.role_inventory import RoleInventory, RoleInventoryConfig


def test_role_inventory_shape():
    cfg = RoleInventoryConfig(num_roles=8, d_role=5)
    inv = RoleInventory(cfg)
    embeds = inv.role_embeddings()
    assert embeds.shape == (8, 5)


def test_role_inventory_inside_poincare_ball():
    cfg = RoleInventoryConfig(num_roles=8, d_role=5)
    inv = RoleInventory(cfg)
    embeds = inv.role_embeddings()
    norms = embeds.norm(dim=-1)
    assert torch.all(norms < 1.0 - 1e-3), f"Embeddings escape ball: max norm {norms.max()}"


def test_role_inventory_distance_symmetric():
    cfg = RoleInventoryConfig(num_roles=4, d_role=5)
    inv = RoleInventory(cfg)
    e = inv.role_embeddings()
    d_ab = inv.poincare_distance(e[0], e[1])
    d_ba = inv.poincare_distance(e[1], e[0])
    assert torch.allclose(d_ab, d_ba, atol=1e-5)


def test_role_inventory_gradient_flow():
    cfg = RoleInventoryConfig(num_roles=4, d_role=5)
    inv = RoleInventory(cfg)
    e = inv.role_embeddings()
    loss = e.norm(dim=-1).sum()
    loss.backward()
    # raw param has gradient
    assert inv._raw.grad is not None
    assert inv._raw.grad.abs().sum() > 0
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_role_inventory.py -v`
Expected: all 4 tests fail with `ModuleNotFoundError: src.hicore.role_inventory`.

- [ ] **Step 3: Implement RoleInventory**

```python
# src/hicore/role_inventory.py
"""Learnable role embeddings on the Poincaré ball.

Roles are stored as raw Euclidean parameters (`_raw`) and projected to
the open Poincaré ball via `tanh` scaling. This avoids dealing with
manifold optimizers in the Phase HC-1 prototype while preserving
hyperbolic geometry semantics for distance.

Phase HC-3: switch to `geoopt.ManifoldParameter` + Riemannian Adam if
the simple parameterization is unstable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RoleInventoryConfig:
    num_roles: int = 8
    d_role: int = 5
    init_scale: float = 0.1


class RoleInventory(nn.Module):
    def __init__(self, cfg: RoleInventoryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._raw = nn.Parameter(torch.randn(cfg.num_roles, cfg.d_role) * cfg.init_scale)

    def role_embeddings(self) -> torch.Tensor:
        """Project raw params to Poincaré ball via tanh on the norm."""
        x = self._raw
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.tanh(norm) / norm
        return x * scale * 0.999  # safety margin from boundary

    @staticmethod
    def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """d(u, v) = arccosh(1 + 2 ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))."""
        diff = (u - v).norm(dim=-1) ** 2
        u_norm = u.norm(dim=-1) ** 2
        v_norm = v.norm(dim=-1) ** 2
        denom = (1 - u_norm).clamp_min(1e-6) * (1 - v_norm).clamp_min(1e-6)
        arg = 1 + 2 * diff / denom
        return torch.acosh(arg.clamp_min(1.0 + 1e-7))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_role_inventory.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hicore/role_inventory.py tests/test_role_inventory.py
git commit -m "feat(hicore): hyperbolic role inventory with Poincare ball embeddings"
```

---

## Task 3: TPR Bind / Unbind

Outer product binding `T = Σ filler ⊗ role`. Unbinding via inner product against role.

**Files:**
- Create: `src/hicore/tpr.py`
- Test: `tests/test_tpr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tpr.py
import torch

from src.hicore.tpr import bind_tpr, unbind_tpr


def test_bind_shape():
    fillers = torch.randn(2, 4, 16)  # (B, N, d_filler)
    roles = torch.randn(2, 4, 8)     # (B, N, d_role)
    mask = torch.ones(2, 4, dtype=torch.bool)
    T = bind_tpr(fillers, roles, mask)
    assert T.shape == (2, 16, 8)


def test_bind_respects_mask():
    fillers = torch.randn(1, 3, 4)
    roles = torch.randn(1, 3, 4)
    full_mask = torch.tensor([[True, True, True]])
    half_mask = torch.tensor([[True, True, False]])
    T_full = bind_tpr(fillers, roles, full_mask)
    T_half = bind_tpr(fillers, roles, half_mask)
    expected_diff = torch.einsum("ni,nj->ij", fillers[0, 2:3], roles[0, 2:3])
    assert torch.allclose(T_full[0] - T_half[0], expected_diff, atol=1e-5)


def test_unbind_recovers_orthogonal_filler():
    # If roles are orthonormal, unbinding recovers the original filler.
    d_filler, d_role = 4, 3
    fillers = torch.randn(1, 3, d_filler)
    roles = torch.eye(d_role).unsqueeze(0)  # (1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    T = bind_tpr(fillers, roles, mask)
    f0_recovered = unbind_tpr(T, roles[:, 0])
    assert torch.allclose(f0_recovered[0], fillers[0, 0], atol=1e-5)


def test_unbind_batch():
    T = torch.randn(3, 8, 4)  # (B, d_filler, d_role)
    role_query = torch.randn(3, 4)
    out = unbind_tpr(T, role_query)
    assert out.shape == (3, 8)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_tpr.py -v`
Expected: 4 fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement TPR ops**

```python
# src/hicore/tpr.py
"""Tensor Product Representation: bind and unbind operations.

Binding sums per-token outer products of fillers and roles, masking out
padded positions. Unbinding takes the inner product of the bound tensor
with a query role vector to recover the corresponding filler.
"""
from __future__ import annotations

import torch


def bind_tpr(fillers: torch.Tensor, roles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """T[b] = Σ_i mask[b, i] * fillers[b, i] ⊗ roles[b, i].

    fillers : (B, N, d_filler)
    roles   : (B, N, d_role)
    mask    : (B, N) bool, True = valid token

    Returns : (B, d_filler, d_role)
    """
    m = mask.float().unsqueeze(-1)  # (B, N, 1)
    f = fillers * m
    # einsum over per-token outer products, sum over N
    return torch.einsum("bni,bnj->bij", f, roles)


def unbind_tpr(T: torch.Tensor, role_query: torch.Tensor) -> torch.Tensor:
    """Recover filler corresponding to a role.

    T           : (B, d_filler, d_role)
    role_query  : (B, d_role)

    Returns     : (B, d_filler) — T · role_query
    """
    return torch.einsum("bij,bj->bi", T, role_query)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_tpr.py -v`
Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/hicore/tpr.py tests/test_tpr.py
git commit -m "feat(hicore): TPR bind/unbind operations"
```

---

## Task 4: Role Inference Head

Maps each token's filler vector to a discrete role via Gumbel-softmax + Poincaré projection.

**Files:**
- Create: `src/hicore/role_inference.py`
- Test: `tests/test_role_inference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_role_inference.py
import torch

from src.hicore.role_inference import RoleInference, RoleInferenceConfig
from src.hicore.role_inventory import RoleInventory, RoleInventoryConfig


def test_role_inference_shapes():
    inv = RoleInventory(RoleInventoryConfig(num_roles=8, d_role=5))
    cfg = RoleInferenceConfig(d_filler=64, num_roles=8, d_role=5, hidden=32)
    head = RoleInference(cfg)
    fillers = torch.randn(2, 4, 64)
    role_emb, role_logits = head(fillers, inv, hard=False)
    assert role_emb.shape == (2, 4, 5)
    assert role_logits.shape == (2, 4, 8)


def test_role_inference_hard_one_hot():
    inv = RoleInventory(RoleInventoryConfig(num_roles=4, d_role=5))
    cfg = RoleInferenceConfig(d_filler=8, num_roles=4, d_role=5, hidden=16)
    head = RoleInference(cfg)
    fillers = torch.randn(1, 3, 8)
    _, logits = head(fillers, inv, hard=True)
    # Hard mode: argmax produces deterministic one-hot
    assert logits.shape == (1, 3, 4)


def test_role_inference_gradient_through_inventory():
    inv = RoleInventory(RoleInventoryConfig(num_roles=4, d_role=5))
    cfg = RoleInferenceConfig(d_filler=8, num_roles=4, d_role=5, hidden=16)
    head = RoleInference(cfg)
    fillers = torch.randn(1, 2, 8, requires_grad=True)
    role_emb, _ = head(fillers, inv, hard=False)
    loss = role_emb.norm(dim=-1).sum()
    loss.backward()
    assert inv._raw.grad is not None
    assert inv._raw.grad.abs().sum() > 0
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_role_inference.py -v`
Expected: 3 fails.

- [ ] **Step 3: Implement RoleInference**

```python
# src/hicore/role_inference.py
"""Per-token role classification with Gumbel-softmax.

Given filler vectors, predict a distribution over the K roles. In
training (`hard=False`) we use straight-through Gumbel-softmax so the
discrete selection is differentiable. At inference (`hard=True`) we
argmax for stable behavior.

Returns the role embedding (a Poincaré-ball vector) per token plus the
raw logits for auxiliary supervision losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .role_inventory import RoleInventory


@dataclass
class RoleInferenceConfig:
    d_filler: int = 64
    num_roles: int = 8
    d_role: int = 5
    hidden: int = 64
    tau: float = 0.5


class RoleInference(nn.Module):
    def __init__(self, cfg: RoleInferenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(cfg.d_filler, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.num_roles),
        )

    def forward(
        self, fillers: torch.Tensor, inventory: RoleInventory, hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fillers : (B, N, d_filler)
        Returns (role_emb (B, N, d_role), role_logits (B, N, num_roles)).
        """
        logits = self.head(fillers)  # (B, N, K)
        if hard:
            idx = logits.argmax(dim=-1)  # (B, N)
            one_hot = F.one_hot(idx, num_classes=self.cfg.num_roles).float()
        else:
            one_hot = F.gumbel_softmax(logits, tau=self.cfg.tau, hard=True, dim=-1)
        # (B, N, K) @ (K, d_role) -> (B, N, d_role)
        role_table = inventory.role_embeddings()
        role_emb = one_hot @ role_table
        return role_emb, logits
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_role_inference.py -v`
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/hicore/role_inference.py tests/test_role_inference.py
git commit -m "feat(hicore): role inference head with Gumbel-softmax"
```

---

## Task 5: Filler Encoder

Light transformer encoder mapping input tokens → fillers. No slot attention (HiCoRe binds via roles, not slots).

**Files:**
- Create: `src/hicore/filler_encoder.py`
- Test: `tests/test_filler_encoder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_filler_encoder.py
import torch

from src.hicore.filler_encoder import FillerEncoder, FillerEncoderConfig


def test_filler_encoder_shapes():
    cfg = FillerEncoderConfig(vocab_size=20, d_model=32, n_heads=4, n_layers=2,
                              ffn_mult=2, dropout=0.0, max_len=16, pad_id=0)
    enc = FillerEncoder(cfg)
    src = torch.randint(1, 20, (2, 10))
    src_mask = torch.ones(2, 10, dtype=torch.bool)
    fillers, ctx = enc(src, src_mask)
    assert fillers.shape == (2, 10, 32)
    assert ctx.shape == (2, 10, 32)


def test_filler_encoder_padding_zeroed():
    cfg = FillerEncoderConfig(vocab_size=20, d_model=16, n_heads=4, n_layers=1,
                              ffn_mult=2, dropout=0.0, max_len=16, pad_id=0)
    enc = FillerEncoder(cfg)
    src = torch.zeros(1, 5, dtype=torch.long)
    src[0, :3] = torch.tensor([1, 2, 3])
    src_mask = torch.tensor([[True, True, True, False, False]])
    fillers, _ = enc(src, src_mask)
    # Output for masked positions should not be NaN; valid positions non-zero
    assert not torch.isnan(fillers).any()
    assert fillers[0, 0].abs().sum() > 0
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_filler_encoder.py -v`
Expected: 2 fails.

- [ ] **Step 3: Implement FillerEncoder**

```python
# src/hicore/filler_encoder.py
"""Transformer encoder producing per-token filler vectors.

Lighter than `src/ldc/encoder.py` — there is no slot attention here.
HiCoRe relies on TPR binding (role × filler) for compositional
structure; the encoder only needs to produce decent token-level
embeddings.

Returns a single tensor used as both filler source and decoder
cross-attention context.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.modules import LearnedPositionalEmbedding


@dataclass
class FillerEncoderConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 64
    pad_id: int = 0


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, key_padding_mask=src_pad_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class FillerEncoder(nn.Module):
    def __init__(self, cfg: FillerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_len, d)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """src: (B, L)  src_mask: (B, L) bool, True=valid.
        Returns (fillers (B, L, d), context (B, L, d)).
        """
        x = self.embed(src)
        x = self.pos(x)
        src_pad_mask = ~src_mask
        for layer in self.layers:
            x = layer(x, src_pad_mask)
        h = self.final_norm(x)
        # In HiCoRe Phase HC-1 fillers and context are the same tensor.
        return h, h
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_filler_encoder.py -v`
Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/hicore/filler_encoder.py tests/test_filler_encoder.py
git commit -m "feat(hicore): filler encoder transformer"
```

---

## Task 6: Iterative Refinement

Refines the bound tensor `T` via attention against context. Fixed K iterations in Phase HC-1; adaptive halting deferred to Phase HC-4.

**Files:**
- Create: `src/hicore/refinement.py`
- Test: `tests/test_refinement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_refinement.py
import torch

from src.hicore.refinement import RefinementLoop, RefinementConfig


def test_refinement_preserves_shape():
    cfg = RefinementConfig(d_filler=16, d_role=5, d_model=32, n_heads=4,
                           ffn_mult=2, dropout=0.0, num_iter=3)
    ref = RefinementLoop(cfg)
    T = torch.randn(2, 16, 5)
    ctx = torch.randn(2, 8, 32)
    ctx_mask = torch.ones(2, 8, dtype=torch.bool)
    T_out = ref(T, ctx, ctx_mask)
    assert T_out.shape == T.shape


def test_refinement_zero_iter_identity():
    cfg = RefinementConfig(d_filler=16, d_role=5, d_model=32, n_heads=4,
                           ffn_mult=2, dropout=0.0, num_iter=0)
    ref = RefinementLoop(cfg)
    T = torch.randn(1, 16, 5)
    ctx = torch.randn(1, 4, 32)
    ctx_mask = torch.ones(1, 4, dtype=torch.bool)
    T_out = ref(T, ctx, ctx_mask)
    assert torch.allclose(T_out, T)


def test_refinement_gradient_flow():
    cfg = RefinementConfig(d_filler=16, d_role=5, d_model=32, n_heads=4,
                           ffn_mult=2, dropout=0.0, num_iter=2)
    ref = RefinementLoop(cfg)
    T = torch.randn(1, 16, 5, requires_grad=True)
    ctx = torch.randn(1, 4, 32, requires_grad=True)
    ctx_mask = torch.ones(1, 4, dtype=torch.bool)
    T_out = ref(T, ctx, ctx_mask)
    T_out.sum().backward()
    assert T.grad is not None
    assert ctx.grad is not None
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_refinement.py -v`
Expected: 3 fails.

- [ ] **Step 3: Implement RefinementLoop**

```python
# src/hicore/refinement.py
"""Iterative refinement of bound TPR tensors.

The bound tensor T = Σ filler ⊗ role lives in R^{d_filler × d_role}.
We flatten it and apply a small transformer that cross-attends over
encoder context. Each iteration produces a residual update.

Phase HC-1: fixed number of iterations.
Phase HC-4: replace with adaptive halting (ACT / Universal Transformer).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RefinementConfig:
    d_filler: int = 64
    d_role: int = 5
    d_model: int = 96
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    num_iter: int = 3


class _RefinementBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor, ctx_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        c, _ = self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_pad_mask, need_weights=False)
        x = x + self.drop(c)
        h = self.norm3(x)
        x = x + self.drop(self.ffn(h))
        return x


class RefinementLoop(nn.Module):
    """Refines T by treating each role as a token in a sequence.

    Internal representation: T (B, d_filler, d_role) is treated as
    a sequence of d_role tokens, each of dim d_filler. We project up
    to d_model for transformer ops, run num_iter blocks, then project
    back down.
    """

    def __init__(self, cfg: RefinementConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_filler, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_filler)
        self.blocks = nn.ModuleList(
            [
                _RefinementBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
                for _ in range(cfg.num_iter)
            ]
        )

    def forward(
        self, T: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor
    ) -> torch.Tensor:
        """T : (B, d_filler, d_role)
        ctx : (B, L, d_model_ctx) — assumed to match d_model here
        ctx_mask : (B, L) bool, True=valid

        Returns: refined T (B, d_filler, d_role).
        """
        if self.cfg.num_iter == 0:
            return T
        # (B, d_filler, d_role) -> (B, d_role, d_filler)
        x = T.transpose(1, 2)
        x = self.in_proj(x)  # (B, d_role, d_model)
        ctx_pad_mask = ~ctx_mask
        for block in self.blocks:
            x = block(x, ctx, ctx_pad_mask)
        x = self.out_proj(x)  # (B, d_role, d_filler)
        T_refined = x.transpose(1, 2)  # (B, d_filler, d_role)
        return T + T_refined
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_refinement.py -v`
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/hicore/refinement.py tests/test_refinement.py
git commit -m "feat(hicore): iterative refinement loop"
```

---

## Task 7: Typed Decoder

AR transformer decoder. Cross-attention is replaced by a role-query unbinding mechanism: at each step, a role distribution is predicted from the decoder hidden state and used to unbind a filler from the bound tensor.

**Files:**
- Create: `src/hicore/decoder.py`
- Test: `tests/test_typed_decoder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_typed_decoder.py
import torch

from src.hicore.decoder import TypedDecoder, TypedDecoderConfig
from src.hicore.role_inventory import RoleInventory, RoleInventoryConfig


def test_typed_decoder_shape():
    inv = RoleInventory(RoleInventoryConfig(num_roles=4, d_role=5))
    cfg = TypedDecoderConfig(vocab_size=20, d_model=32, d_filler=32, num_roles=4,
                             d_role=5, n_heads=4, n_layers=2, ffn_mult=2,
                             dropout=0.0, max_len=16, pad_id=0)
    dec = TypedDecoder(cfg)
    tgt = torch.randint(1, 20, (2, 6))
    tgt_mask = torch.ones(2, 6, dtype=torch.bool)
    T = torch.randn(2, 32, 5)
    logits = dec(tgt, tgt_mask, T, inv)
    assert logits.shape == (2, 6, 20)


def test_typed_decoder_generate():
    inv = RoleInventory(RoleInventoryConfig(num_roles=4, d_role=5))
    cfg = TypedDecoderConfig(vocab_size=20, d_model=32, d_filler=32, num_roles=4,
                             d_role=5, n_heads=4, n_layers=2, ffn_mult=2,
                             dropout=0.0, max_len=16, pad_id=0)
    dec = TypedDecoder(cfg)
    T = torch.randn(2, 32, 5)
    out = dec.generate(T, inv, sos_id=1, eos_id=2, max_len=8)
    assert out.shape[0] == 2
    assert out.shape[1] <= 8
    assert (out[:, 0] == 1).all()  # all start with sos
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_typed_decoder.py -v`
Expected: 2 fails.

- [ ] **Step 3: Implement TypedDecoder**

```python
# src/hicore/decoder.py
"""Autoregressive decoder with role-query cross-attention.

At each generation step:
  1. Self-attention over the partial output sequence.
  2. Predict a role-query vector from the hidden state.
  3. Use the role-query to unbind a filler from the bound tensor T.
  4. Mix the unbound filler back into the hidden state, then predict
     the next token.

This replaces the standard cross-attention to encoder context. The
typed flow is the architectural commitment to compositional binding.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.modules import LearnedPositionalEmbedding, causal_mask

from .role_inventory import RoleInventory
from .tpr import unbind_tpr


@dataclass
class TypedDecoderConfig:
    vocab_size: int
    d_model: int = 96
    d_filler: int = 64
    num_roles: int = 8
    d_role: int = 5
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 64
    pad_id: int = 0


class _DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_filler: int, num_roles: int, d_role: int,
        n_heads: int, ffn_mult: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.role_query = nn.Linear(d_model, num_roles)
        self.unbind_proj = nn.Linear(d_filler, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.tau = 1.0

    def forward(
        self, x: torch.Tensor, tgt_pad_mask: torch.Tensor, attn_mask: torch.Tensor,
        T: torch.Tensor, inventory: RoleInventory,
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=attn_mask, key_padding_mask=tgt_pad_mask,
                              need_weights=False)
        x = x + self.drop(a)

        # Role-query unbinding: per-position role distribution -> Poincaré role -> unbind.
        h = self.norm2(x)
        role_logits = self.role_query(h)  # (B, T, num_roles)
        role_probs = torch.softmax(role_logits / self.tau, dim=-1)
        role_table = inventory.role_embeddings()  # (num_roles, d_role)
        role_vecs = role_probs @ role_table  # (B, T, d_role)
        # Unbind per-position: T (B, d_filler, d_role) · role_vecs (B, T, d_role)
        unbound = torch.einsum("bij,btj->bti", T, role_vecs)  # (B, T, d_filler)
        unbound_proj = self.unbind_proj(unbound)
        x = x + self.drop(unbound_proj)

        h = self.norm3(x)
        x = x + self.drop(self.ffn(h))
        return x


class TypedDecoder(nn.Module):
    def __init__(self, cfg: TypedDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_len, d)
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(
                    d_model=d, d_filler=cfg.d_filler, num_roles=cfg.num_roles,
                    d_role=cfg.d_role, n_heads=cfg.n_heads, ffn_mult=cfg.ffn_mult,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size)

    def forward(
        self, tgt: torch.Tensor, tgt_mask: torch.Tensor,
        T: torch.Tensor, inventory: RoleInventory,
    ) -> torch.Tensor:
        """tgt : (B, T_len) input ids
        tgt_mask : (B, T_len) bool
        T : (B, d_filler, d_role)
        Returns logits (B, T_len, vocab).
        """
        B, T_len = tgt.shape
        x = self.embed(tgt)
        x = self.pos(x)
        tgt_pad_mask = ~tgt_mask
        attn_mask = causal_mask(T_len, device=x.device)
        for layer in self.layers:
            x = layer(x, tgt_pad_mask, attn_mask, T, inventory)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, T: torch.Tensor, inventory: RoleInventory,
        sos_id: int, eos_id: int, max_len: int,
    ) -> torch.Tensor:
        B = T.size(0)
        device = T.device
        effective_max = min(max_len, self.cfg.max_len)
        out = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(effective_max - 1):
            tgt_mask = (out != self.cfg.pad_id)
            logits = self.forward(out, tgt_mask, T, inventory)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            next_tok = torch.where(finished.unsqueeze(-1), torch.full_like(next_tok, self.cfg.pad_id), next_tok)
            out = torch.cat([out, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == eos_id)
            if finished.all():
                break
        return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_typed_decoder.py -v`
Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/hicore/decoder.py tests/test_typed_decoder.py
git commit -m "feat(hicore): typed decoder with role-query unbinding"
```

---

## Task 8: End-to-End HiCoReModel

Composes all submodules into a single `nn.Module`. Forward returns `dict` matching the `LDCModel` interface so `train.py` works without modification.

**Files:**
- Create: `src/hicore/model.py`
- Test: `tests/test_hicore_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hicore_model.py
import torch
import torch.nn.functional as F

from src.hicore.decoder import TypedDecoderConfig
from src.hicore.filler_encoder import FillerEncoderConfig
from src.hicore.model import HiCoReConfig, HiCoReModel
from src.hicore.refinement import RefinementConfig
from src.hicore.role_inference import RoleInferenceConfig
from src.hicore.role_inventory import RoleInventoryConfig


def _tiny_cfg(vocab_src: int, vocab_tgt: int) -> HiCoReConfig:
    return HiCoReConfig(
        encoder=FillerEncoderConfig(vocab_size=vocab_src, d_model=32, n_heads=4,
                                    n_layers=1, ffn_mult=2, dropout=0.0,
                                    max_len=16, pad_id=0),
        decoder=TypedDecoderConfig(vocab_size=vocab_tgt, d_model=32, d_filler=32,
                                   num_roles=4, d_role=5, n_heads=4, n_layers=1,
                                   ffn_mult=2, dropout=0.0, max_len=16, pad_id=0),
        inventory=RoleInventoryConfig(num_roles=4, d_role=5),
        role_inference=RoleInferenceConfig(d_filler=32, num_roles=4, d_role=5, hidden=32),
        refinement=RefinementConfig(d_filler=32, d_role=5, d_model=32, n_heads=4,
                                    ffn_mult=2, dropout=0.0, num_iter=2),
        alpha_lm=1.0, beta_role=0.5,
    )


def test_hicore_forward_shapes():
    cfg = _tiny_cfg(20, 20)
    model = HiCoReModel(cfg)
    src = torch.randint(1, 20, (2, 8))
    src_mask = torch.ones(2, 8, dtype=torch.bool)
    tgt = torch.randint(1, 20, (2, 6))
    tgt_mask = torch.ones(2, 6, dtype=torch.bool)
    out = model(src, src_mask, tgt, tgt_mask)
    assert "loss" in out
    assert "logits" in out
    assert out["logits"].shape == (2, 5, 20)  # tgt[:, :-1]


def test_hicore_backward():
    cfg = _tiny_cfg(20, 20)
    model = HiCoReModel(cfg)
    src = torch.randint(1, 20, (2, 8))
    src_mask = torch.ones(2, 8, dtype=torch.bool)
    tgt = torch.randint(1, 20, (2, 6))
    tgt_mask = torch.ones(2, 6, dtype=torch.bool)
    out = model(src, src_mask, tgt, tgt_mask)
    out["loss"].backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0


def test_hicore_generate():
    cfg = _tiny_cfg(20, 20)
    model = HiCoReModel(cfg)
    src = torch.randint(1, 20, (2, 8))
    src_mask = torch.ones(2, 8, dtype=torch.bool)
    out = model.generate(src, src_mask, sos_id=1, eos_id=2, max_len=10)
    assert out.shape[0] == 2
    assert out.shape[1] <= 10


def test_hicore_role_loss_with_labels():
    cfg = _tiny_cfg(20, 20)
    model = HiCoReModel(cfg)
    src = torch.randint(1, 20, (2, 4))
    src_mask = torch.ones(2, 4, dtype=torch.bool)
    tgt = torch.randint(1, 20, (2, 4))
    tgt_mask = torch.ones(2, 4, dtype=torch.bool)
    role_labels = torch.zeros(2, 4, dtype=torch.long)  # all role 0
    out = model(src, src_mask, tgt, tgt_mask, role_labels=role_labels)
    assert "l_role" in out
    assert out["l_role"].item() >= 0
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_hicore_model.py -v`
Expected: 4 fails.

- [ ] **Step 3: Implement HiCoReModel**

```python
# src/hicore/model.py
"""End-to-end HiCoRe model.

Loss = alpha_lm * L_lm + beta_role * L_role  (when role_labels provided)

Forward returns a dict with the same keys as LDCModel so the existing
training loop in src/train.py works with minimal modification.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import TypedDecoder, TypedDecoderConfig
from .filler_encoder import FillerEncoder, FillerEncoderConfig
from .refinement import RefinementConfig, RefinementLoop
from .role_inference import RoleInference, RoleInferenceConfig
from .role_inventory import RoleInventory, RoleInventoryConfig
from .tpr import bind_tpr


@dataclass
class HiCoReConfig:
    encoder: FillerEncoderConfig
    decoder: TypedDecoderConfig
    inventory: RoleInventoryConfig
    role_inference: RoleInferenceConfig
    refinement: RefinementConfig
    alpha_lm: float = 1.0
    beta_role: float = 0.5


class HiCoReModel(nn.Module):
    def __init__(self, cfg: HiCoReConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Cross-config sanity checks.
        assert cfg.role_inference.d_filler == cfg.encoder.d_model
        assert cfg.role_inference.num_roles == cfg.inventory.num_roles
        assert cfg.role_inference.d_role == cfg.inventory.d_role
        assert cfg.refinement.d_filler == cfg.encoder.d_model
        assert cfg.refinement.d_role == cfg.inventory.d_role
        assert cfg.decoder.d_filler == cfg.encoder.d_model
        assert cfg.decoder.num_roles == cfg.inventory.num_roles
        assert cfg.decoder.d_role == cfg.inventory.d_role

        self.encoder = FillerEncoder(cfg.encoder)
        self.inventory = RoleInventory(cfg.inventory)
        self.role_head = RoleInference(cfg.role_inference)
        self.refinement = RefinementLoop(cfg.refinement)
        self.decoder = TypedDecoder(cfg.decoder)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        role_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fillers, ctx = self.encoder(src, src_mask)
        role_emb, role_logits = self.role_head(fillers, self.inventory, hard=False)
        T = bind_tpr(fillers, role_emb, src_mask)
        T_refined = self.refinement(T, ctx, src_mask)

        tgt_in = tgt[:, :-1]
        tgt_in_mask = tgt_mask[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_out_mask = tgt_mask[:, 1:]
        logits = self.decoder(tgt_in, tgt_in_mask, T_refined, self.inventory)

        l_lm = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.cfg.decoder.pad_id,
            reduction="mean",
        )

        l_role = torch.tensor(0.0, device=l_lm.device)
        if role_labels is not None:
            valid = src_mask
            l_role = F.cross_entropy(
                role_logits.reshape(-1, role_logits.size(-1)),
                role_labels.reshape(-1),
                reduction="none",
            )
            l_role = (l_role * valid.reshape(-1).float()).sum() / valid.sum().clamp_min(1).float()

        loss = self.cfg.alpha_lm * l_lm + self.cfg.beta_role * l_role

        with torch.no_grad():
            preds = logits.argmax(-1)
            valid_out = tgt_out_mask
            correct = ((preds == tgt_out) & valid_out).sum().float()
            n = valid_out.sum().clamp_min(1).float()
            tok_acc = correct / n

        return {
            "loss": loss,
            "l_lm": l_lm.detach(),
            "l_role": l_role.detach(),
            "logits": logits,
            "token_acc": tok_acc,
        }

    @torch.no_grad()
    def generate(
        self, src: torch.Tensor, src_mask: torch.Tensor,
        sos_id: int, eos_id: int, max_len: int,
    ) -> torch.Tensor:
        fillers, ctx = self.encoder(src, src_mask)
        role_emb, _ = self.role_head(fillers, self.inventory, hard=True)
        T = bind_tpr(fillers, role_emb, src_mask)
        T_refined = self.refinement(T, ctx, src_mask)
        return self.decoder.generate(T_refined, self.inventory, sos_id, eos_id, max_len)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_hicore_model.py -v`
Expected: 4 pass.

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: previously passing tests still pass; new HiCoRe tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hicore/model.py tests/test_hicore_model.py
git commit -m "feat(hicore): end-to-end HiCoReModel"
```

---

## Task 9: SCAN Role Tagger

Maps SCAN tokens to role labels. Used for auxiliary `L_role` supervision.

**Files:**
- Create: `src/data/role_tagger.py`
- Test: extend `tests/test_scan_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scan_loader.py`:

```python
from src.data.role_tagger import tag_roles, ROLE_VOCAB


def test_role_tagger_basic():
    tokens = ["walk", "twice", "and", "jump", "around", "left"]
    role_ids = tag_roles(tokens)
    assert len(role_ids) == 6
    # All ids in valid range
    for r in role_ids:
        assert 0 <= r < len(ROLE_VOCAB)


def test_role_tagger_known_roles():
    tokens = ["walk", "twice", "left"]
    role_ids = tag_roles(tokens)
    name = lambda i: ROLE_VOCAB[i]
    assert name(role_ids[0]) == "action"
    assert name(role_ids[1]) == "modifier"
    assert name(role_ids[2]) == "direction"


def test_role_tagger_unknown_token():
    tokens = ["foobar"]
    role_ids = tag_roles(tokens)
    assert ROLE_VOCAB[role_ids[0]] == "unknown"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_scan_loader.py -v -k role_tagger`
Expected: 3 fails.

- [ ] **Step 3: Implement role_tagger**

```python
# src/data/role_tagger.py
"""Map SCAN source tokens to role labels for auxiliary L_role supervision.

The SCAN grammar is small, so we hand-build a lookup. For Phase HC-1
this is sufficient. Phase HC-3 explores unsupervised role induction.
"""
from __future__ import annotations

ROLE_VOCAB = ["unknown", "action", "modifier", "direction", "connector", "padding"]

_ACTIONS = {"walk", "run", "jump", "look", "turn"}
_MODIFIERS = {"twice", "thrice", "around", "opposite"}
_DIRECTIONS = {"left", "right"}
_CONNECTORS = {"and", "after"}


def _role_for(tok: str) -> str:
    if tok in _ACTIONS:
        return "action"
    if tok in _MODIFIERS:
        return "modifier"
    if tok in _DIRECTIONS:
        return "direction"
    if tok in _CONNECTORS:
        return "connector"
    return "unknown"


def tag_roles(tokens: list[str]) -> list[int]:
    """Returns the role id (index into ROLE_VOCAB) for each token."""
    return [ROLE_VOCAB.index(_role_for(t)) for t in tokens]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_scan_loader.py -v -k role_tagger`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/data/role_tagger.py tests/test_scan_loader.py
git commit -m "feat(data): SCAN role tagger for auxiliary supervision"
```

---

## Task 10: Loader Integration for Role Labels

Adapt `SCANDataset` and `collate_seq2seq` to also produce per-token role labels matching the source token sequence.

**Files:**
- Modify: `src/data/scan.py`
- Test: extend `tests/test_scan_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scan_loader.py`:

```python
def test_scan_dataset_includes_roles():
    from src.data.scan import SCANDataset
    from src.data.tokenizer import WordTokenizer
    pairs = [("walk twice", "I_WALK I_WALK"), ("jump", "I_JUMP")]
    src_tok = WordTokenizer.from_corpus(s for s, _ in pairs)
    tgt_tok = WordTokenizer.from_corpus(t for _, t in pairs)
    ds = SCANDataset.from_pairs(pairs, src_tok, tgt_tok, include_roles=True)
    item = ds[0]
    assert "src_roles" in item
    assert item["src_roles"].shape == item["src_ids"].shape


def test_collate_passes_role_labels():
    from src.data.scan import SCANDataset, collate_seq2seq
    from src.data.tokenizer import WordTokenizer
    pairs = [("walk twice", "I_WALK I_WALK"), ("jump", "I_JUMP")]
    src_tok = WordTokenizer.from_corpus(s for s, _ in pairs)
    tgt_tok = WordTokenizer.from_corpus(t for _, t in pairs)
    ds = SCANDataset.from_pairs(pairs, src_tok, tgt_tok, include_roles=True)
    items = [ds[0], ds[1]]
    batch = collate_seq2seq(items, src_tok.pad_id, tgt_tok.pad_id)
    assert "src_roles" in batch
    assert batch["src_roles"].shape == batch["src"].shape
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_scan_loader.py -v -k "scan_dataset_includes_roles or collate_passes_role"`
Expected: 2 fails.

- [ ] **Step 3: Modify SCANDataset and collate_seq2seq**

Modify `src/data/scan.py`. Update the `SCANDataset` class:

```python
from src.data.role_tagger import tag_roles, ROLE_VOCAB


class SCANDataset(Dataset):
    def __init__(
        self,
        src_ids: list[list[int]],
        tgt_ids: list[list[int]],
        src_roles: list[list[int]] | None = None,
    ) -> None:
        assert len(src_ids) == len(tgt_ids)
        self._src = src_ids
        self._tgt = tgt_ids
        self._roles = src_roles

    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[str, str]],
        src_tok: WordTokenizer,
        tgt_tok: WordTokenizer,
        include_roles: bool = False,
    ) -> "SCANDataset":
        src_ids = [src_tok.encode(s, add_sos=True, add_eos=True) for s, _ in pairs]
        tgt_ids = [tgt_tok.encode(t, add_sos=True, add_eos=True) for _, t in pairs]
        roles = None
        if include_roles:
            roles = []
            pad_role = ROLE_VOCAB.index("padding")
            for (s, _), encoded in zip(pairs, src_ids):
                # encoded includes <sos> and <eos>; align role labels.
                tokens = s.split()
                inner_roles = tag_roles(tokens)
                # encoded layout: [sos, ...tokens..., eos]
                aligned = [pad_role] + inner_roles + [pad_role]
                # If tokenizer dropped some words (unlikely with WordTokenizer), pad to length.
                if len(aligned) < len(encoded):
                    aligned = aligned + [pad_role] * (len(encoded) - len(aligned))
                roles.append(aligned[: len(encoded)])
        return cls(src_ids, tgt_ids, roles)

    def __len__(self) -> int:
        return len(self._src)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = {
            "src_ids": torch.tensor(self._src[idx], dtype=torch.long),
            "tgt_ids": torch.tensor(self._tgt[idx], dtype=torch.long),
        }
        if self._roles is not None:
            item["src_roles"] = torch.tensor(self._roles[idx], dtype=torch.long)
        return item
```

Update `collate_seq2seq` to handle the optional `src_roles` field:

```python
def collate_seq2seq(
    items: list[dict[str, Tensor]],
    src_pad_id: int,
    tgt_pad_id: int,
) -> dict[str, Any]:
    src_seqs = [it["src_ids"] for it in items]
    tgt_seqs = [it["tgt_ids"] for it in items]

    src = _pad_sequences(src_seqs, src_pad_id)
    tgt = _pad_sequences(tgt_seqs, tgt_pad_id)
    src_mask = src != src_pad_id
    tgt_mask = tgt != tgt_pad_id

    out = {"src": src, "tgt": tgt, "src_mask": src_mask, "tgt_mask": tgt_mask}

    if "src_roles" in items[0]:
        pad_role = ROLE_VOCAB.index("padding")
        role_seqs = [it["src_roles"] for it in items]
        out["src_roles"] = _pad_sequences(role_seqs, pad_role)
    return out
```

Update `build_scan_loaders` signature: add `include_roles: bool = False` parameter and pass it to `SCANDataset.from_pairs`.

```python
def build_scan_loaders(
    split: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    max_src_len: int,
    max_tgt_len: int,
    val_fraction: float,
    seed: int,
    include_roles: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, WordTokenizer, WordTokenizer]:
    # existing body...
    def _make_loader(pairs, shuffle):
        ds = SCANDataset.from_pairs(pairs, src_tok, tgt_tok, include_roles=include_roles)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda b: collate_seq2seq(b, src_tok.pad_id, tgt_tok.pad_id),
        )
    # rest unchanged
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_scan_loader.py -v`
Expected: all SCAN loader tests pass, including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add src/data/scan.py tests/test_scan_loader.py
git commit -m "feat(data): include per-token role labels in SCAN loader"
```

---

## Task 11: Build Pipeline Integration

Modify `src/build.py` to construct `HiCoReModel` from a Hydra config when `cfg.model.type == "hicore"`.

**Files:**
- Modify: `src/build.py`

- [ ] **Step 1: Add hicore branch to build_model**

Modify `src/build.py`. After the `if mtype == "ldc_v2":` block and before `raise ValueError(...)`, add:

```python
    if mtype == "hicore":
        from src.hicore.decoder import TypedDecoderConfig
        from src.hicore.filler_encoder import FillerEncoderConfig
        from src.hicore.model import HiCoReConfig, HiCoReModel
        from src.hicore.refinement import RefinementConfig
        from src.hicore.role_inference import RoleInferenceConfig
        from src.hicore.role_inventory import RoleInventoryConfig

        d_model = cfg.model.d_model
        d_role = cfg.model.d_role
        num_roles = cfg.model.num_roles

        encoder_cfg = FillerEncoderConfig(
            vocab_size=src_tok.vocab_size,
            d_model=d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.enc_layers,
            ffn_mult=cfg.model.ffn_mult,
            dropout=cfg.model.dropout,
            max_len=cfg.data.max_src_len,
            pad_id=src_tok.pad_id,
        )
        decoder_cfg = TypedDecoderConfig(
            vocab_size=tgt_tok.vocab_size,
            d_model=d_model,
            d_filler=d_model,
            num_roles=num_roles,
            d_role=d_role,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.dec_layers,
            ffn_mult=cfg.model.ffn_mult,
            dropout=cfg.model.dropout,
            max_len=cfg.data.max_tgt_len,
            pad_id=tgt_tok.pad_id,
        )
        inventory_cfg = RoleInventoryConfig(num_roles=num_roles, d_role=d_role)
        role_cfg = RoleInferenceConfig(
            d_filler=d_model,
            num_roles=num_roles,
            d_role=d_role,
            hidden=cfg.model.role_hidden,
            tau=cfg.model.role_tau,
        )
        refine_cfg = RefinementConfig(
            d_filler=d_model,
            d_role=d_role,
            d_model=d_model,
            n_heads=cfg.model.n_heads,
            ffn_mult=cfg.model.ffn_mult,
            dropout=cfg.model.dropout,
            num_iter=cfg.model.refine_iter,
        )
        return HiCoReModel(
            HiCoReConfig(
                encoder=encoder_cfg,
                decoder=decoder_cfg,
                inventory=inventory_cfg,
                role_inference=role_cfg,
                refinement=refine_cfg,
                alpha_lm=cfg.model.alpha_lm,
                beta_role=cfg.model.beta_role,
            )
        )
```

Also modify `build_loaders`: pass `include_roles=cfg.data.get("include_roles", False)` to `build_scan_loaders`. Replace the SCAN call with:

```python
        train, val, test, src_tok, tgt_tok = build_scan_loaders(
            split=cfg.data.scan_split,
            data_root=cfg.data.data_root,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            max_src_len=cfg.data.max_src_len,
            max_tgt_len=cfg.data.max_tgt_len,
            val_fraction=cfg.data.val_fraction,
            seed=cfg.seed,
            include_roles=cfg.data.get("include_roles", False),
        )
```

- [ ] **Step 2: Quick sanity check**

Run: `python -c "from src.build import build_model; print('ok')"`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/build.py
git commit -m "feat(build): wire HiCoReModel into build_model factory"
```

---

## Task 12: Train Loop Role Loss Pass-through

Modify `src/train.py` so the role labels are forwarded to the model when present in the batch.

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Update training step to pass role labels**

In `src/train.py`, locate the training-step forward call:

```python
        out = model(batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"])
```

Replace with:

```python
        forward_kwargs = {}
        if "src_roles" in batch:
            forward_kwargs["role_labels"] = batch["src_roles"]
        out = model(batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"], **forward_kwargs)
```

Also update the log message block — extend to log `l_role` when present:

```python
            if "l_role" in out:
                log_msg["l_role"] = float(out["l_role"].item())
```

(Place this next to the existing `if "l_diff" in out:` block.)

- [ ] **Step 2: Update eval to skip role labels (eval doesn't need them)**

In `src/eval.py`, the existing `out = model(batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"])` call works because `role_labels` defaults to `None`. No change needed; but ensure nothing breaks if `src_roles` is present in batch. If `evaluate()` does anything that breaks, drop the key:

```python
        batch = {k: v.to(device) for k, v in batch.items() if k != "src_roles" or hasattr(model, "role_head")}
```

For simplicity, leave the existing code alone — `model(...)` with positional args ignores extra batch keys.

- [ ] **Step 3: Smoke test the training loop on synthetic data**

Run: `pytest tests/test_train_smoke.py -v`
Expected: existing baseline + LDC v2 smoke tests still pass.

- [ ] **Step 4: Commit**

```bash
git add src/train.py
git commit -m "feat(train): forward role labels to HiCoReModel"
```

---

## Task 13: Hydra Configs for HiCoRe

Three config files: simple, addjump, aroundright. They mirror the LDC v2 layout.

**Files:**
- Create: `experiments/configs/hicore_scan_simple.yaml`
- Create: `experiments/configs/hicore_scan_addjump.yaml`
- Create: `experiments/configs/hicore_scan_aroundright.yaml`

- [ ] **Step 1: Write hicore_scan_simple.yaml**

```yaml
defaults:
  - base
  - _self_

experiment_name: hicore_scan_simple

model:
  type: hicore
  d_model: 96
  n_heads: 4
  enc_layers: 2
  dec_layers: 2
  ffn_mult: 4
  dropout: 0.1
  num_roles: 8
  d_role: 5
  refine_iter: 3
  role_hidden: 64
  role_tau: 0.5
  alpha_lm: 1.0
  beta_role: 0.5

trainer:
  max_steps: 20000
  lr: 3.0e-4
  warmup_steps: 1000

data:
  scan_split: simple
  batch_size: 64
  include_roles: true

wandb:
  tags:
    - hicore
    - scan
    - simple
```

- [ ] **Step 2: Write hicore_scan_addjump.yaml**

Same content as `hicore_scan_simple.yaml`, but change:
- `experiment_name: hicore_scan_addjump`
- `data.scan_split: addprim_jump`
- `wandb.tags: [hicore, scan, addjump]` (block style)

```yaml
defaults:
  - base
  - _self_

experiment_name: hicore_scan_addjump

model:
  type: hicore
  d_model: 96
  n_heads: 4
  enc_layers: 2
  dec_layers: 2
  ffn_mult: 4
  dropout: 0.1
  num_roles: 8
  d_role: 5
  refine_iter: 3
  role_hidden: 64
  role_tau: 0.5
  alpha_lm: 1.0
  beta_role: 0.5

trainer:
  max_steps: 20000
  lr: 3.0e-4
  warmup_steps: 1000

data:
  scan_split: addprim_jump
  batch_size: 64
  include_roles: true

wandb:
  tags:
    - hicore
    - scan
    - addjump
```

- [ ] **Step 3: Write hicore_scan_aroundright.yaml**

```yaml
defaults:
  - base
  - _self_

experiment_name: hicore_scan_aroundright

model:
  type: hicore
  d_model: 96
  n_heads: 4
  enc_layers: 2
  dec_layers: 2
  ffn_mult: 4
  dropout: 0.1
  num_roles: 8
  d_role: 5
  refine_iter: 3
  role_hidden: 64
  role_tau: 0.5
  alpha_lm: 1.0
  beta_role: 0.5

trainer:
  max_steps: 20000
  lr: 3.0e-4
  warmup_steps: 1000

data:
  scan_split: template_around_right
  batch_size: 64
  include_roles: true

wandb:
  tags:
    - hicore
    - scan
    - aroundright
```

- [ ] **Step 4: Hydra dry-run sanity check**

Run: `python -c "from omegaconf import OmegaConf; from src.build import build_model, build_loaders; import hydra; from hydra import compose, initialize_config_dir; from pathlib import Path; cfg_dir = str(Path('experiments/configs').resolve()); initialize_config_dir(config_dir=cfg_dir, version_base=None); cfg = compose(config_name='hicore_scan_simple'); print('cfg loaded:', cfg.experiment_name)"`
Expected: prints `cfg loaded: hicore_scan_simple`.

- [ ] **Step 5: Commit**

```bash
git add experiments/configs/hicore_scan_simple.yaml experiments/configs/hicore_scan_addjump.yaml experiments/configs/hicore_scan_aroundright.yaml
git commit -m "feat(configs): HiCoRe SCAN configs (simple, addjump, aroundright)"
```

---

## Task 14: End-to-End Smoke Test

Verify the model trains for a few steps on synthetic data and produces non-NaN losses.

**Files:**
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Add smoke test for HiCoRe**

Append to `tests/test_train_smoke.py`:

```python
def test_hicore_smoke(tmp_path):
    """Train HiCoRe for a handful of steps on synthetic SCAN-like data.

    This catches integration bugs across encoder/inference/binding/
    refinement/decoder without requiring the real SCAN dataset.
    """
    import torch
    from src.data.tokenizer import WordTokenizer
    from src.hicore.decoder import TypedDecoderConfig
    from src.hicore.filler_encoder import FillerEncoderConfig
    from src.hicore.model import HiCoReConfig, HiCoReModel
    from src.hicore.refinement import RefinementConfig
    from src.hicore.role_inference import RoleInferenceConfig
    from src.hicore.role_inventory import RoleInventoryConfig

    tokenizer = WordTokenizer.from_corpus(
        ["walk twice", "jump", "look around left", "run thrice"]
    )
    cfg = HiCoReConfig(
        encoder=FillerEncoderConfig(vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4,
                                    n_layers=1, ffn_mult=2, dropout=0.0, max_len=16,
                                    pad_id=tokenizer.pad_id),
        decoder=TypedDecoderConfig(vocab_size=tokenizer.vocab_size, d_model=32, d_filler=32,
                                   num_roles=4, d_role=5, n_heads=4, n_layers=1, ffn_mult=2,
                                   dropout=0.0, max_len=16, pad_id=tokenizer.pad_id),
        inventory=RoleInventoryConfig(num_roles=4, d_role=5),
        role_inference=RoleInferenceConfig(d_filler=32, num_roles=4, d_role=5, hidden=32),
        refinement=RefinementConfig(d_filler=32, d_role=5, d_model=32, n_heads=4,
                                    ffn_mult=2, dropout=0.0, num_iter=2),
        alpha_lm=1.0, beta_role=0.5,
    )
    model = HiCoReModel(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    src = torch.randint(1, tokenizer.vocab_size, (4, 6))
    src_mask = torch.ones(4, 6, dtype=torch.bool)
    tgt = torch.randint(1, tokenizer.vocab_size, (4, 6))
    tgt_mask = torch.ones(4, 6, dtype=torch.bool)
    role_labels = torch.zeros(4, 6, dtype=torch.long)

    losses = []
    for _ in range(10):
        out = model(src, src_mask, tgt, tgt_mask, role_labels=role_labels)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        assert not torch.isnan(loss), "loss became NaN"

    # Loss should generally decrease (even on tiny random data, optim should overfit fast)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_train_smoke.py::test_hicore_smoke -v`
Expected: passes, loss decreases over 10 steps.

- [ ] **Step 3: Run full test suite for regression check**

Run: `pytest -v`
Expected: 100% green; no LDC v2 or baseline tests broken.

- [ ] **Step 4: Commit**

```bash
git add tests/test_train_smoke.py
git commit -m "test(hicore): end-to-end smoke test"
```

---

## Task 15: First Real Run — SCAN simple

This is the first checkpoint where we expect HiCoRe to behave like a working seq2seq on the easy split. If `simple` doesn't reach high val_seq_acc, the architecture has a bug.

**Files:** none new — running existing scripts.

- [ ] **Step 1: Push branch and pull on Lightning AI**

```bash
git push -u origin claude/hicore-dev
```

On Lightning AI:
```bash
cd /teamspace/studios/this_studio/ldc-research
git fetch
git checkout claude/hicore-dev
source .venv/bin/activate
pip install -e .
```

- [ ] **Step 2: Verify deps install**

```bash
python -c "import geoopt; from src.hicore.model import HiCoReModel; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Launch training**

```bash
nohup python -m src.train --config-name hicore_scan_simple > outputs/hicore_simple.log 2>&1 &
echo PID=$!
```

- [ ] **Step 4: Monitor, expect ~%99 val_seq_acc**

Periodically: `tail -20 outputs/hicore_simple.log`.

Expected at step 20K:
- `eval@step=20000: seq_acc ≥ 0.95` (lenient: 0.90)
- `TEST (held-out): seq_acc ≥ 0.95`
- No NaN in loss, both `l_lm` and `l_role` decreasing

If `seq_acc < 0.5` at step 20K: stop and debug. Likely role_inference saturation, or refinement diverging.

- [ ] **Step 5: Record result**

Append to `results/2026-05-10-ldc-v2-postmortem.md` or create `results/2026-05-XX-hicore-simple.md`:
- Final val_seq_acc, test_seq_acc
- Loss curves (text summary)
- Comparison to baseline + LDC v2 simple results
- Notes on training stability

- [ ] **Step 6: Commit results**

```bash
git add results/
git commit -m "results(hicore): SCAN simple baseline result"
```

---

## Task 16: Real Run — SCAN addprim_jump (the headline test)

The compositional generalization test. This is the empirical fork: HiCoRe's hypothesis is that typed roles + TPR binding + auxiliary role supervision will generalize to unseen primitive combinations.

**Files:** none new.

- [ ] **Step 1: Launch addjump training**

On Lightning AI:
```bash
nohup python -m src.train --config-name hicore_scan_addjump > outputs/hicore_addjump.log 2>&1 &
echo PID=$!
```

- [ ] **Step 2: Monitor**

`tail -20 outputs/hicore_addjump.log` periodically. Expected at step 20K:
- val_seq_acc ≥ 0.95 (in-distribution; should be near-perfect)
- TEST (held-out) seq_acc: this is the unknown. Expected ranges:
  - **≥ 0.20:** architectural hypothesis lives — go to ablations (Task 17+)
  - **0.05 – 0.20:** modest signal, needs hyperparameter tuning
  - **< 0.05:** matches LDC v2 failure pattern; another pivot needed

- [ ] **Step 3: Record result**

Create `results/2026-05-XX-hicore-addjump.md` with:
- Full numerical results (val + test, both losses, accuracy curves)
- Side-by-side comparison vs baseline transformer (%0.09) and LDC v2 (%0.0)
- Initial interpretation
- Decision: proceed to ablations vs another pivot

- [ ] **Step 4: Commit**

```bash
git add results/
git commit -m "results(hicore): SCAN addjump compositional gen result"
```

---

## Task 17: Ablation — No Role Supervision (H3)

Set `beta_role=0.0`. Tests whether the auxiliary role loss is the load-bearing piece.

**Files:**
- Create: `experiments/configs/hicore_addjump_ablate_no_role.yaml`

- [ ] **Step 1: Write config**

Same as `hicore_scan_addjump.yaml` but:
- `experiment_name: hicore_addjump_ablate_no_role`
- `model.beta_role: 0.0`
- `data.include_roles: false` (no need for role labels if loss is zero)
- `wandb.tags: [hicore, ablation, no_role]` (block style)

- [ ] **Step 2: Run on Lightning AI**

```bash
nohup python -m src.train --config-name hicore_addjump_ablate_no_role > outputs/hicore_ablate_no_role.log 2>&1 &
```

- [ ] **Step 3: Record + commit**

Append result to `results/2026-05-XX-hicore-ablations.md`. Compare H2 (full HiCoRe) test_seq_acc vs H3 (no role sup).

```bash
git add experiments/configs/hicore_addjump_ablate_no_role.yaml results/
git commit -m "results(hicore): ablation H3 — no role supervision"
```

---

## Task 18: Ablation — No Refinement (H5)

Set `refine_iter=0`. Tests whether iterative refinement contributes.

**Files:**
- Create: `experiments/configs/hicore_addjump_ablate_no_refine.yaml`

- [ ] **Step 1: Write config**

Same as `hicore_scan_addjump.yaml` but:
- `experiment_name: hicore_addjump_ablate_no_refine`
- `model.refine_iter: 0`
- `wandb.tags: [hicore, ablation, no_refine]`

- [ ] **Step 2: Run on Lightning AI**

```bash
nohup python -m src.train --config-name hicore_addjump_ablate_no_refine > outputs/hicore_ablate_no_refine.log 2>&1 &
```

- [ ] **Step 3: Record + commit**

Append H5 result. Compare H2 vs H5.

```bash
git add experiments/configs/hicore_addjump_ablate_no_refine.yaml results/
git commit -m "results(hicore): ablation H5 — no refinement"
```

---

## Task 19: Ablation — No TPR (H7)

Replaces TPR binding with a flat memory (slot-attention-style, like LDC v2). Tests whether the binding mechanism specifically — not just typed slots — is what's contributing.

**Files:**
- Modify: `src/hicore/model.py` (add a `binding_mode` config option)
- Create: `experiments/configs/hicore_addjump_ablate_no_tpr.yaml`

- [ ] **Step 1: Add `binding_mode` to HiCoReConfig**

In `src/hicore/model.py`, add to `HiCoReConfig`:

```python
    binding_mode: str = "tpr"  # "tpr" | "concat"
```

In `HiCoReModel.forward`, replace the `T = bind_tpr(...)` line with:

```python
        if self.cfg.binding_mode == "tpr":
            T = bind_tpr(fillers, role_emb, src_mask)
        elif self.cfg.binding_mode == "concat":
            # Flat alternative: concat fillers and roles, project back to (d_filler, d_role)
            B, N, _ = fillers.shape
            d_filler = self.cfg.encoder.d_model
            d_role = self.cfg.inventory.d_role
            mean_fill = (fillers * src_mask.float().unsqueeze(-1)).sum(dim=1) / \
                        src_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            mean_role = (role_emb * src_mask.float().unsqueeze(-1)).sum(dim=1) / \
                        src_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            T = mean_fill.unsqueeze(-1) * mean_role.unsqueeze(-2)
        else:
            raise ValueError(f"unknown binding_mode: {self.cfg.binding_mode}")
```

(The `concat` mode collapses to a rank-1 outer product of mean filler × mean role — empirically this should hurt because there's no per-token compositional structure. That's the whole point of the ablation.)

- [ ] **Step 2: Write config**

Same as `hicore_scan_addjump.yaml` but:
- `experiment_name: hicore_addjump_ablate_no_tpr`
- `model.binding_mode: concat`
- `wandb.tags: [hicore, ablation, no_tpr]`

- [ ] **Step 3: Run on Lightning AI**

```bash
nohup python -m src.train --config-name hicore_addjump_ablate_no_tpr > outputs/hicore_ablate_no_tpr.log 2>&1 &
```

- [ ] **Step 4: Record + commit**

Append H7 result.

```bash
git add src/hicore/model.py experiments/configs/hicore_addjump_ablate_no_tpr.yaml results/
git commit -m "feat(hicore): ablation H7 — non-TPR binding mode + run results"
```

---

## Task 20: Final Synthesis Report

Aggregate all results into a single decision document.

**Files:**
- Create: `results/2026-05-XX-hicore-final.md`

- [ ] **Step 1: Write the synthesis**

Required sections:
- Executive summary (1 paragraph): does HiCoRe beat baseline on add_jump?
- Numerical table: baseline, LDC v2, HiCoRe full, H3, H5, H7, on val_seq_acc and test_seq_acc
- Ablation interpretation: which component carries the most weight?
- Loss curves: a textual description (or generated plot, if matplotlib is available)
- Cost analysis: total GPU hours used, wall-clock time per run
- Decision (one of):
  - **Green:** test_seq_acc ≥ 0.20 → continue to Phase HC-4 (adaptive halting), then COGS
  - **Yellow:** 0.05 ≤ test_seq_acc < 0.20 → hyperparameter sweep before deciding
  - **Red:** test_seq_acc < 0.05 → second consecutive pivot; rethink whether 1-10M param is fundamentally insufficient

- [ ] **Step 2: Commit**

```bash
git add results/2026-05-XX-hicore-final.md
git commit -m "results(hicore): final synthesis + go/pivot decision"
```

- [ ] **Step 3: Push**

```bash
git push
```

---

## Summary Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Project setup + geoopt | ☐ |
| 2 | Role inventory (Poincaré) | ☐ |
| 3 | TPR bind/unbind | ☐ |
| 4 | Role inference head | ☐ |
| 5 | Filler encoder | ☐ |
| 6 | Iterative refinement | ☐ |
| 7 | Typed decoder | ☐ |
| 8 | End-to-end HiCoReModel | ☐ |
| 9 | SCAN role tagger | ☐ |
| 10 | Loader role label support | ☐ |
| 11 | Build pipeline integration | ☐ |
| 12 | Train loop role pass-through | ☐ |
| 13 | Hydra configs | ☐ |
| 14 | E2E smoke test | ☐ |
| 15 | Run: SCAN simple | ☐ |
| 16 | Run: SCAN addprim_jump (headline) | ☐ |
| 17 | Ablation H3 (no role) | ☐ |
| 18 | Ablation H5 (no refinement) | ☐ |
| 19 | Ablation H7 (no TPR) | ☐ |
| 20 | Final synthesis | ☐ |

---

## Risk Notes (carried from HICORE.md)

- **R3 hyperbolic numerical stability:** the simple `tanh`-projected raw parameterization in `RoleInventory` avoids `geoopt` Riemannian optimizers in Phase HC-1. If `_raw` grows large, the `tanh` saturates and gradients vanish. Mitigation: keep `init_scale=0.1`, monitor `_raw.norm()` over training; if saturation observed, switch to `geoopt.ManifoldParameter` in a follow-up.
- **R4 adaptive halting:** deferred to Phase HC-4. Phase HC-1 uses fixed `refine_iter=3`.
- **R6 val %100 / test %0:** all training configs use `train.py` which already runs held-out test eval after the loop; the result IS visible per run.

## What this plan does NOT cover

- COGS dataset integration (Phase HC-5)
- Adaptive halting / ACT (Phase HC-4)
- Curriculum scheduling (separate sub-plan if needed)
- Unsupervised role induction (Phase HC-3+)
- Multi-token same-role disambiguation beyond the basic TPR sum

These are out of scope for the Phase HC-1/HC-2 minimum-viable plan and will get their own sub-plans if results justify continuation.
