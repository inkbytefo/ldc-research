"""End-to-end LDC v2 model.

Loss = L_diff + alpha_lm * L_lm

Per `docs/yol-haritasi.md` section 5.2 (Phase 1 simplified):
  - Encoder produces C_0 (initial concept graph) and context.
  - Diffusion forward q(C_t | C_0) = sqrt(a) C_0 + sqrt(1-a) eps.
  - Denoiser predicts eps conditioned on (C_t, t, context).
  - Decoder is teacher-forced on the *clean* C_0 -> standard cross-entropy.
  - At inference, denoiser refines noise -> C* conditioned on context, then
    the decoder generates from C*.

The schemes are decoupled: training the decoder against C_0 gives it a
clean signal; the diffusion learns a model over C_0 conditioned on
context, so at inference we sample from p(C_0 | context). This is the
standard latent-diffusion training recipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_space import ConceptGraph, ConceptSpaceConfig
from .decoder import LDCDecoder, LDCDecoderConfig
from .diffusion import GaussianDiffusion
from .encoder import LDCEncoder, LDCEncoderConfig
from .graph_denoiser import GraphDenoiser, GraphDenoiserConfig


@dataclass
class LDCConfig:
    encoder: LDCEncoderConfig
    decoder: LDCDecoderConfig
    space: ConceptSpaceConfig = field(default_factory=ConceptSpaceConfig)
    denoiser: GraphDenoiserConfig = field(default_factory=GraphDenoiserConfig)
    num_diffusion_steps: int = 20
    schedule: str = "cosine"
    alpha_lm: float = 1.0  # weight on language modelling loss
    alpha_diff: float = 1.0  # weight on diffusion loss
    # Phase 1: at training time the decoder always sees the clean C_0.
    # Phase 6: switch to refined C* with adaptive halting.
    decoder_input: str = "clean"  # "clean" | "refined"


class LDCModel(nn.Module):
    def __init__(self, config: LDCConfig) -> None:
        super().__init__()
        self.config = config

        # Cross-check shapes between sub-configs to fail loud if the user
        # mismatches d_concept anywhere.
        d_concept = config.space.d_total
        assert config.denoiser.d_concept == d_concept, (
            f"denoiser.d_concept={config.denoiser.d_concept} != space.d_total={d_concept}"
        )
        assert config.decoder.d_concept == d_concept, (
            f"decoder.d_concept={config.decoder.d_concept} != space.d_total={d_concept}"
        )
        assert config.denoiser.num_slots == config.space.num_slots
        assert config.denoiser.num_relations == config.space.num_relations

        self.encoder = LDCEncoder(config.encoder, config.space)
        self.diffusion = GaussianDiffusion(
            num_steps=config.num_diffusion_steps, schedule=config.schedule
        )
        self.denoiser = GraphDenoiser(config.denoiser)
        self.decoder = LDCDecoder(config.decoder)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -- Public API ---------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        c_0_graph, context, ctx_mask = self.encoder(src, src_mask)
        c_0 = c_0_graph.nodes  # (B, K, d_concept)
        B = c_0.size(0)
        device = c_0.device

        # ---- Diffusion loss ----
        t = torch.randint(0, self.diffusion.num_steps, (B,), dtype=torch.long, device=device)
        noise = torch.randn_like(c_0)
        c_t = self.diffusion.q_sample(c_0, t, noise)
        noise_pred = self.denoiser(c_t, t, context, ctx_mask)
        l_diff = self.diffusion.loss(noise_pred, noise)

        # ---- Decoder loss (teacher forced on clean C_0) ----
        tgt_in = tgt[:, :-1]
        tgt_in_mask = tgt_mask[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_out_mask = tgt_mask[:, 1:]
        if self.config.decoder_input == "clean":
            decoder_memory = c_0
        elif self.config.decoder_input == "refined":
            # Predicted x_0 from a single denoising step at random t.
            # No detach: LM loss must flow back through the diffusion so the
            # denoiser learns to produce decoder-useful refined graphs.
            sqrt_one_minus = self.diffusion._extract(
                self.diffusion.sqrt_one_minus_alphas_cumprod, t, c_t.shape
            )
            sqrt_alpha = self.diffusion._extract(
                self.diffusion.sqrt_alphas_cumprod, t, c_t.shape
            ).clamp_min(1e-6)
            decoder_memory = (c_t - sqrt_one_minus * noise_pred) / sqrt_alpha
        else:
            raise ValueError(f"unknown decoder_input: {self.config.decoder_input}")

        logits = self.decoder(tgt_in, tgt_in_mask, decoder_memory)
        l_lm = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.config.decoder.pad_id,
            reduction="mean",
        )

        loss = self.config.alpha_diff * l_diff + self.config.alpha_lm * l_lm

        with torch.no_grad():
            preds = logits.argmax(-1)
            valid = tgt_out_mask
            correct = ((preds == tgt_out) & valid).sum().float()
            n = valid.sum().clamp_min(1).float()
            tok_acc = correct / n

        return {
            "loss": loss,
            "l_diff": l_diff.detach(),
            "l_lm": l_lm.detach(),
            "logits": logits,
            "token_acc": tok_acc,
        }

    @torch.no_grad()
    def refine(
        self, src: torch.Tensor, src_mask: torch.Tensor, num_steps: int | None = None
    ) -> ConceptGraph:
        """Run the reverse diffusion process: noise -> C* conditioned on context."""
        c_0_graph, context, ctx_mask = self.encoder(src, src_mask)
        B = c_0_graph.nodes.size(0)
        device = c_0_graph.nodes.device

        c_t = torch.randn_like(c_0_graph.nodes)
        steps = num_steps or self.diffusion.num_steps
        for t_int in reversed(range(steps)):
            t = torch.full((B,), t_int, dtype=torch.long, device=device)
            noise_pred = self.denoiser(c_t, t, context, ctx_mask)
            c_t = self.diffusion.p_sample(c_t, t, noise_pred)
        return ConceptGraph(nodes=c_t, config=c_0_graph.config)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
        use_clean: bool | None = None,
    ) -> torch.Tensor:
        """Generate a target sequence.

        `use_clean=None` (default) respects `config.decoder_input`: "clean" ->
        skip diffusion, "refined" -> run reverse diffusion. Pass True/False to
        override (used for ablation A3).
        """
        if use_clean is None:
            use_clean = self.config.decoder_input == "clean"
        if use_clean:
            c_graph, _, _ = self.encoder(src, src_mask)
        else:
            c_graph = self.refine(src, src_mask)
        return self.decoder.generate(c_graph.nodes, sos_id, eos_id, max_len)
