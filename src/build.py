"""Factory: turn a Hydra config into a model + dataloaders.

Centralizing this keeps `train.py` short and lets unit tests instantiate
the same objects the trainer would.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.baselines.transformer import TransformerConfig, TransformerSeq2Seq
from src.data.scan import build_scan_loaders
from src.data.synthetic import build_synthetic_loaders
from src.data.tokenizer import WordTokenizer
from src.ldc.concept_space import ConceptSpaceConfig
from src.ldc.decoder import LDCDecoderConfig
from src.ldc.encoder import LDCEncoderConfig
from src.ldc.graph_denoiser import GraphDenoiserConfig
from src.ldc.model import LDCConfig, LDCModel


def build_loaders(cfg: DictConfig) -> dict[str, Any]:
    src = cfg.data.source
    if src == "scan":
        train, val, test, src_tok, tgt_tok = build_scan_loaders(
            split=cfg.data.scan_split,
            data_root=cfg.data.data_root,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            max_src_len=cfg.data.max_src_len,
            max_tgt_len=cfg.data.max_tgt_len,
            val_fraction=cfg.data.val_fraction,
            seed=cfg.seed,
        )
        return {
            "train_loader": train,
            "val_loader": val,
            "test_loader": test,
            "src_tok": src_tok,
            "tgt_tok": tgt_tok,
        }
    elif src == "synthetic":
        train, val, tok = build_synthetic_loaders(
            n_train=cfg.data.synthetic_train_size,
            n_val=cfg.data.synthetic_val_size,
            batch_size=cfg.data.batch_size,
            seed=cfg.seed,
            n_letters=cfg.data.synthetic_letters,
        )
        return {
            "train_loader": train,
            "val_loader": val,
            "test_loader": val,  # synthetic uses val as test
            "src_tok": tok,
            "tgt_tok": tok,
        }
    else:
        raise ValueError(f"unknown data.source: {src!r}")


def build_model(cfg: DictConfig, src_tok: WordTokenizer, tgt_tok: WordTokenizer) -> nn.Module:
    mtype = cfg.model.type
    if mtype == "baseline":
        return TransformerSeq2Seq(
            TransformerConfig(
                src_vocab_size=src_tok.vocab_size,
                tgt_vocab_size=tgt_tok.vocab_size,
                d_model=cfg.model.d_model,
                n_heads=cfg.model.n_heads,
                n_enc_layers=cfg.model.n_enc_layers,
                n_dec_layers=cfg.model.n_dec_layers,
                ffn_mult=cfg.model.ffn_mult,
                dropout=cfg.model.dropout,
                max_src_len=cfg.data.max_src_len,
                max_tgt_len=cfg.data.max_tgt_len,
                src_pad_id=src_tok.pad_id,
                tgt_pad_id=tgt_tok.pad_id,
            )
        )
    if mtype == "ldc_v2":
        space = ConceptSpaceConfig(
            num_slots=cfg.model.num_slots,
            d_euclidean=cfg.model.d_concept,
            d_hyperbolic=0,
            num_relations=cfg.model.num_relations,
        )
        encoder = LDCEncoderConfig(
            vocab_size=src_tok.vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.enc_layers,
            ffn_mult=cfg.model.ffn_mult,
            dropout=cfg.model.dropout,
            max_len=cfg.data.max_src_len,
            pad_id=src_tok.pad_id,
        )
        decoder = LDCDecoderConfig(
            vocab_size=tgt_tok.vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.dec_layers,
            ffn_mult=cfg.model.ffn_mult,
            dropout=cfg.model.dropout,
            max_len=cfg.data.max_tgt_len,
            pad_id=tgt_tok.pad_id,
            d_concept=space.d_total,
        )
        denoiser = GraphDenoiserConfig(
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.denoiser_layers,
            ffn_mult=cfg.model.ffn_mult,
            dropout=cfg.model.dropout,
            num_slots=space.num_slots,
            num_relations=space.num_relations,
            d_concept=space.d_total,
        )
        return LDCModel(
            LDCConfig(
                encoder=encoder,
                decoder=decoder,
                space=space,
                denoiser=denoiser,
                num_diffusion_steps=cfg.model.num_diffusion_steps,
                schedule=cfg.model.schedule,
                alpha_lm=cfg.model.alpha_lm,
                alpha_diff=cfg.model.alpha_diff,
                decoder_input=cfg.model.decoder_input,
            )
        )
    raise ValueError(f"unknown model.type: {mtype!r}")
