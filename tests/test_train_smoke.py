"""End-to-end smoke test: build loaders + model, run a few train+eval steps."""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.build import build_loaders, build_model
from src.eval import evaluate
from src.utils import set_seed


def _smoke_cfg(model_type: str) -> OmegaConf:
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "experiment_name": "smoke",
            "output_dir": "outputs/smoke",
            "data": {
                "source": "synthetic",
                "data_root": "data",
                "scan_split": "simple",
                "batch_size": 4,
                "num_workers": 0,
                "max_src_len": 16,
                "max_tgt_len": 16,
                "val_fraction": 0.1,
                "synthetic_train_size": 16,
                "synthetic_val_size": 4,
                "synthetic_letters": 4,
            },
            "model": {},
            "trainer": {
                "max_steps": 4,
                "log_every": 100,
                "eval_every": 1000,
                "save_every": 1000,
                "grad_clip": 1.0,
                "lr": 3e-3,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "optimizer": "adamw",
                "amp": False,
                "num_eval_batches": 1,
                "num_gen_examples": 0,
            },
            "wandb": {"enabled": False, "project": "x", "entity": None, "mode": "offline", "tags": []},
        }
    )

    if model_type == "baseline":
        cfg.model = OmegaConf.create({
            "type": "baseline",
            "d_model": 16, "n_heads": 2, "n_enc_layers": 1, "n_dec_layers": 1,
            "ffn_mult": 2, "dropout": 0.0,
        })
    elif model_type == "ldc_v2":
        cfg.model = OmegaConf.create({
            "type": "ldc_v2",
            "d_model": 16, "d_concept": 16, "n_heads": 2,
            "enc_layers": 1, "denoiser_layers": 1, "dec_layers": 1,
            "ffn_mult": 2, "dropout": 0.0,
            "num_slots": 4, "num_relations": 2,
            "num_diffusion_steps": 3, "schedule": "cosine",
            "alpha_lm": 1.0, "alpha_diff": 1.0, "decoder_input": "clean",
        })
    return cfg


def _run_training(model_type: str) -> None:
    cfg = _smoke_cfg(model_type)
    set_seed(cfg.seed)
    loaders = build_loaders(cfg)
    model = build_model(cfg, loaders["src_tok"], loaders["tgt_tok"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr)

    train_iter = iter(loaders["train_loader"])
    losses = []
    for _ in range(cfg.trainer.max_steps):
        batch = next(train_iter)
        out = model(batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].item()))
        assert torch.isfinite(out["loss"])

    metrics = evaluate(
        model,
        loaders["val_loader"],
        device=torch.device("cpu"),
        src_tok=loaders["src_tok"],
        tgt_tok=loaders["tgt_tok"],
        max_batches=1,
        num_gen_examples=0,
    )
    assert "loss" in metrics
    assert "token_acc" in metrics
    assert "seq_acc" in metrics


def test_baseline_smoke():
    _run_training("baseline")


def test_ldc_v2_smoke():
    _run_training("ldc_v2")
