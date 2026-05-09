"""Training entry point.

Run with:
    python -m src.train --config-name baseline_scan
    python -m src.train --config-name ldc_v2_scan
    python -m src.train --config-name baseline_scan_smoke   # CPU smoke test

Hydra picks the config from `experiments/configs/`. Wandb logging is
opt-in via `wandb.enabled=true`.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.build import build_loaders, build_model
from src.eval import evaluate
from src.utils import count_parameters, format_param_count, set_seed

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "experiments" / "configs")


def _build_lr_lambda(warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # cosine decay over the remaining steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _wandb_init(cfg: DictConfig) -> Any | None:
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb requested but not installed; skipping.")
        return None
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
    )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="baseline_scan")
def main(cfg: DictConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if str(device) != cfg.device:
        log.warning("Requested device=%s but using %s (cuda not available?)", cfg.device, device)

    loaders = build_loaders(cfg)
    model = build_model(cfg, loaders["src_tok"], loaders["tgt_tok"]).to(device)
    n_params = count_parameters(model)
    log.info("Model: %s | Parameters: %s", cfg.model.type, format_param_count(n_params))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer, lr_lambda=_build_lr_lambda(cfg.trainer.warmup_steps, cfg.trainer.max_steps)
    )

    wandb_run = _wandb_init(cfg)
    if wandb_run is not None:
        wandb_run.summary["param_count"] = n_params

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders.get("test_loader")

    step = 0
    best_val_acc = -1.0
    last_log_time = time.time()
    train_iter = iter(train_loader)
    final_metrics: dict[str, float] = {}

    while step < cfg.trainer.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        model.train()
        out = model(batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"])
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.trainer.grad_clip and cfg.trainer.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % cfg.trainer.log_every == 0:
            now = time.time()
            steps_per_sec = cfg.trainer.log_every / max(1e-6, now - last_log_time)
            last_log_time = now
            log_msg = {
                "step": step,
                "loss": float(loss.item()),
                "lr": optimizer.param_groups[0]["lr"],
                "token_acc": float(out["token_acc"].item()),
                "steps_per_sec": steps_per_sec,
            }
            if "l_diff" in out:
                log_msg["l_diff"] = float(out["l_diff"].item())
                log_msg["l_lm"] = float(out["l_lm"].item())
            log.info("train: %s", log_msg)
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in log_msg.items() if k != "step"}, step=step)

        if step % cfg.trainer.eval_every == 0 or step == cfg.trainer.max_steps:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                src_tok=loaders["src_tok"],
                tgt_tok=loaders["tgt_tok"],
                max_batches=cfg.trainer.num_eval_batches,
                num_gen_examples=cfg.trainer.num_gen_examples,
                max_gen_len=cfg.data.max_tgt_len,
            )
            log.info("eval@step=%d: %s", step, val_metrics)
            if wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

            if val_metrics["token_acc"] > best_val_acc:
                best_val_acc = val_metrics["token_acc"]
                ckpt_path = output_dir / "best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "step": step,
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
                log.info("new best (%.4f) -> %s", best_val_acc, ckpt_path)
            final_metrics = val_metrics

        if step % cfg.trainer.save_every == 0:
            ckpt_path = output_dir / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

    # ---- Final evaluation on held-out test set ----
    test_metrics: dict[str, float] = {}
    if test_loader is not None:
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            src_tok=loaders["src_tok"],
            tgt_tok=loaders["tgt_tok"],
            max_batches=0,  # full test set, no limit
            num_gen_examples=0,
            max_gen_len=cfg.data.max_tgt_len,
        )
        log.info("TEST (held-out): %s", test_metrics)
        if wandb_run is not None:
            wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()}, step=step)

    if wandb_run is not None:
        wandb_run.finish()

    log.info("Done. val_metrics: %s | test_metrics: %s", final_metrics, test_metrics)
    return {**final_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}


if __name__ == "__main__":
    main()
