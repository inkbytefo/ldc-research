"""Compare clean (encoder C_0) vs refined (diffusion C*) inference.

Loads a trained LDC v2 checkpoint and runs the held-out test set twice:
once with use_clean=True (encoder C_0 directly) and once with
use_clean=False (run reverse diffusion to get refined C*).

Usage:
    python -m scripts.eval_refined outputs/ldc_v2_scan_addjump/best.pt
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from src.build import build_loaders, build_model
from src.eval import _strip_after_eos
from src.ldc.model import LDCModel


@torch.no_grad()
def eval_with_mode(model, loader, device, tgt_tok, max_gen_len, use_clean):
    model.eval()
    seq_correct = 0
    seq_total = 0
    tok_correct = 0
    tok_total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        gen = model.generate(
            batch["src"], batch["src_mask"],
            sos_id=tgt_tok.sos_id, eos_id=tgt_tok.eos_id,
            max_len=max_gen_len, use_clean=use_clean,
        )
        B = batch["src"].size(0)
        for b in range(B):
            gen_ids = gen[b].tolist()[1:]
            gen_clean = _strip_after_eos(gen_ids, tgt_tok.eos_id, tgt_tok.pad_id)
            ref_ids = batch["tgt"][b].tolist()[1:]
            ref_clean = _strip_after_eos(ref_ids, tgt_tok.eos_id, tgt_tok.pad_id)
            if gen_clean == ref_clean:
                seq_correct += 1
            seq_total += 1
            n = min(len(gen_clean), len(ref_clean))
            for i in range(n):
                if gen_clean[i] == ref_clean[i]:
                    tok_correct += 1
            tok_total += max(len(gen_clean), len(ref_clean))
    return {
        "seq_acc": seq_correct / max(1, seq_total),
        "tok_acc": tok_correct / max(1, tok_total),
        "n": seq_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg)
    model = build_model(cfg, loaders["src_tok"], loaders["tgt_tok"]).to(device)
    model.load_state_dict(ckpt["model"])

    if not isinstance(model, LDCModel):
        raise ValueError("Only LDCModel supports clean vs refined comparison.")

    test_loader = loaders["test_loader"]
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Split: {cfg.data.scan_split}")
    print(f"Test size: {len(test_loader.dataset)}")

    print("\n--- CLEAN (encoder C_0, no diffusion) ---")
    clean_m = eval_with_mode(model, test_loader, device, loaders["tgt_tok"],
                             cfg.data.max_tgt_len, use_clean=True)
    print(clean_m)

    print("\n--- REFINED (reverse diffusion C*) ---")
    refined_m = eval_with_mode(model, test_loader, device, loaders["tgt_tok"],
                               cfg.data.max_tgt_len, use_clean=False)
    print(refined_m)

    delta = refined_m["seq_acc"] - clean_m["seq_acc"]
    print(f"\nDelta seq_acc (refined - clean): {delta:+.4f}")


if __name__ == "__main__":
    main()
