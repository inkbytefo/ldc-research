# LDC (Latent Diffusion Cognition) Research Project

Research project to design and validate a new neural architecture called LDC.

## Overview
LDC v2 is a hybrid neural architecture that performs iterative refinement in a structured latent space (Concept Graph) residing in a hybrid manifold (Euclidean + Hyperbolic).

## Project Structure
- `docs/`: Architecture design, validation reports, roadmap.
- `src/`:
  - `baselines/transformer.py` — encoder-decoder transformer baseline
  - `ldc/` — LDC v2 components (encoder, concept_space, diffusion, graph_denoiser, decoder, model)
  - `data/` — SCAN loader, synthetic toy tasks, tokenizer
  - `modules/` — shared building blocks (positional, time embeddings, masks)
  - `train.py`, `eval.py`, `build.py` — entry points
- `experiments/configs/`: Hydra configs (`baseline_scan.yaml`, `ldc_v2_scan.yaml`, `*_smoke.yaml`).
- `tests/`: Unit + smoke tests.

## Getting Started
1. `pip install -r requirements.txt` (or `make setup`).
2. Smoke run on CPU: `python -m src.train --config-name baseline_scan_smoke`
3. Full SCAN run on GPU: `python -m src.train --config-name baseline_scan device=cuda data.data_root=/path/to/scan`
4. LDC v2 run: `python -m src.train --config-name ldc_v2_scan`
5. Tests: `make test`

## SCAN data
The loader expects files at `${data_root}/SCAN/{split}/tasks_{train,test}*.txt`.
Download from https://github.com/brendenlake/SCAN.

## Documents (read in this order)
1. `docs/yol-haritasi.md` — phased roadmap with Go/No-Go gates (start here).
2. `docs/ldc-dogrulama-raporu.md` — frozen LDC v2 architecture decisions.
3. `docs/latent-diffusion-cognition-mimarisi.md` — original vision.
4. `docs/ldc-gelistirme-rehberi.md` — engineering guide.
5. `docs/ne-yaptik-durust-ozet.md` — honest status of what is/isn't validated.
