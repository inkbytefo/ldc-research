# LDC (Latent Diffusion Cognition) Research Project — Briefing for Claude Code

## What I need from you

I'm working on a research project to design and validate a new neural architecture called LDC (Latent Diffusion Cognition). I'm not an ML expert. I need you to act as my research engineer: read the design docs, do current literature research, build a minimal validation pipeline, and run experiments.

**Important: I do NOT need you to build a production-scale model. I need a small-scale prototype to test whether the core architectural ideas work in principle.**

---

## Context: What's been done so far

I've worked with another Claude instance to design this architecture and produce these documents (all in this repo):

1. `latent-diffusion-cognition-mimarisi.md` — The full architecture design (in Turkish, you can translate as you go)
2. `ldc-dogrulama-raporu.md` — Validation report with literature review and toy experiments  
3. `ldc-gelistirme-rehberi.md` — Step-by-step development guide
4. `ne-yaptik-durust-ozet.md` — Honest summary of what's been validated and what hasn't

Before doing anything, **read all four documents carefully**. They contain the design rationale, literature review, and explicit limitations.

---

## Current state of the architecture

**LDC v2 design (the version we're going with):**

```
Input text → Transformer Encoder → Concept Graph in hybrid manifold
                                           ↓
                                  Adaptive-depth diffusion
                                  (refines the concept graph)
                                           ↓
                                  Refined concept representation
                                           ↓
                                  Transformer Decoder (autoregressive)
                                           ↓
                                       Output text
```

**Key architectural choices:**
- **Hybrid manifold for concepts:** Euclidean + hyperbolic (Poincaré ball, dim ≥ 3)
- **Graph-structured concept space:** Multiple "concept slots" with typed relations between them, NOT a single vector (this is what differentiates us from Meta's LCM)
- **Diffusion in concept space, NOT in token space:** Iterative refinement of the latent graph
- **Adaptive computation:** Number of diffusion steps depends on task difficulty
- **Autoregressive decoder:** SONAR-LLM (2025) showed that autoregressive decoding works better than diffusion for language output, so we keep diffusion only in the "thinking" phase

---

## What I need you to do (in order)

### Phase 0: Research & Understanding (1-2 days)

1. **Read all four documents in this repo.**

2. **Search the web for these papers and read them (latest 2024-2026 versions):**
   - Meta's "Large Concept Models" (LCM) paper — the closest thing to what we're building
   - SONAR-LLM (arXiv 2508.05305) — important baseline
   - HELM: Hyperbolic Large Language Models (NeurIPS 2025)
   - Coconut (Meta, arXiv 2412.06769) — chain of continuous thought
   - LaDiR (arXiv 2510.04573) — latent diffusion for text reasoning
   - "Diffusion Beats Autoregressive in Data-Constrained Settings" (CMU 2025)

3. **Write a short summary (2-3 pages, in this repo as `literature-review.md`) covering:**
   - What's already been tried in this space
   - Where our LDC v2 differs from existing work
   - What the most likely failure modes are based on prior work
   - What evaluation benchmarks would be most informative

### Phase 1: Build the Validation Platform (3-7 days)

Set up a clean experimental pipeline. Create the following structure:

```
ldc-research/
├── README.md
├── docs/                     # all the .md files
├── src/
│   ├── baselines/
│   │   └── transformer.py    # vanilla transformer baseline
│   ├── ldc/
│   │   ├── encoder.py
│   │   ├── concept_space.py  # hybrid manifold representation
│   │   ├── diffusion.py      # diffusion core
│   │   ├── decoder.py
│   │   └── model.py
│   ├── data/
│   │   ├── scan.py           # SCAN dataset loader
│   │   └── synthetic.py      # synthetic compositional tasks
│   └── train.py
├── experiments/
│   └── configs/              # Hydra configs for each experiment
├── results/                  # wandb-logged or local results
└── tests/                    # unit tests
```

**Important practical requirements:**
- Use PyTorch
- Use `geoopt` library for hyperbolic operations (Poincaré ball)
- Use Weights & Biases (`wandb`) for experiment tracking — this is non-negotiable for research
- Use Hydra for configuration management
- Write unit tests for the geometry operations and diffusion steps (these are easy to get subtly wrong)

**Start small:**
- Initial models should be 1-10M parameters, NOT billion-scale
- Use SCAN dataset first (small, fast, well-known)
- Get a working pipeline before optimizing anything

### Phase 2: Run the Critical Experiments (1-2 weeks)

The goal is to answer ONE question: **does LDC v2 beat a same-size transformer on compositional generalization tasks?**

**Required experiments:**

**Experiment 1: Baseline transformer on SCAN**
- 6-layer transformer, ~1M params
- Train on SCAN train split
- Evaluate on:
  - Standard test split
  - "add jump" compositional split (this is the hard one)
  - "around right" split

**Experiment 2: LDC v2 on SCAN**
- Same parameter budget as Experiment 1
- Same training data and hyperparameters where possible
- Evaluate on the same splits

**Experiment 3: Ablation studies**
Run LDC v2 with each component disabled, one at a time:
- LDC without hyperbolic component (pure Euclidean)
- LDC without graph structure (single latent vector like LCM)
- LDC without adaptive computation (fixed number of diffusion steps)
- LDC without diffusion (just the encoder/decoder)

This tells us which components actually matter.

**Experiment 4: If Phase 2 shows promise, scale to 10M parameters and try GSM8K (math)**

### Phase 3: Honest Reporting (3-5 days)

Write a final report (`results/report.md`) that includes:
- All numerical results in tables
- Loss curves and training dynamics
- Honest assessment: did LDC beat the baseline? By how much? On which splits?
- Failure analysis: what didn't work and why?
- Cost analysis: how much slower is LDC than the baseline (per training step, per sample)?
- Recommendations: should this research direction continue, pivot, or stop?

**This last point is critical. If LDC doesn't show clear advantage, say so. We need truth, not flattery.**

---

## What I want you to NOT do

1. **Do not skip the baseline.** The transformer baseline must be carefully tuned, not a strawman. Half of researchers' inflated claims come from weak baselines.

2. **Do not over-engineer the first version.** Get a working pipeline first, optimize later. A working bad model is worth more than a beautifully designed model that doesn't run.

3. **Do not present optimistic results without ablations.** If LDC works, we need to know *why* it works. Is it the hyperbolic geometry? The diffusion? The graph structure? Without ablations, we don't know.

4. **Do not invent numbers.** If something didn't run, say so. If a result is suspicious, investigate before reporting.

5. **Do not make this look bigger than it is.** This is a small-scale research prototype, not a foundation model. The success criterion is "shows enough promise to justify continued research," not "beats GPT-4."

---

## Practical questions you might have

**Q: How much compute do I have access to?**
Tell me what you need. Initially we should be able to run on a single GPU (Colab Pro, RunPod, or whatever I can rent for ~$50-200/month).

**Q: What if a step fails?**
Document the failure clearly, propose alternatives, and ask me. Don't silently work around problems — I want to learn what's hard.

**Q: How fast should this go?**
The whole project (Phases 0-3) should take 3-6 weeks of your active work, depending on how many experiments succeed/fail.

**Q: What if I find something that contradicts the design?**
Tell me. The design is a hypothesis, not gospel. If literature search reveals a fatal flaw, I want to know before we waste time implementing.

---

## Important: Manage my expectations honestly

I'm not an ML expert. I might ask for things that are unrealistic. Please push back when I do. Specifically:

- If I ask for a result by a specific date and it's not realistic, say so
- If I propose a design change that's likely to break things, warn me
- If results look suspiciously good, investigate before celebrating
- If results look bad, don't soften them — we need accurate information to make decisions

The best outcome of this project is **truth**: knowing whether LDC v2 is a real research direction or not. Both "yes" and "no" are valuable answers. The worst outcome is fake "yes" that wastes the next 12 months.

---

## First message I want from you

After reading the documents, please respond with:

1. A 2-paragraph summary of the architecture as you understand it
2. The 3 biggest risks you see based on what you've read
3. A proposed concrete plan for the first week (what you'll do day by day)
4. Any clarifying questions

Do NOT start coding until I confirm the plan.

---

*Author: [your name]*
*Project start date: [today]*
*This briefing version: 1.0*
