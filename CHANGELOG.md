# Changelog — vllm-sm12x-nvfp4-dflash2

## NIAH gate fix (2026-08-28)

`verify.sh --full` no longer crashes with `TypeError: argument of type
'NoneType' is not a container or iterable` when the model returns a
thinking-enabled response (`content: null`, reasoning in
`reasoning_content`). The NIAH gate was rewritten as `bench/niah_gate.py`:

- **Fixed crash (issue #5):** answer extraction now reads `content` OR
  `reasoning_content` and never assumes a non-null string, matching the safe
  pattern already used by the canary and `bench/vision_gate.py`.
- **Fixed latent bug:** the old gate *built* a haystack but *never sent it* —
  the request only contained the question, so retrieval was impossible by
  construction. The new gate actually plants the codeword in a ~30K-token
  haystack and requires retrieval at two depths (25%/75%).
- **Structural guard:** passes only when `usage.prompt_tokens` proves the
  haystack was ingested, so a request that omits the needle can never pass.
- Requests disable thinking (`chat_template_kwargs.enable_thinking=False`)
  like the vision gate, so the codeword lands in `content` when the server
  honors the kwarg; the `reasoning_content` fallback covers servers that
  don't.

Upgrade: `git pull --ff-only`, then `./verify.sh --full`.

## CPU vision sidecar improvements (2026-08-26)

Host-side sidecar tuning; the `.3` runtime image and all model artifacts are
unchanged. The sidecar is bind-mounted from the host (`./sidecar.py`), so these
changes take effect on `docker compose up -d --force-recreate vision` with no
rebuild.

- **Core cap 4 → 8** (`compose.yaml` vision `cpus`, OMP/MKL/OPENBLAS/TORCH
  threads). Measured ~1.5–1.6× faster encode (memory-bandwidth-bound under
  WSL2, so sub-linear). 1 MP image 16.97 s → 10.66 s.
- **INT8 ViT tower by default** (`SIDECAR_INT8=1`). Dynamic `qint8`
  quantization of the 110 ViT Linear layers (weights INT8, activations fp32).
  ~1.5× on top of the core cap; 1 MP image now ~6.8 s end-to-end (≈2.5× vs the
  original 4-core eager baseline). Embeddings are ~0.90 cosine-similar to fp32
  (0.897–0.906); disable with `SIDECAR_INT8=0` for the fp32 fallback.
- `README.md`, `.env.example` updated to document the 8-CPU / INT8-default
  sidecar.
- **New `BENCHMARKS.md`** — TLDR of expected 5090 decode, prefill, and vision
  sidecar numbers, with a reproduce section.

Upgrade: `git pull --ff-only`, then
`docker compose --profile vision up -d --force-recreate vision`.

## v0.27.1-sm12x-dflash2.3 (2026-08-25)

Optional vision restored with the narrow fused M-RoPE backport; the `.2` base
image and model artifacts remain unchanged.

- Add incremental `0002-qwen3-next-fused-mrope-vision.patch` containing the
  reviewed two production Python changes and targeted CUDA correctness test.
  The minimal Docker overlay is recorded as a two-file, 12,600-byte layer;
  no native CUDA rebuild is required because Triton JIT-compiles the new
  variants and persists them in `TRITON_CACHE_DIR`.
- Restore `--enable-mm-embeds` with zero image/video limits and the bounded
  optional `vision` Compose profile (`4 CPU`, `6 GB`, `pids_limit=256`).
- Restore `VISION_PORT`, `start.sh --vision`, profile-aware status/stop, and
  `verify.sh --vision` for the exact fixture/concurrency gate.
- Evidence: CUDA **9/9**, vision exact fixture **2/2**, and matched c1-c4
  candidate/control measurements are recorded in `EVIDENCE.md`.
- The candidate is validated on SM120 only; SM121 remains unvalidated.
- Upstream duplicate efforts are [vLLM #49744](https://github.com/vllm-project/vllm/pull/49744)
  and [vLLM #43056](https://github.com/vllm-project/vllm/pull/43056); no duplicate
  PR is opened.

## v0.27.1-sm12x-dflash2.2 (2026-08-25)

Text-throughput correction; runtime image and model pins are unchanged.

- Make `--language-model-only` mandatory in `compose.yaml`.
- Remove `--enable-mm-embeds`, the zero-count multimodal limits, the Compose
  vision service, and the `start.sh --vision` deployment path.
- Remove vision-specific status, stop, verify, environment, and release-check
  behavior. The sidecar source and historical vision benchmark remain in-tree
  for future development but are not part of the supported deployment.
- Document the root cause: the embedding-capable Qwen3.5 path disables vLLM's
  fused QK-norm + RoPE + gate decoder kernel even when no image is active.
- Same-image A/B: narrative 61.6 -> 116.2 tok/s; code 105.7 -> 202.4 tok/s;
  GPU allocation about 32,102 -> 30,944 MiB.

Upgrade with `git pull --ff-only`, `docker compose down --remove-orphans`, and
`./start.sh`.

## v0.27.1-sm12x-dflash2.1 (2026-08-25)

First public release of the all-NVFP4 DFlash2 stack.

### What ships
- `0001-v0271-sm12x-dflash2-nvfp4.patch` — 51-file Python-only overlay on
  vLLM v0.27.1 (commit `6e448d0ea`): DFlash2 backport (upstream PR #52816
  final merge), r0b0tlab SM121 safety deltas, NVFP4 non-causal prefill via
  the fa2 backend, fused-KV dequant fallback (#51581 class), mixed
  cache-dtype layout, target RoPE layout copy, non-causal CUDA-graph
  metadata, GDN ReplaySSM speculative half, Mamba page-padding guard,
  runtime-K + GDN active-width fixes, and the FlashInfer #4346 NVFP4
  paged-prefill integration.
- `compose.yaml` — server + optional CPU vision sidecar (profile `vision`),
  one-shot cache/draft initializers, localhost-only bindings, healthchecks,
  restart policy, 8 GiB shared memory, UID 2000 runtime, named model caches.
- `start.sh` / `stop.sh` / `status.sh` / `verify.sh` — preflight (GPU/SM
  12.0/12.1, VRAM, Docker, disk), pinned-image pull, readiness wait, and
  real chat smoke (deterministic canary `19×23 → 437`).
- `build.sh` — reproducible SM120 (x86_64, arch 12.0) and SM121
  (aarch64, arch 12.1, `SPARK=1`) build of the official vLLM Dockerfile
  with the overlay applied; `--push` for GHCR.
- `bench/` — long-decode corruption gate (v2), spec-decode gate (tools,
  non-repetition, K7 acceptance), vision gate, c8 concurrency proof.
- `Dockerfile.release-metadata` — OCI provenance labels (filled at publish).

### Validated runtime (RTX 5090 / SM120)
- NVFP4 target + NVFP4 draft + NVFP4 KV, DFlash2 K7, BF16 GDN/SSM state,
  8 GiB KV pin → 325,139-token pool, 262K context, max 4 concurrent seqs.
- Greedy determinism PASS; canary 437; NIAH at 184,024 tokens PASS;
  tools 10/10; JSON-schema structured output PASS; vision short/long
  probes PASS; c4 4,096-token soak 368.51 aggregate tok/s (all four
  streams completed, Running=4, Waiting=0); zero restarts, zero OOM.

### Not yet validated
- SM121/aarch64 native build (recipe ships; no GB10 hardware at release
  time). The multi-arch tag is withheld until the SM121 build passes the
  full correctness matrix natively.
