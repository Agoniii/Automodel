---
name: nemo-automodel-perf-benchmarking
description: Measure and attribute training-throughput changes in NeMo AutoModel on a shared Slurm GPU cluster so that a reported gain survives review — paired A/B/A2 jobs, step-time and tok/s cross-checks, numerics gates, nsys window analysis, and the evidence a perf PR needs.
when_to_use: Evaluating a kernel, config, or parallelism change for throughput or memory; writing the Evidence section of a perf PR; deciding whether a measured difference is real; profiling a host-bound or GPU-bound training step; onboarding a new model or cluster for benchmarking.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - nemo-automodel
    - perf-benchmarking
---


# Performance benchmarking methodology for NeMo AutoModel

Distilled from the Qwen3.8-Flash-Next 180B SFT optimization on GB200 (September–October 2026, PRs #3963, #4005, #4006, #4230, #4231, #4232). The rules below cost real GPU hours to learn; apply them in order.

## 0. Ground rules

- **Fix the workload before touching anything.** Model config, sequence length, global batch, data source, GPU count per comparison. Change one thing per pair. Local batch size may move only when the user says so, and then it is the thing under test.
- **Benchmarks use a force-balanced MoE gate** (`fake_balanced_gate=true, fake_gate_noise=0.0`) and are labelled as such (`fb_` tag prefix). Measure "learned gate vs forced balance" once so the balance contribution is known; never ship force balance in a training config.
- **BF16 first.** Low-precision GEMM paths are a separate study; when GEMMs are a few percent of GPU time they are a net loss.
- **Develop at the small scale, confirm at the target scale.** Gains do not transfer in either direction (32 → 64 GPUs changed a 2.1x code gain into 1.3x once activation checkpointing was off).
- **Propose before executing.** Each round ends with a short list: variant, expected gain, GPU hours. Submit only what the owner approves. Never submit a configuration that is expected to OOM: a process killed mid-NVLink traffic produced Xid 137/145 on neighbouring nodes and drained a rack row.
- **Every result reports step time, tok/s/GPU and MFU together**, with the FLOPs/token formula named. PR bodies state step time and tok/s/GPU; MFU stays in internal notes unless the owner asks for it.

## 1. Environment check before any multi-node job

Containers lag `main`. Run a one-GPU check that prints versions and imports the benchmark entry point and the model class (`scripts/check_env.py <model.module:Class>`) after **every** baseline change. Symptoms this catches in two minutes instead of after an eight-node allocation: `ImportError: cannot import name 'ResolvedRevision' from 'huggingface_hub'` (hub too old), transformers refusing to import because `tokenizers` is too old, missing `pytest-timeout` for the repo conftest.

Override the container without rebuilding it: pure-Python packages (`transformers`, `huggingface_hub`, `pytest-timeout`) go into a `site/` directory via `pip install --no-deps --target`; binary wheels (`tokenizers`) are downloaded for the compute node's architecture (`--platform manylinux2014_aarch64 --python-version 3.12 --only-binary=:all:`) and **unzipped** into a separate `site_<arch>/` (pip refuses `--target` for a foreign-architecture wheel). `PYTHONPATH=site_<arch>:site` in every job template. Align versions with the repo's `uv.lock`, not with "latest".

Also check on a new cluster: `scontrol show partition` (GRES, exclusive nodes), `scontrol show topology` (NVLink domains), job-name conventions, whether the login node can run `enroot import` (usually not), and where colleagues already have the checkpoint.

## 2. Measurement design: the three-phase pair

Cross-job noise on the same configuration was 17–54%; node groups differ by 30%; a single job can run 25% slow for its whole duration or switch into a slow state at step 4 or step 11. Only **within-job ratios** are trustworthy.

A paired job (sketch below; adapt account, partition, container and paths) runs, on one allocation and one node set:

1. **A** – baseline, 30 steps;
2. **B** – variant, 30 steps;
3. **A2** – baseline again (`THREE_PHASE=1`).

Container is reused across phases (`--container-name`), the rendezvous port is derived from the job id (`20000 + jobid % 20000`) and a phase that dies before step 0 is retried once on another port. A and B may mount **different source trees** (`AUTOMODEL_SRC_A` / `AUTOMODEL_SRC_B`) for code-vs-code comparisons, or the same tree with different configs (`CONFIG_B`, `B_ARGS`) for knob comparisons.

The collector reads both phases' logs and reports per phase: step time from log timestamps over steps 10–29, tok/s/GPU from the recipe's `tps` over the same window, MFU, peak GiB, and the ratios B/A, A2/A, B/A2. It flags:

- **A2/A outside 0.90–1.10** → the pair is void; rerun.
- **STEP-CHANGE** → the mean of steps 1–9, 10–19 and 20–29 differ by more than 10% inside one phase. Inspect the per-step series: a fast warm-up segment is harmless if steps 10–29 are flat; a jump inside the measurement window voids the phase.

Cross-check the two throughput readings: `step_s × tok/s/GPU × world_size` must equal tokens per step. If it does not, the parser is wrong or the recipe's tps is scaled by a pipeline factor.

Read the ratio honestly: quote B/A and B/A2 both; when they disagree, report the range. A difference smaller than |A2/A − 1| is noise, whatever its sign.

## 3. Numerics gate

Every pair also compares per-step `loss` and `grad_norm` of A vs B **and** A vs A2 over all steps. The A/A2 spread is the noise floor for that job (bf16 multi-GPU reductions give loss ≈ 1e-4 relative, grad_norm 1e-3 to 1e-2); B counts as numerically clean when its spread against A is the same order. Do not hard-code a threshold: the floor moved by 5x between jobs.

Changes that must be bitwise (route selection, mask construction, replayed decisions) get a CPU unit test against a verbatim copy of the previous implementation as oracle, plus a one-GPU check with real dimensions. Kernel rewrites that are allclose rather than bitwise say so in the PR and show the tolerance used.

Before a PR, the fix or feature also needs the configuration that reviewers will try: a packed + context-parallel regression surfaced only because a reviewer ran CP=2 on two H100s. Add the two-GPU case as a `torchrun` functional test (`tests/functional_tests/<area>/run_*.py` + an `L2_*.sh` wrapper registered in the folder's `test_*.py`), not as an `mp.spawn` unit test: unit tests must stay CPU-only and `torch.compile` inside spawned pytest workers failed in CI.

## 4. Profiling: find out what the GPU is waiting for

Take one nsys timeline of every new variant: the benchmark entry (`python -m nemo_automodel.recipes.llm.benchmark`) calls `cudaProfilerStart/Stop` on `nsys_ranks` between `nsys_start` and `nsys_end`; only node 0 is wrapped in `nsys profile -c cudaProfilerApi -t cuda,nvtx,cublas --resolve-symbols=false` (no `osrt`: symbol download hangs on nodes without internet and the report is never written). Export `nsys stats` CSVs inside the container; the login node has no nsys. At 64 GPUs a single profiled rank can trip the HybridEP all-gather timeout, so take decisive timelines at 32.

Analyse the sqlite, not the `*_sum` tables, which accumulate over the whole process (3.2 s of `cuMemRelease` turned out to be teardown, not a per-layer cost):

- `scripts/nsys_window_stats.py <report.sqlite> iteration_13 <DecoderLayerClass>` – per microbatch: wall, GPU-busy union, kernel count, launch API count and time, synchronizations, memcpy/D2H, idle-gap histogram, kernel time by category.
- `scripts/nsys_kernel_attribution.py <report.sqlite> iteration_13_ga_step_0` – kernel time per innermost NVTX module, forward vs backward.
- NVTX ranges come from all local processes while CUDA activity comes from the profiled rank only: filter by the profiled pid and divide NVTX totals by the local process count (`Instances=4` is the tell).

Decide the regime from **GPU-busy fraction**. Below ~50% the step is host-bound: launch count (~1k launches per microbatch ≈ 1% of step time on GB200), host synchronizations and Python overhead decide everything, and GPU-side kernel fusion shows nothing in wall-clock. Above ~70% kernel work decides. The same optimization moved from +17.7% to +5% to −2% as the baseline became more host-bound; the effect is real GPU time (visible in GPU-busy ms) that only reaches wall-clock when the GPU is on the critical path.

What tends to pay in a host-bound MoE step: fewer launches per token (larger local batch or sequence, not more microbatches), removing per-layer host syncs (`.item()`, dispatcher count all-reduce), replaying deterministic forward decisions on checkpoint recompute (routes, dispatch layouts), building masks from data already on device instead of re-deriving them through `vmap`. What does not: faster GEMMs, Triton fusions of small elementwise chains (Triton launch ≈ 30–50 µs vs ≈ 10 µs for aten), per-op selective checkpointing when recompute replay already exists.

## 5. Ladder and final comparison

Build the optimization ladder cumulatively: each level measured as a pair on top of the previous levels, on the small scale. Then re-measure every level's PR **standalone on current `main`**; main moves, and a flag can silently stop being forwarded (a `checkpoint_moe_only` field was parsed but dropped by a new config-forwarding path; the pair showed identical memory on both sides, which is how it was caught — check that B's memory or kernel count actually changed when it should).

Final comparison at the target scale: baseline `main` vs full stack, each with the activation-checkpointing and local-batch settings that fit memory there. Report the chain of within-job ratios and the one absolute number per node group, never cross-job absolutes.

## 6. Operations

- `dist_env.timeout_minutes: 20` so a hang does not hold nodes for the default hour.
- Monitors must distinguish terminal states: Slurm state + step count in the log, and the first `[rank0]: XxxError` line; grepping for `Traceback` alone mislabels config errors as OOM.
- `sbatch --parsable` failures must not write empty job ids; a shell function used in an `&&` chain ignores `set -e`; `pytest … | tail` hides the pytest exit code unless `set -o pipefail`.
- Memory headroom is estimated (parameter state = params × (2+2+2+4+4) bytes / world, activations extrapolated linearly in tokens from measured points), not probed on the cluster.
- A worktree that a running job mounts must not be edited; create a new worktree per variant and remove them when the branch is merged or shelved.

## 7. What a perf PR carries

What / Changelog / Evidence / pre-checks, with the Evidence section holding: the numerics tests (bitwise or tolerance), the paired table (A, B, A2 rows with step time, tok/s/GPU, peak GiB), the ratios B/A and B/A2, the per-step loss and grad_norm spread versus the A/A2 floor, and one line of timeline evidence (what dropped: syncs, launches, GPU-busy ms). State the exact baseline commit and what sits on top of it. Keep the title within 80 characters (`type(scope): description`); the repository's import-linter contracts forbid `components.*` importing each other, so shared helpers go under `nemo_automodel/shared/`.

## Paired job sketch

```bash
#SBATCH -N 8 -t 01:40:00 ...            # one allocation, one node set, exclusive nodes
PORT=$((20000 + SLURM_JOB_ID % 20000))  # avoid a stale c10d store from the previous job
run_phase() {  # name config extra_args port src_tree
  srun --container-name=bench_$SLURM_JOB_ID --container-image=$CONT \
       --container-mounts=/data:/data,$5:/opt/Automodel --output=logs/${TAG}_$1.out --error=logs/${TAG}_$1.err \
       bash -c "cd /opt/Automodel && torchrun --nproc_per_node=$GPUS --nnodes=$SLURM_NNODES \
                --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$4 \
                -m nemo_automodel.cli.app $2 --step_scheduler.max_steps=30 $3"
}
run_phase A  "$CONFIG_A" "$A_ARGS" $PORT       "$SRC_A"
run_phase B  "$CONFIG_B" "$B_ARGS" $((PORT+1)) "$SRC_B"
run_phase A2 "$CONFIG_A" "$A_ARGS" $((PORT+2)) "$SRC_A"
```

Retry a phase once on another port if it dies before step 0; use a second container name when A and B mount different source trees. The collector parses `| step N |` lines for timestamps, `tps(x/gpu)`, `mem`, `loss` and `grad_norm`.

## Files

- `scripts/check_env.py` – container version report plus entry-point and model imports; run on one GPU before multi-node jobs.
- `scripts/nsys_window_stats.py` – per-microbatch host/GPU accounting from an nsys sqlite export (GPU-busy union, launches, syncs, idle gaps, kernel categories).
- `scripts/nsys_kernel_attribution.py` – kernel time per innermost NVTX module range for one microbatch, forward vs backward.
