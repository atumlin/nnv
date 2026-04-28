# NNV 3.0 Repeatability Package

This directory bundles the four experiments demonstrating NNV 3.0's new
capabilities — **FairNNV**, **ProbVer**, **GNNV**, and **VideoStar** — together
with a Dockerfile that builds a self-contained MATLAB R2024b environment and a
single `run_all.sh` driver that executes all four end-to-end.

## Prerequisites

Before the first `docker build`, the host must already have:

- **Docker** ≥ 24, with the daemon running
- **NVIDIA driver** ≥ 535 (CUDA 12+) on the host (only required for the GPU
  experiments — ProbVer, GNNV, VideoStar)
- **NVIDIA Container Toolkit** so `docker run --gpus all` works. On Windows,
  this means **WSL2** with at least one registered Linux distro (e.g. Ubuntu)
  and Docker Desktop's WSL Integration enabled
- **MATLAB licence** that the container can reach. Either a network licence
  server (`port@host`) or a node-locked licence file you can mount into the
  container. The Dockerfile takes a `LICENSE_SERVER` build arg; if you set
  it, it's baked in as `MLM_LICENSE_FILE`. Otherwise, supply
  `-e MLM_LICENSE_FILE=...` (or mount `network.lic`) at `docker run` time.
- **Disk**: ~14 GB for the image, plus ~1 GB for experiment artefacts.

Quick GPU sanity check:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If your GPU appears, `--gpus all` is wired up correctly.

## Build

From the **repository root** (so `.dockerignore` and the entire NNV checkout
are available to the build):

```bash
docker build \
    -t nnv3.0 \
    -f code/nnv/examples/NNV3.0/Dockerfile \
    --build-arg LICENSE_SERVER=<port>@<host> \
    .
```

Omit `--build-arg LICENSE_SERVER=...` if you'd rather provide the licence at
run time.

The build is roughly:

| Step                              | Wall-clock (RTX 5090, 1 Gbps link) |
|-----------------------------------|------------------------------------|
| Context copy + base image fetch   | ~1 min  (with `.dockerignore` — 5 min without) |
| `mpm install` MATLAB toolboxes    | ~3 min  |
| pip install Torch + CUDA wheels   | ~3 min  |
| `matlab -batch install.m`         | ~1 min  |
| Image export                      | ~3 min  |
| **Total**                         | **~10–12 min** |

## Run

```bash
docker run --gpus all -it nnv3.0
```

You'll land in `/home/matlab/nnv/code/nnv/examples/NNV3.0` with NNV already on
the MATLAB path.

### One-shot: everything

```bash
bash run_all.sh
```

Runs FairNNV, then ProbVer, then GNNV, then VideoStar, each in its own MATLAB
session. Per-experiment logs land in `repeatability_logs/`, and a final
`summary.csv` records wall-clock time and exit status per experiment.

Skip individual experiments with `NNV3_SKIP="probver videostar" bash run_all.sh`.

### Individual experiments

#### FairNNV

```bash
cd FairNNV
matlab -nodisplay -r "run('run_fm26_fairnnv.m'); exit()"
```

Outputs (in `FairNNV/fm26_fairnnv_results/`):
- `fm26_counterfactual_*.csv` — counterfactual fairness results
- `fm26_individual_*.csv` — individual fairness results across ε
- `fm26_timing_*.csv`, `fm26_timing_table.tex`
- `fm26_individual_fairness_combined.{png,pdf}`

#### ProbVer (requires `--gpus all`)

```bash
cd ProbVer
matlab -nodisplay -r "run('run_probver.m'); exit()"
```

The script verifies a random subset (default `numSamples = 3`) of the
TinyYOLO `yolo_2023` benchmark using the cp-star reachability method.
Results are written to `results_summary.csv` *incrementally* — one row
per instance, flushed before the next iteration begins, so a crash in
instance N preserves results for instances 1..N-1.

#### GNNV (uses GPU, ~5 min)

```bash
cd GNNV
matlab -nodisplay -r "run('run_gnn_experiments.m'); exit()"
```

Verifies voltage bounds on three GNN architectures (GCN, GINE, GINE+Edge)
across three perturbation levels and ten test scenarios (90 verifications
total). Outputs land in `GNNV/figures/` and `GNNV/results/`.

Options:
- `run_gnn_experiments('quiet')` — log to file, minimal stdout
- `run_gnn_experiments('no_figures')` — skip figure generation

#### VideoStar ZoomIn-4f (requires `--gpus all`)

```bash
cd VideoStar
matlab -nodisplay -r "run('run_zoomin_4f.m'); exit()"
```

Verifies the `zoomin_4f.onnx` video classifier on the first 10 ZoomIn test
samples across ε ∈ {1/255, 2/255, 3/255}. Configuration sits in the script's
top-level `config` struct: change `config.sampleIndices`, `config.verAlgorithm`
(`'relax'` or `'approx'`), or `config.timeout` as needed.

Results: `/tmp/results/VideoStar/ZoomIn/4/eps=*.csv`.

#### ModelStar

ModelStar's experiments live in `nnv/code/nnv/examples/Tutorial/NN/MNIST/weightPerturb`.
Run `run_expt_for_compute.m` to produce the data, then `EXPT.m` for the figure.

## Copy results out of the container

```bash
docker cp <container_id>:/home/matlab/nnv/code/nnv/examples/NNV3.0 ./nnv30_results
```

(`docker ps` to find the container ID.)

## Reference timings on this machine

The table below records wall-clock times measured on a Windows 11 host with an
RTX 5090 (32 GB VRAM, Blackwell, CC 12.0), driver 581.95, CUDA 13. MATLAB R2024b
runs inside the container; the same host has MATLAB R2025b natively but it
isn't used for these numbers.

| Experiment            | Wall-clock | Notes                                                          |
|-----------------------|-----------:|----------------------------------------------------------------|
| FairNNV               |     118 s  | 100 samples × 7 ε × 2 ONNX models. CPU only.                   |
| ProbVer (3 instances) |     ~7 min | TinyYOLO + cp-star reach. GPU. See `results_summary.csv`.      |
| GNNV (90 verifs)      |     239 s  | 10 scenarios × 3 models × 3 ε. GPU. README baseline was ~5 min.|
| VideoStar ZoomIn-4f   |   ~12.6 min| 10 samples × 3 ε with `relax` algorithm. GPU.                  |
| **End-to-end suite**  | **~22 min**| All four via `run_all.sh`, RTX 5090.                           |

These numbers are intended as a baseline. Expect roughly 1.5–3× longer on a
mid-range workstation GPU (RTX 4070 / A4000) and 5–10× longer on CPU-only hosts.

## Troubleshooting

**`License Manager Error` on first `matlab` invocation.** The build doesn't
validate the licence; it's consumed at first MATLAB run. Pass
`--build-arg LICENSE_SERVER=port@host` at build time, or
`-e MLM_LICENSE_FILE=port@host` at run time, or mount a `network.lic`.

**`GPU device is not supported because it has a higher compute capability...`**
RTX 50-series (Blackwell, CC 12.0) is too new for the CUDA libraries shipped
with MATLAB R2024b. The experiment scripts call
`parallel.gpu.enableCUDAForwardCompatibility(true)` automatically. If you
are running a different MATLAB version that lacks this API, upgrade to
MATLAB R2025a+ (CUDA 12+) or R2025b (CUDA 13).

**`Undefined function 'load_vnnlib'` from ProbVer.** The script auto-bootstraps
the NNV path from its own location, so this should not happen if launched
from the `ProbVer/` directory. If it does, run
`addpath(genpath('/home/matlab/nnv/code/nnv'))` first.

**`tbxmanager.com` URL fetch error during `install.m`.** That mirror is
sometimes unreachable. The build catches it; modern NNV3.0 examples do not
require MPT3, so the install completes anyway. Run `check_nnv_setup` to
confirm core NNV is healthy before invoking experiments.

**`npy-matlab directory not found`.** Pull the latest image — the Dockerfile
now clones `kwikteam/npy-matlab` during the build. For pre-built images,
manually clone it into
`code/nnv/examples/Submission/FORMALISE2025/npy-matlab/`.
