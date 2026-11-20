# RAOCS — Resource-Aware Optimal Client Sampling for Wireless Federated Learning

This repository provides the Python simulation code for **“Joint Client Scheduling and Resource Allocation for Optimal Federated Learning over Wireless Networks”**, by **Ahmad Dabaja and Rachid El-Azouzi**, accepted at **IEEE FLTA 2026**.

In wireless federated learning, participating clients have unequal update importance and unequal wireless upload costs. **RAOCS** uses cached local-update norms and wireless resource-block (RB) requirements to assign variance-aware **Bernoulli client-sampling probabilities** under an expected RB budget. The repository also includes an energy-constrained extension (**REAOCS**), deterministic schedulers, conventional FL baselines, and a separate computation-time sensitivity experiment. This is a **simulation**, not a deployment stack for physical devices.

## Implemented methods

| Directory | Selection policy | Aggregation | Paper experiments |
| --- | --- | --- | --- |
| [RAOCS4WFL](RAOCS4WFL/README.md) | Bernoulli, `p_k` from cached-norm / RB-cost water-filling (`Σ p_k b_k,min ≤ B`) | Inverse-probability-weighted updates | I–IV |
| [REAOCS](REAOCS/README.md) | RAOCS with client-specific expected energy caps | Inverse-probability-weighted updates | IV |
| [RAOCSGreedy](RAOCSGreedy/README.md) | Deterministic greedy ranking by importance per square-root RB cost | Empirical-frequency or water-filling denominator | I, III |
| [RAGreedySampling](RAGreedySampling/README.md) | Deterministic greedy ranking by *lowest* RB cost | Empirical-frequency correction | I |
| [RAOCS-Tcomp](RAOCS-Tcomp/README.md) | RAOCS-style sampling after variable-time precomputation and decision cutoff | Inverse-probability-weighted updates | V |
| [FedAvg](FedAvg/README.md) | All clients participate | Data-size-weighted model average | I |
| [UCS](UCS/README.md) | Fixed-size uniform sample without replacement | Inverse-probability-weighted updates | I |
| [OCS](OCS/README.md) | Bernoulli, variance-aware probabilities under expected client count | Inverse-probability-weighted updates | I |

**For algorithm details and every `config.py` parameter**, open the method's linked README. Each method has its own `config.py`, `run_simulation.py`, and `src/`. No special runner or command-line flags are required: edit the configuration and run that method.

## Installation

- **Python 3.10+** (CPU-only is supported; GPUs can accelerate larger runs).
- PyTorch and torchvision must be installed as a **compatible pair**. The [PyTorch installation guide](https://pytorch.org/get-started/locally/) describes CUDA/ROCm-specific wheels.
- An Internet connection is needed to obtain torchvision datasets on first use unless they are already cached.

```bash
# In the repository root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start: RAOCS

1. Open `RAOCS4WFL/config.py`. For a first, short run, set `num_server_rounds = 2`, `local_epochs = 1`, and `run_name = "raocs_quickcheck"`. Confirm `dataset_name` and `model_name` form a supported pair.
2. Start the simulation from the method directory:

```bash
cd RAOCS4WFL
python run_simulation.py
```

3. Inspect `RAOCS4WFL/results/raocs_quickcheck.json` and `RAOCS4WFL/results/raocs_quickcheck_final_model.pt` (relative to repository root). Return to the root with `cd ..` before running plot scripts.

The code uses the torchvision **training split**, allocates its examples among simulated clients by a class-wise Dirichlet distribution (`alpha`), then holds out a `test_ratio` proportion of each client's partition for evaluation. The downloaded data are saved under the **repository root `datasets/`** directory. Outputs contain `config` and per-round `metrics`, including selected clients and, for RAOCS/REAOCS, refreshed clients and energy usage. Generated output paths are ignored by Git.

**Important:** settings come from fields in `config.py`, **not the text in `run_name`**. Some provided defaults are diagnostic or exploratory and are not the paper's published configurations.

## Sampling, cached norms, and refreshing

Let `α_k=n_k/Σ_j n_j` be a data weight, `g_k` the cached norm of a local update, and `b_k,min` the RBs required to upload before the effective deadline. In RAOCS the non-saturated sampling rule has the structure `p_k ∝ α_k g_k / sqrt(b_k,min)`, subject to `Σ_k p_k b_k,min ≤ B` and `p_k ≤ 1`. Sampling is independent Bernoulli; consequently, a particular round may use more or fewer than the **expected** RB budget.

At the start of round `t`, the server computes `p_k` from **previously cached** norm information. Selected clients train from the current global model and upload a fresh update. Each non-selected client may recompute its norm with probability `refresh_probability = q`, without contributing a model update. That norm is used **from the next round**. The selected-client update is aggregated with the code's denominator `max(p_k,p_min)`; `p_min` is a numerical floor **in aggregation**, not a modification of the Bernoulli draw.

- `q=0`: only selected clients update their cached norm.
- `q=1`: RAOCS refreshes all non-selected clients; REAOCS refreshes those passing its expected-energy feasibility check.
- `0<q<1`: eligible non-selected clients refresh independently using a dedicated RNG.

**REAOCS** adds expected-energy caps using download, local computation, and upload energy; **RAOCS** logs energy without applying an energy cap. **RAOCS-Tcomp** uses a different **pre-selection** computation-time protocol, not the `q=1` mode.

## Reproduce the experiments and figures

See **[Reproducing Tables I–V](docs/REPRODUCING_PAPER.md)**. It gives each table's datasets, methods, sweeps, parameters, output interpretation, and relevant plot script. The optional **[configuration manifest](docs/published_experiments.csv)** lists 49 historical table/case configurations (including detailed wireless and energy ranges); it is an archival parameter reference, not a file the simulator reads automatically.

Plot source scripts are at the repository root:

```bash
python "plot benchmarking.py"
python "plot sensitivity tua.py"
python "plot sensitivity B.py"
python "plot energy.py"
python "plot sensitivity t_comp.py"
```

Run them **from the root after producing the referenced JSON files**. These scripts reference specific historical result filenames and directories; review their `EXPERIMENTS` and path settings, and update them to match `run_name`. Large original result JSONs, figures and trained checkpoints are not shipped in the repository.

## Citation

If you use the code, cite: **Ahmad Dabaja and Rachid El-Azouzi**, “Joint Client Scheduling and Resource Allocation for Optimal Federated Learning over Wireless Networks,” **IEEE FLTA 2026**.

For implementation-specific differences from the historical conference logs (notably corrected energy accounting), see the [reproducibility notes](docs/REPRODUCING_PAPER.md#differences-between-the-historical-runs-and-the-current-code).
