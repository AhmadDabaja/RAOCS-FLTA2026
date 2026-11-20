# Reproducing the experiments in the RAOCS paper

This guide maps **Tables I–V** to the implementation directories, model/dataset combinations, configuration changes, and plotting scripts. Use it with each method's `README.md` for a full description of **every** parameter in `config.py`. The paper's results were obtained with specified settings, which may differ from the repository's example defaults.

## General workflow

1. Install the dependencies from the [root README](../README.md#installation). First runs download torchvision training datasets into root `datasets/` unless already present.
2. Choose the table, dataset, method, and parameter sweep below. For complete historical configuration values, use the corresponding row in [`published_experiments.csv`](published_experiments.csv), selecting its `table`, `case`, and `method_directory` fields.
3. Edit that method's `config.py`: set dataset/model pair, number of clients, number of rounds, training parameters, deadline, RB budget, **all relevant wireless/energy/computation dictionary values**, and method-specific parameters. Set a **unique** `run_name` for each run.
4. Enter the method directory and run `python run_simulation.py`. A new run produces `results/<run_name>.json` (with `config` and `metrics`) and `results/<run_name>_final_model.pt`.
5. Return to the repository root. Edit the corresponding plotting script's active experiment filenames/paths to match the generated JSONs, then run the plotting script.

The CSV is a **configuration reference**, **not an executable sweep or a required input file**. It stores **49 table/case entries**, some reusing the same underlying historical run. `historical_result_path` records where original experiments were saved, sometimes under an archival folder belonging to another method; for a new run, use `method_directory` to identify which code to execute. The original JSON histories are not bundled here.

### Shared training and data setup

| Field | Typical published value | Meaning |
| --- | --- | --- |
| `num_partitions` | `50` | FL clients |
| `alpha` | `0.2` | Class-wise Dirichlet distribution across clients (non-IID) |
| `test_ratio` | `0.2` | Locally held-out client evaluation fraction |
| `local_epochs` | `16` | Local training epochs per participating client |
| `batch_size` | `64` | Client minibatch size |
| `learning_rate` | `0.01` | Client SGD learning rate |
| `seed` | `42` | Simulation seed |
| `p_min` | `0.1` where implemented | Floor of aggregation denominator; check individual experiment manifest |

| Dataset | `dataset_name` / `model_name` | Standard comparison rounds |
| --- | --- | --- |
| Fashion-MNIST | `fashionmnist` / `fcn-fashion` | 500 |
| CIFAR-10 | `cifar10` / `resnet8-cifar10` | 800 |
| CIFAR-100 | `cifar100` / `resnet20-cifar100` | 800 for published comparison; some historical run files have 1,200 rounds |

**How accuracy is evaluated:** the torchvision original **training split** is partitioned among clients; each client's local held-out share supplies evaluation results. It is **not** the untouched torchvision test set. Ensure model and dataset identifiers are changed as a pair.

## Table I — Baseline comparison

**Question:** how does resource-aware variance-based sampling compare with full participation, uniform/random sampling, norm-only OCS, and deterministic RB-aware alternatives?

| Scheme to run | Directory | Sampling rule / distinction |
| --- | --- | --- |
| FedAvg | [`FedAvg/`](../FedAvg/README.md) | All clients every round, data-weighted model averaging |
| UCS | [`UCS/`](../UCS/README.md) | Uniform fixed-count sampling without replacement; historical `sampling_fraction=0.1` |
| OCS | [`OCS/`](../OCS/README.md) | Bernoulli optimal sampling with expected `num_sampled_clients=5.0` |
| Greedy RB | [`RAGreedySampling/`](../RAGreedySampling/README.md) | Cheapest minimum-RB clients first |
| RAOCS-Greedy | [`RAOCSGreedy/`](../RAOCSGreedy/README.md) | Highest cached-norm/RB scores first; realized budget |
| RAOCS | [`RAOCS4WFL/`](../RAOCS4WFL/README.md) | Bernoulli probability optimized for expected RB budget |

| Dataset/model | Deadline `round_deadline_seconds` | `resource_block_budget` | Comparison rounds |
| --- | ---: | ---: | ---: |
| Fashion-MNIST / FCN | 5 s | 5 | 500 |
| CIFAR-10 / ResNet-8 | 1.5 s | 5 | 800 |
| CIFAR-100 / ResNet-20 | 4 s | 7 | 800 |

**Method-specific adjustments:** UCS uses `sampling_fraction=0.1`, `p_min=0.1`; OCS uses `num_sampled_clients=5`, `update_norm_mode="real"`, probe initialization; RAOCS uses `refresh_probability=1`; RAOCS-Greedy uses `update_norm_mode="real"` and the manifest's `use_empirical_aggregation_probability` choice. Set `resource_block_budget_std=0` in methods that support variable B. **Output:** accuracy and loss against simulated time and rounds. **Plot script:** `plot benchmarking.py` (change its active `EXPERIMENTS` list and `PLOTS_DIR`).

## Table II — Deadline sensitivity

**Question:** how does the available upload deadline τ affect selection feasibility, convergence, and accuracy?

| Method | Dataset/model | B | Sweep: `round_deadline_seconds` | Rounds |
| --- | --- | ---: | --- | ---: |
| RAOCS (`RAOCS4WFL/`) | CIFAR-10 / ResNet-8 | 5 | 0.2, 0.6, 1.5, 3, 6 seconds | 800 |
| RAOCS (`RAOCS4WFL/`) | CIFAR-100 / ResNet-20 | 7 | 0.5, 1, 2, 4, 8 seconds | 800 |

Run **one experiment for each deadline value**, keeping the remaining scenario settings from the manifest, and set `refresh_probability=1`, `resource_block_budget_std=0`. The standard RAOCS implementation treats τ as the effective upload window rather than explicitly simulating per-client computation time. **Plot script:** `plot sensitivity tua.py`; point its active `EXPERIMENTS` and `PLOTS_DIR` at the appropriate dataset's runs.

## Table III — Variable resource-block budgets

**Question:** how do stochastic per-round RB budgets affect probabilistic RAOCS versus a deterministic, norm-aware greedy scheduler?

| Methods | Dataset/model | Rounds | τ | Budget parameters |
| --- | --- | ---: | ---: | --- |
| `RAOCS4WFL/`, `RAOCSGreedy/` | CIFAR-10 / ResNet-8 | 800 | 1.5 s | `resource_block_budget=5`, `resource_block_budget_std` = 0, 1, 2, 4, `minimum_resource_block_budget=1` |

Run the four standard deviations **for each method**. The actual per-round budget is sampled around mean B=5 and clipped at the configured minimum. RAOCS uses an **expected** RB constraint, whereas the greedy method enforces the budget of each realized selection. RAOCS uses `refresh_probability=1`; RAOCS-Greedy still uses its `update_norm_mode` and configured aggregation policy. **Plot script:** `plot sensitivity B.py`; verify `RAOCS_RESULTS_DIR`, `GREEDY_RESULTS_DIR`, and the eight JSON names.

## Table IV — Energy-aware extension

**Question:** how do energy-capped sampling and norm refreshing affect time and cumulative energy required to reach a target accuracy?

| Cases (run all four per dataset) | Method folder | q value | Enforce energy cap? |
| --- | --- | ---: | --- |
| RAOCS, full norm refreshing | `RAOCS4WFL/` | 1 | No (measures energy only) |
| RAOCS, selected-only norm renewal | `RAOCS4WFL/` | 0 | No |
| REAOCS, full permitted norm refreshing | `REAOCS/` | 1 | Yes |
| REAOCS, selected-only norm renewal | `REAOCS/` | 0 | Yes |

| Dataset/model | `round_deadline_seconds` | `resource_block_budget` | Rounds | Accuracy targets |
| --- | ---: | ---: | ---: | --- |
| CIFAR-10 / ResNet-8 | 6 s | 10 | 800 | 75%, 78%, 80% |
| CIFAR-100 / ResNet-20 | 16 s | 30 | 800 | 30%, 35%, 38% |

Set `refresh_probability` in **both** method directories, with `resource_block_budget_std=0` for RAOCS. **Set the `energy_config` dictionary using the dataset-specific row of the CSV**, including in RAOCS, where these ranges are used for *measurement* rather than eligibility. `REAOCS` constrains expected energy `p(D+C+V)+(1−p)q(D+C)≤E`. **Plot script:** `plot energy.py`, with the relevant four result paths under its `EXPERIMENTS` setting.

**Numerical summary convention:** for each accuracy target, locate the **first round** whose **raw** `fed_eval_acc` reaches the target; calculate deadline-based elapsed time to that round and the cumulative logged `actual_energy_consumed_joules`. Do **not** smooth accuracy for threshold crossing.

## Table V — Computation-time variability

**Question:** what happens when clients finish local computation at different times and the server must decide when to start the shared uplink interval?

| Method | Dataset/model | Rounds | Deadline | Decision cutoff | B |
| --- | --- | ---: | ---: | ---: | ---: |
| `RAOCS-Tcomp/` | CIFAR-10 / ResNet-8 | 800 | 6 s total | 4.5 s | 5 |

In `computation_config` use `mean_computation_time_range_seconds=(3.5,3.5)`, `std_computation_time_range_seconds=(σ,σ)` for **σ = 0.25, 0.5, 1, 2, 4 seconds**, and `minimum_computation_time_seconds=0.1`. Set `update_norm_mode="real"`. Each available client computes **before** selection; selected clients upload already-computed updates within the remaining interval `τ−T_dec`. This is not the q-refresh algorithm. **Plot script:** `plot sensitivity t_comp.py`. Its checked-in `RESULTS_DIR` refers to old `RAOCSplus/results`; change it to `RAOCS-Tcomp/results` and point to the five actual JSON filenames (which may retain the `RAOCSplus(...)` naming convention).

## Reading results and generating figures

| Output | Where | How to interpret |
| --- | --- | --- |
| `results/<run_name>.json` | Method directory (when run there) | `config` plus `metrics` list containing per-round evaluation and selection information |
| `results/<run_name>_final_model.pt` | Method directory | Final PyTorch state dictionary |
| Plot images/summary files | Controlled by plotting script | Generated locally from chosen JSON histories; not shipped in the repository |

**Tables I, II, III, V (accuracy):** the plotting convention uses EMA `smoothed_t = 0.25 × raw_t + 0.75 × smoothed_(t−1)` and reports at the comparison round. The 0.75 historical-EMA weight is unrelated to Dirichlet `alpha=0.2`. When a recorded CIFAR-100 run contains 1,200 rounds but the table compares 800, use its **first 800** rounds. The CSV distinguishes `num_server_rounds` from `rounds_evaluated`.

**Plot scripts require manual path selection.** Execute each from the repository root **after** new JSONs exist:

```bash
python "plot benchmarking.py"       # Table I
python "plot sensitivity tua.py"     # Table II
python "plot sensitivity B.py"       # Table III
python "plot energy.py"              # Table IV
python "plot sensitivity t_comp.py" # Table V
```

## Differences between the historical runs and the current code

- Old RAOCS/REAOCS runs configured `update_norm_mode="newest"` or `"real"`; use `refresh_probability=0` or `1` respectively in the current versions. **OCS, RAOCS-Greedy and RAOCS-Tcomp retain their own older norm-mode fields.**
- Revised RAOCS/REAOCS energy logs reflect **actual operations**: selected `D+C+V`, refreshed-only `D+C`, otherwise 0. Previous `real` runs sometimes omitted energy for refreshed but uplink-ineligible clients. REAOCS now also prevents refreshing if `q(D+C)>E`; as a result **fresh-run energy totals or trajectories need not match old Table IV logs exactly**.
- Historical accuracy tables and Table IV quantities can be reconstructed from original archived result files, but **full paper-scale retraining of the revised q-based code has not been completed for every dataset**. Reproducibility should not be interpreted as bitwise identical results across PyTorch installations and hardware.
- A small rounding difference remains in one historical Table III RAOCS-Greedy entry at σ_B=1 (archive ~79.1455% versus reported 79.16%).

## Troubleshooting

- **No results / wrong path:** run `cd METHOD && python run_simulation.py`, not from the root; paths such as `results/` are relative to the working directory.
- **Download failure:** torchvision loads data with `download=True`; check internet access or place the expected dataset structure under root `datasets/`.
- **Wrong model:** set `dataset_name` and `model_name` to a supported pair before running.
- **Missing plot JSON:** plots use explicit file lists; align script paths with `run_name` and `results_dir` of your new outputs.
- **Slow runs:** 800-round FL experiments can be expensive; smoke-test with two rounds and fewer epochs first, but such a test is not a paper reproduction.
