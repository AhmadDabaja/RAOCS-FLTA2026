# RAOCS-Tcomp — computation-time variability

**RAOCS computation-time sensitivity experiment (Table V).** This is a separate **pre-selection computation protocol**: clients have heterogeneous stochastic computation times; the server chooses a decision time from the deadline and cutoff, considers the clients that finish before that decision time, and—in the published `real` mode—uses their **current freshly computed** update norms for Bernoulli RAOCS sampling. Only selected precomputed updates are uploaded. It is not the post-selection q-refreshing mechanism of RAOCS/REAOCS.

## How the method works

**Selection:** identify clients that completed local precomputation before the decision time, apply wireless feasibility in the remaining upload window, then sample with RAOCS-style probabilities.

**Aggregation:** only selected, already-computed local updates enter `w_next = w + Σ_selected [α_k / max(p_k, p_min)] (w_k−w)`.

**Published `real` protocol:** client computation times are sampled from client-specific Gaussian distributions; `T_dec=min(computation_cutoff_seconds, max_k T_k,comp)`. A client with `T_k,comp≤T_dec` is available; uploading must fit in `τ−T_dec` using `b_k,min=Z/(ν_k(τ−T_dec))`. Available clients compute and report their **current** norms before selection, and selected clients use those precomputed updates (not trained twice). The legacy `update_norm_mode` parameter still exists here, but the published Table V used `real`. This experiment is not implemented by setting `q=1` in `RAOCS4WFL`.

The model parameters received from selected clients are converted into local differences `Δ_k = w_k−w`; only the specified selected clients contribute to the global update. Federated evaluation is calculated from the held-out examples in client partitions.

## Run this method

From the repository root, after installing the [project dependencies](../README.md#installation):

```bash
cd RAOCS-Tcomp
# Edit config.py to select the dataset, model, rounds and an unused run_name.
python run_simulation.py
```

For a short diagnostic run, change `num_server_rounds` to `2`, `local_epochs` to `1`, and assign a unique `run_name`. Model/data loading may download the chosen torchvision training dataset into the root `datasets/` folder on first use. Outputs are written to `results/<run_name>.json` and `results/<run_name>_final_model.pt` relative to this directory. JSON has `config` and per-round `metrics` fields. No new command-line runner is needed.

## `config.py` parameter reference

The values below are **current source defaults**, not necessarily the published experimental settings. Edit the dataclass fields in `SimulationConfig`; `run_name` is not parsed to choose experimental parameters.

### Training, data, selection, and output settings

| Parameter | Current default | Meaning / effect |
| --- | --- | --- |
| `results_dir` | `results` | Directory for JSON histories and model checkpoints, resolved from the working directory. |
| `run_name` | `RAOCS4WFL(FCN-MNIST, 30_rounds, 5_RBs, 5_sec, seed_42)` | Output filename stem only; **does not** configure the dataset, rounds, or RBs. Choose a unique value for each run. |
| `num_server_rounds` | `30` | Maximum federated communication rounds (positive integer). |
| `communication_budget` | `None` | Optional cumulative communication limit in **bytes**; `None` disables this early-stop condition. |
| `learning_rate` | `0.01` | Local SGD learning rate used in client optimization (positive). |
| `batch_size` | `64` | Local mini-batch size in training (positive integer). |
| `local_epochs` | `16` | Local passes through each participating client’s training data per local update (positive integer). |
| `dataset_name` | `cifar10` | Torchvision dataset: `fashionmnist`, `cifar10`, or `cifar100`. |
| `model_name` | `resnet8-cifar10` | Architecture identifier. Supported pairs: `fashionmnist/fcn-fashion`, `cifar10/fcn-cifar10`, `cifar10/resnet8-cifar10`, `cifar100/resnet20-cifar100`. |
| `num_partitions` | `50` | Number of FL clients / dataset partitions, denoted K. |
| `alpha` | `0.2` | Class-wise Dirichlet concentration controlling non-IID data allocation; smaller values make partitions more heterogeneous. |
| `test_ratio` | `0.2` | Fraction of each client’s locally assigned examples reserved for evaluation (0–1). |
| `seed` | `42` | Seed for Python, NumPy and PyTorch random generators in simulation setup. |
| `round_deadline_seconds` | `6` | Round/upload deadline τ, in **seconds**; standard methods use it as the effective uplink window. |
| `resource_block_budget` | `5` | Number of available wireless resource blocks B; resource-aware schedulers use it as a constraint. |
| `computation_cutoff_seconds` | `4.5` | Maximum server waiting time c (seconds) for pre-selection computation; must be strictly below the total round deadline. |
| `p_min` | `0.1` | Positive floor used in the **aggregation denominator**, `max(p_k, p_min)`; not a lower bound imposed on the sampling lottery. |
| `probe_update_norm_initialization` | `True` | When true, initialize cached update norms with a pre-round local-computation probe where applicable. |
| `update_norm_mode` | `real` | Legacy norm-information policy for this method: `real` (query current norms), `newest` (reuse cached norms), or `holt` (trend prediction); see method description for timing. |
| `holt_rho` | `0.5` | Holt predictor discount/smoothing parameter, active only in `holt` mode. *(only when `update_norm_mode == 'holt'`)* |
| `holt_beta` | `0.3` | Holt level-update coefficient, active only in `holt` mode. *(only when `update_norm_mode == 'holt'`)* |
| `holt_gamma` | `0.9` | Holt trend-update coefficient, active only in `holt` mode. *(only when `update_norm_mode == 'holt'`)* |
| `holt_epsilon` | `1e-12` | Small positive offset for numerical stability in logarithmic Holt norm prediction. *(only when `update_norm_mode == 'holt'`)* |

### `wireless_config` (physical wireless channel)

These keys are inside the `wireless_config` dictionary and are passed to `PhysicalSystem`; they affect the simulated channel and resource needs.

| Parameter | Current default | Meaning / effect |
| --- | --- | --- |
| `cell_radius` | `500` | Cell coverage radius (metres) for sampled client positions. |
| `rb_bandwidth_hz` | `1e+06` | Bandwidth of one radio resource block (Hz). |
| `carrier_frequency_hz` | `3.5e+09` | Carrier frequency for propagation/path-loss calculations (Hz). |
| `delta_t` | `1` | Physical-system simulation time step between channel/mobility updates (seconds). |
| `rate_efficiency` | `0.65` | Fractional efficiency factor applied in the achievable-rate calculation (0–1). |
| `path_loss_exponent` | `3.2` | Distance-decay exponent in the path-loss model. |
| `shadowing_std_db` | `8` | Standard deviation of log-normal shadowing, in dB. |
| `shadowing_corr_distance_m` | `30` | Spatial correlation distance for shadowing, in metres. |
| `mean_speed_range_mps` | `(0.2, 3.0)` | Range for sampling clients’ mean mobility speeds (metres/second). |
| `mean_tx_power_range_dbm` | `(15.0, 23.0)` | Range for sampling clients’ mean uplink transmit power (dBm). |
| `noise_figure_range_db` | `(5.0, 10.0)` | Range for the per-client receiver noise figure (dB). |

### `energy_config` (energy model)

These ranges provide energy-related diagnostics for the computation-time experiment. They do not impose REAOCS-style probability caps here.

The source declares different `energy_config` defaults conditional on the dataset **at class definition time**. If you later switch `dataset_name`, explicitly check/update `energy_config` too; changing just the dataset field does not automatically regenerate these ranges. The values below are for the current CIFAR-10 class definition.

| Parameter | Current default | Meaning / effect |
| --- | --- | --- |
| `energy_budget_range_joules` | `(180.0, 220.0)` | Range for sampling each client’s expected energy budget E_k (J). This is **not** an explicit battery-charge simulator. |
| `download_energy_range_joules` | `(18.0, 22.0)` | Range for model download energy D_k per local training event (J). |
| `computation_energy_range_joules` | `(140.0, 180.0)` | Range for local training energy C_k per computed update (J). |
| `upload_energy_range_joules` | `(40.0, 120.0)` | Range for selected-client update upload energy V_k (J). |

### `computation_config` (computational heterogeneity)

Each client receives a mean and a standard deviation from these ranges; computation time is sampled again each round. This is independent of the radio-channel speed.

| Parameter | Current default | Meaning / effect |
| --- | --- | --- |
| `mean_computation_time_range_seconds` | `(3.5, 3.5)` | Range from which each client’s mean compute time μ_k is sampled (seconds). |
| `std_computation_time_range_seconds` | `(0.5, 0.5)` | Range from which each client’s per-round compute-time standard deviation σ_k is sampled (seconds). |
| `minimum_computation_time_seconds` | `0.1` | Lower truncation of sampled computation time (seconds). |

## Reproduce the paper experiments

**Relevant experiments:** Table V: CIFAR-10/ResNet-8, B=5, 800 rounds, τ=6 s, cutoff c=4.5 s; client mean computation time 3.5 s with standard deviations {0.25,0.5,1,2,4} s.

For exact dataset-specific parameters—including `energy_config` and all wireless ranges—see the **[Table I–V reproduction guide](../docs/REPRODUCING_PAPER.md)** and the optional **[historical configuration manifest](../docs/published_experiments.csv)**. The manifest is a record of parameters, **not executable code**. A plot script at the repository root must be pointed at the JSON filenames produced by your runs.

## Practical notes

- All method simulations run independently; run `python run_simulation.py` **from this directory** so relative `results/` paths are predictable.
- The script writes a JSON history and final `.pt` checkpoint. Outputs and datasets are intentionally excluded from Git.
- A 2-round smoke test checks execution, not whether paper-scale accuracy is reproduced.
