# OCS — optimal client sampling

**Optimal Client Sampling (OCS) baseline.** The server estimates each client’s update norm and assigns independent Bernoulli probabilities according to `p_k ∝ α_k × cached_norm_k`, capped at 1, subject to the **expected client-count** constraint `Σ_k p_k = num_sampled_clients`. It does **not** optimize the wireless RB budget as RAOCS does. A fallback selects the highest-probability client if a Bernoulli draw returns an empty set.

## How the method works

**Selection:** independent Bernoulli draws with capped OCS probabilities; if all draws fail, the code force-selects `argmax(p_k)`.

**Aggregation:** `w_next = w + Σ_selected [α_k / max(p_k, p_min)] (w_k−w)`; `p_k` is the computed OCS probability.

**Update-norm information:** the `update_norm_mode` field remains in this baseline (it is **not** replaced by `refresh_probability`). `real` queries unselected clients for current norms after sampling, `newest` retains their stored norms, and `holt` applies the legacy predictor; the current round's selection uses the norms already available. The code optionally probes before the first round.

The model parameters received from selected clients are converted into local differences `Δ_k = w_k−w`; only the specified selected clients contribute to the global update. Federated evaluation is calculated from the held-out examples in client partitions.

## Run this method

From the repository root, after installing the [project dependencies](../README.md#installation):

```bash
cd OCS
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
| `run_name` | `OCS(FCN-MNIST, 5_rounds, 5_clients,  5_RBs, 5_sec, seed_42)` | Output filename stem only; **does not** configure the dataset, rounds, or RBs. Choose a unique value for each run. |
| `num_server_rounds` | `5` | Maximum federated communication rounds (positive integer). |
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
| `num_sampled_clients` | `4.4` | OCS target **expected** number of selected clients m: `sum_k p_k = m` before Bernoulli sampling. |
| `p_min` | `0.1` | Positive floor used in the **aggregation denominator**, `max(p_k, p_min)`; not a lower bound imposed on the sampling lottery. |
| `probe_update_norm_initialization` | `True` | When true, initialize cached update norms with a pre-round local-computation probe where applicable. |
| `update_norm_mode` | `real` | Legacy norm-information policy for this method: `real` (query current norms), `newest` (reuse cached norms), or `holt` (trend prediction); see method description for timing. |
| `holt_rho` | `0.5` | Holt predictor discount/smoothing parameter, active only in `holt` mode. *(only when `update_norm_mode == 'holt'`)* |
| `holt_beta` | `0.3` | Holt level-update coefficient, active only in `holt` mode. *(only when `update_norm_mode == 'holt'`)* |
| `holt_gamma` | `0.9` | Holt trend-update coefficient, active only in `holt` mode. *(only when `update_norm_mode == 'holt'`)* |
| `holt_epsilon` | `1e-12` | Small positive offset for numerical stability in logarithmic Holt norm prediction. *(only when `update_norm_mode == 'holt'`)* |
| `round_deadline_seconds` | `5` | Round/upload deadline τ, in **seconds**; standard methods use it as the effective uplink window. |
| `resource_block_budget` | `5` | Number of available wireless resource blocks B; resource-aware schedulers use it as a constraint. |

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

## Reproduce the paper experiments

**Relevant experiments:** Table I: the same three dataset/model scenarios as FedAvg. Set `num_sampled_clients=5.0`, `update_norm_mode="real"`, `p_min=0.1`, and enable initial norm probing.

For exact dataset-specific parameters—including `energy_config` and all wireless ranges—see the **[Table I–V reproduction guide](../docs/REPRODUCING_PAPER.md)** and the optional **[historical configuration manifest](../docs/published_experiments.csv)**. The manifest is a record of parameters, **not executable code**. A plot script at the repository root must be pointed at the JSON filenames produced by your runs.

## Practical notes

- All method simulations run independently; run `python run_simulation.py` **from this directory** so relative `results/` paths are predictable.
- The script writes a JSON history and final `.pt` checkpoint. Outputs and datasets are intentionally excluded from Git.
- A 2-round smoke test checks execution, not whether paper-scale accuracy is reproduced.
