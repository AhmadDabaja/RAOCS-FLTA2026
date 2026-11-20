# FedAvg — full participation

**Full-participation FedAvg baseline.** Every client receives the round model, trains locally, and sends its model parameters. The server uses the number of training examples to weight the local models. There is no randomized client subsampling, cached-norm scoring, or energy-cap optimizer. The code computes an equal-share RB quantity for reporting, **not** RAOCS feasibility-based scheduling.

## How the method works

**Selection:** all K clients every round (`p_k=1`). No wireless exclusion at this selection stage.

**Aggregation:** weighted average of the K returned local models: `w_next = Σ_k (n_k/Σ_j n_j) w_k`. No inverse-probability correction.

The configured `round_deadline_seconds`/`resource_block_budget` exist for physical-system comparison and reporting. FedAvg does not apply RAOCS's deadline-based eligibility or RB water-filling during selection.

The model parameters received from selected clients are converted into local differences `Δ_k = w_k−w`; only the specified selected clients contribute to the global update. Federated evaluation is calculated from the held-out examples in client partitions.

## Run this method

From the repository root, after installing the [project dependencies](../README.md#installation):

```bash
cd FedAvg
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
| `run_name` | `FedAvg(ResNet20-CIFAR100, 100_rounds, 7_RBs, 4_sec, seed_42)` | Output filename stem only; **does not** configure the dataset, rounds, or RBs. Choose a unique value for each run. |
| `num_server_rounds` | `100` | Maximum federated communication rounds (positive integer). |
| `communication_budget` | `None` | Optional cumulative communication limit in **bytes**; `None` disables this early-stop condition. |
| `learning_rate` | `0.01` | Local SGD learning rate used in client optimization (positive). |
| `batch_size` | `64` | Local mini-batch size in training (positive integer). |
| `local_epochs` | `16` | Local passes through each participating client’s training data per local update (positive integer). |
| `dataset_name` | `cifar100` | Torchvision dataset: `fashionmnist`, `cifar10`, or `cifar100`. |
| `model_name` | `resnet20-cifar100` | Architecture identifier. Supported pairs: `fashionmnist/fcn-fashion`, `cifar10/fcn-cifar10`, `cifar10/resnet8-cifar10`, `cifar100/resnet20-cifar100`. |
| `num_partitions` | `50` | Number of FL clients / dataset partitions, denoted K. |
| `alpha` | `0.2` | Class-wise Dirichlet concentration controlling non-IID data allocation; smaller values make partitions more heterogeneous. |
| `test_ratio` | `0.2` | Fraction of each client’s locally assigned examples reserved for evaluation (0–1). |
| `seed` | `42` | Seed for Python, NumPy and PyTorch random generators in simulation setup. |
| `round_deadline_seconds` | `4` | Round/upload deadline τ, in **seconds**; standard methods use it as the effective uplink window. |
| `resource_block_budget` | `7` | Number of available wireless resource blocks B; resource-aware schedulers use it as a constraint. |

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

**Relevant experiments:** Table I: Fashion-MNIST/FCN (500 comparison rounds; B=5, τ=5 s); CIFAR-10/ResNet-8 (800, B=5, τ=1.5 s); CIFAR-100/ResNet-20 (800, B=7, τ=4 s). Some archived CIFAR-100 runs contain 1,200 rounds but the table uses the first 800.

For exact dataset-specific parameters—including `energy_config` and all wireless ranges—see the **[Table I–V reproduction guide](../docs/REPRODUCING_PAPER.md)** and the optional **[historical configuration manifest](../docs/published_experiments.csv)**. The manifest is a record of parameters, **not executable code**. A plot script at the repository root must be pointed at the JSON filenames produced by your runs.

## Practical notes

- All method simulations run independently; run `python run_simulation.py` **from this directory** so relative `results/` paths are predictable.
- The script writes a JSON history and final `.pt` checkpoint. Outputs and datasets are intentionally excluded from Git.
- A 2-round smoke test checks execution, not whether paper-scale accuracy is reproduced.
