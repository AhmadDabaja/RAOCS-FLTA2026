# REAOCS — energy-constrained resource-aware sampling

**REAOCS: energy-constrained RAOCS.** Uses the same cached-norm, post-selection q-refreshing process and expected RB constraint as RAOCS. Adds per-client **expected energy** limits by capping each Bernoulli sampling probability. Ineligible clients can be prevented from selection or norm refreshing by energy rules. Wireless uplink ineligibility does not by itself prevent local norm refresh. The energy logger records actual operations, independently of whether the realized round energy exceeds the expected threshold.

## How the method works

**Selection:** independent Bernoulli draws with water-filling `p_k ≤ p̄_k`; the same nonempty-draw fallback applies if positive probabilities yield no selected client.

**Aggregation:** same inverse-probability update formula as RAOCS, using the **energy-capped** probabilities `p_k` and `max(p_k,p_min)` only in the denominator; refreshing-only clients do not contribute model updates.

Let `L_k=D_k+C_k`, `V_k` be upload energy, and `E_k` the expected energy budget. The code constrains `p_k(L_k+V_k)+(1−p_k)qL_k ≤ E_k`, with cap `p̄_k = min(1,(E_k−qL_k)/((1−q)L_k+V_k))` when feasible. Clients with `qL_k>E_k` have `p_k=0` and are excluded from norm refreshing; uplink-ineligible clients may still refresh if `qL_k≤E_k`. `E_k` is **not** a simulated battery level. Selected energy is `D+C+V`; refreshed-only energy `D+C`; otherwise `0`. All refreshing uses norms available in the next round.

The model parameters received from selected clients are converted into local differences `Δ_k = w_k−w`; only the specified selected clients contribute to the global update. Federated evaluation is calculated from the held-out examples in client partitions.

## Run this method

From the repository root, after installing the [project dependencies](../README.md#installation):

```bash
cd REAOCS
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
| `run_name` | `REAOCS(ResNet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42)` | Output filename stem only; **does not** configure the dataset, rounds, or RBs. Choose a unique value for each run. |
| `num_server_rounds` | `800` | Maximum federated communication rounds (positive integer). |
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
| `round_deadline_seconds` | `1.5` | Round/upload deadline τ, in **seconds**; standard methods use it as the effective uplink window. |
| `resource_block_budget` | `5` | Number of available wireless resource blocks B; resource-aware schedulers use it as a constraint. |
| `p_min` | `0.1` | Positive floor used in the **aggregation denominator**, `max(p_k, p_min)`; not a lower bound imposed on the sampling lottery. |
| `probe_update_norm_initialization` | `True` | When true, initialize cached update norms with a pre-round local-computation probe where applicable. |
| `refresh_probability` | `0` | Independent refresh probability q for each eligible non-selected client (0–1); selected clients always renew their cached norm. |

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

These client-specific ranges are sampled at physical-system initialization. They **constrain** sampling and refreshing in REAOCS. The table below shows the current CIFAR-10 defaults; the dataset-specific ranges are:

| Dataset/model | E budget (J) | Download D (J) | Computation C (J) | Upload V (J) |
| --- | --- | --- | --- | --- |
| Fashion-MNIST / FCN | 11.25–13.75 | 1.125–1.375 | 9–11 | 2.5–7.5 |
| CIFAR-10 / ResNet-8 | 45–55 | 4.5–5.5 | 36–44 | 10–30 |
| CIFAR-100 / ResNet-20 | 180–220 | 18–22 | 140–180 | 40–120 |

| Parameter | Current default | Meaning / effect |
| --- | --- | --- |
| `energy_budget_range_joules` | `(45.0, 55.0)` | Range for sampling each client’s expected energy budget E_k (J). This is **not** an explicit battery-charge simulator. |
| `download_energy_range_joules` | `(4.5, 5.5)` | Range for model download energy D_k per local training event (J). |
| `computation_energy_range_joules` | `(36.0, 44.0)` | Range for local training energy C_k per computed update (J). |
| `upload_energy_range_joules` | `(10.0, 30.0)` | Range for selected-client update upload energy V_k (J). |

## Reproduce the paper experiments

**Relevant experiments:** Table IV (energy-capped configurations): CIFAR-10/ResNet-8 (B=10, τ=6 s), CIFAR-100/ResNet-20 (B=30, τ=16 s), each at q=0 and q=1. Set `energy_config` to the **dataset-specific historical ranges** listed in the experiment manifest.

For exact dataset-specific parameters—including `energy_config` and all wireless ranges—see the **[Table I–V reproduction guide](../docs/REPRODUCING_PAPER.md)** and the optional **[historical configuration manifest](../docs/published_experiments.csv)**. The manifest is a record of parameters, **not executable code**. A plot script at the repository root must be pointed at the JSON filenames produced by your runs.

## Practical notes

- All method simulations run independently; run `python run_simulation.py` **from this directory** so relative `results/` paths are predictable.
- The script writes a JSON history and final `.pt` checkpoint. Outputs and datasets are intentionally excluded from Git.
- A 2-round smoke test checks execution, not whether paper-scale accuracy is reproduced.
