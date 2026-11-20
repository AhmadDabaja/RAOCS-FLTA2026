# RAOCS4WFL — resource-aware optimal client sampling

**RAOCS: resource-aware optimal client sampling.** Cached norms from the start of each round define scores `α_k × g_k / sqrt(b_k,min)`. A capped water-filling rule produces Bernoulli probabilities under the **expected** RB constraint `Σ_k p_k b_k,min ≤ B`; realized RB use can fluctuate. Selected clients train and upload fresh updates. Non-selected clients independently refresh cached norms with probability q, for use in the **next** round. This implementation **measures** energy but imposes **no energy-based selection or refresh restriction**.

## How the method works

**Selection:** independent Bernoulli draws with water-filling `p_k`; if all p are zero, no client is selected; if a nonzero probability vector happens to yield none, select `argmax(p_k)`.

**Aggregation:** only selected fresh updates enter `w_next = w + Σ_selected [α_k / max(p_k, p_min)] (w_k−w)`. Refreshed-only clients never upload an FL update. The denominator floor is an implementation safeguard and does not alter the actual Bernoulli probabilities.

**Refreshing:** `q=0` keeps non-selected norms unchanged; `q=1` refreshes **all** non-selected clients, including those unable to upload; `0<q<1` draws refreshing independently using a dedicated RNG. Sampling uses *start-of-round* cached norms. Refreshed norms affect later rounds only. Energy accounting records selected `D+C+V`, refreshed-only `D+C`, and inactive `0` J; these values do not affect sampling. `resource_block_budget_std` controls per-round B variability.

The model parameters received from selected clients are converted into local differences `Δ_k = w_k−w`; only the specified selected clients contribute to the global update. Federated evaluation is calculated from the held-out examples in client partitions.

## Run this method

From the repository root, after installing the [project dependencies](../README.md#installation):

```bash
cd RAOCS4WFL
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
| `run_name` | `RAOCS4WFL(ResNet8-CIFAR10, 800_rounds, 5+-1_RBs, 1.5_sec, seed_42)` | Output filename stem only; **does not** configure the dataset, rounds, or RBs. Choose a unique value for each run. |
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
| `round_deadline_seconds` | `1.5` | Round/upload deadline τ, in **seconds**; standard methods use it as the effective uplink window. |
| `resource_block_budget` | `5` | Number of available wireless resource blocks B; resource-aware schedulers use it as a constraint. |
| `resource_block_budget_std` | `1` | Standard deviation of the sampled per-round RB budget (RBs); 0 gives a fixed budget. |
| `minimum_resource_block_budget` | `1` | Minimum allowed RB budget in variable-budget rounds (RBs). |
| `p_min` | `0.1` | Positive floor used in the **aggregation denominator**, `max(p_k, p_min)`; not a lower bound imposed on the sampling lottery. |
| `probe_update_norm_initialization` | `True` | When true, initialize cached update norms with a pre-round local-computation probe where applicable. |
| `refresh_probability` | `1` | Independent refresh probability q for each eligible non-selected client (0–1); selected clients always renew their cached norm. |

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

These ranges are used to **measure** operation energy; they do **not** constrain sampling or refreshing in RAOCS.

The source declares different `energy_config` defaults conditional on the dataset **at class definition time**. If you later switch `dataset_name`, explicitly check/update `energy_config` too; changing just the dataset field does not automatically regenerate these ranges. The values below are for the current CIFAR-10 class definition.

| Parameter | Current default | Meaning / effect |
| --- | --- | --- |
| `energy_budget_range_joules` | `(180.0, 220.0)` | Range for sampling each client’s expected energy budget E_k (J). This is **not** an explicit battery-charge simulator. |
| `download_energy_range_joules` | `(18.0, 22.0)` | Range for model download energy D_k per local training event (J). |
| `computation_energy_range_joules` | `(140.0, 180.0)` | Range for local training energy C_k per computed update (J). |
| `upload_energy_range_joules` | `(40.0, 120.0)` | Range for selected-client update upload energy V_k (J). |

## Reproduce the paper experiments

**Relevant experiments:** Tables I–IV. Table I: three baseline datasets; Table II: deadline sweeps (CIFAR-10 τ={0.2,0.6,1.5,3,6} s, B=5; CIFAR-100 τ={0.5,1,2,4,8} s, B=7); Table III: CIFAR-10 σ_B={0,1,2,4}; Table IV: q={0,1}, CIFAR-10 (B=10, τ=6 s) and CIFAR-100 (B=30, τ=16 s), both **without** energy probability caps.

For exact dataset-specific parameters—including `energy_config` and all wireless ranges—see the **[Table I–V reproduction guide](../docs/REPRODUCING_PAPER.md)** and the optional **[historical configuration manifest](../docs/published_experiments.csv)**. The manifest is a record of parameters, **not executable code**. A plot script at the repository root must be pointed at the JSON filenames produced by your runs.

## Practical notes

- All method simulations run independently; run `python run_simulation.py` **from this directory** so relative `results/` paths are predictable.
- The script writes a JSON history and final `.pt` checkpoint. Outputs and datasets are intentionally excluded from Git.
- A 2-round smoke test checks execution, not whether paper-scale accuracy is reproduced.
