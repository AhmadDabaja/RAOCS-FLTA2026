from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationConfig:
    """
    Central configuration for the federated learning simulation.

        This object stores all hyperparameters and runtime settings needed by
    the server, clients, and data pipeline.
    """
    results_dir: str = "results"
    run_name: str = "REAOCS(ResNet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42)"

    num_server_rounds: int = 800
    communication_budget: int | None = None
    learning_rate: float = 0.01
    batch_size: int = 64
    local_epochs: int = 16

    dataset_name: str = "cifar10"      # "fashionmnist" or "cifar10" or "cifar100"
    model_name: str = "resnet8-cifar10"         # "fcn-fashion" or "fcn-cifar10" or "resnet8-cifar10" or "resnet20-cifar100"
    num_partitions: int = 50
    alpha: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42

    # Communication Constraints
    round_deadline_seconds: float = 1.5
    resource_block_budget: int = 5

    # Minimum participation probability used in importance-corrected aggregation.
    # All p_k^t values are floored by p_min before aggregation.
    p_min: float = 0.1

    # Initialize cached norms with an optional probe before round 1.
    # After selection, each energy-eligible non-selected client independently
    # refreshes its cached norm with probability q (q=0: none; q=1: all eligible).
    probe_update_norm_initialization: bool = True
    refresh_probability: float = 0.0


    # Physical system
    wireless_config: dict[str, object] = field(default_factory=lambda: {
        "cell_radius": 500.0,
        "rb_bandwidth_hz": 1e6,
        "carrier_frequency_hz": 3.5e9,
        "delta_t": 1.0,
        "rate_efficiency": 0.65,
        "path_loss_exponent": 3.2,
        "shadowing_std_db": 8.0,
        "shadowing_corr_distance_m": 30.0,
        "mean_speed_range_mps": (0.2, 3.0),
        "mean_tx_power_range_dbm": (15.0, 23.0),
        "noise_figure_range_db": (5.0, 10.0),
    })

    if dataset_name == "fashionmnist":
        energy_config: dict[str, object] = field(default_factory=lambda: {
            "energy_budget_range_joules": (11.25, 13.75),
            "download_energy_range_joules": (1.125, 1.375),
            "computation_energy_range_joules": (9.0, 11.0),
            "upload_energy_range_joules": (2.5, 7.5),
        })
    elif dataset_name == "cifar10":
        energy_config: dict[str, object] = field(default_factory=lambda: {
            "energy_budget_range_joules": (45.0, 55.0),
            "download_energy_range_joules": (4.5, 5.5),
            "computation_energy_range_joules": (36.0, 44.0),
            "upload_energy_range_joules": (10.0, 30.0),
        })
    elif dataset_name == "cifar100":
        energy_config: dict[str, object] = field(default_factory=lambda: {
            "energy_budget_range_joules": (180.0, 220.0),
            "download_energy_range_joules": (18.0, 22.0),
            "computation_energy_range_joules": (140.0, 180.0),
            "upload_energy_range_joules": (40.0, 120.0),
        })


    def validate(self) -> None:
        """
        Validate configuration values before starting the simulation.
        """
        dataset_name = self.dataset_name.lower()
        model_name = self.model_name.lower()

        if dataset_name not in {"fashionmnist", "cifar10", "cifar100"}:
            raise ValueError(
                "dataset_name must be either 'fashionmnist', 'cifar10', or 'cifar100'."
            )

        valid_pairs = {
            ("fashionmnist", "fcn-fashion"),
            ("cifar10", "fcn-cifar10"),
            ("cifar10", "resnet8-cifar10"),
            ("cifar100", "resnet20-cifar100"),
        }

        if (dataset_name, model_name) not in valid_pairs:
            raise ValueError(
                f"Unsupported dataset/model pair: {dataset_name}/{model_name}. "
                f"Valid pairs are: {valid_pairs}."
            )

        if self.num_server_rounds <= 0:
            raise ValueError("num_server_rounds must be > 0.")

        if self.communication_budget is not None and self.communication_budget <= 0:
            raise ValueError("communication_budget must be > 0 or None.")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be > 0.")

        if self.num_partitions <= 0:
            raise ValueError("num_partitions must be > 0.")

        if self.round_deadline_seconds <= 0:
            raise ValueError("round_deadline_seconds must be > 0.")

        if self.resource_block_budget <= 0:
            raise ValueError("resource_block_budget must be > 0.")

        if not (0.0 < self.p_min <= 1.0):
            raise ValueError("p_min must be in (0, 1].")

        if not (0.0 <= self.refresh_probability <= 1.0):
            raise ValueError("refresh_probability must be in [0, 1].")

        if not isinstance(self.probe_update_norm_initialization, bool):
            raise ValueError("probe_update_norm_initialization must be a boolean.")

        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0.")

        if not (0.0 < self.test_ratio < 1.0):
            raise ValueError("test_ratio must be between 0 and 1.")