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
    run_name: str = "OCS(FCN-MNIST, 5_rounds, 5_clients,  5_RBs, 5_sec, seed_42)"

    num_server_rounds: int = 5
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

    # Optimal client sampling
    num_sampled_clients: float = 4.4

    # Minimum participation probability used in importance-corrected aggregation.
    # All p_k^t values are floored by p_min before aggregation.
    p_min: float = 0.1

    # Update-norm initialization and update mode.
    # probe_update_norm_initialization:
    #   If True, the server performs a fake/probe round before round 1
    #   to initialize all clients' update norms.
    #
    # update_norm_mode:
    #   "holt"   : selected clients use observed updates; unselected clients use Holt extrapolation.
    #   "newest" : selected clients use observed updates; unselected clients keep latest stored norm.
    #   "real"   : selected clients use observed updates; unselected clients are queried for real norms.
    probe_update_norm_initialization: bool = True
    update_norm_mode: str = "real"  # "holt", "newest", or "real"

    # Holt update-norm estimator.
    # Used only when update_norm_mode == "holt".
    if update_norm_mode == "holt":
        holt_rho: float = 0.5
        holt_beta: float = 0.3
        holt_gamma: float = 0.9
        holt_epsilon: float = 1e-12

    # Communication Constraints
    round_deadline_seconds: float = 5.0
    resource_block_budget: int = 5


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

        if self.num_sampled_clients <= 0:
            raise ValueError("num_sampled_clients must be > 0.")

        if self.num_sampled_clients > self.num_partitions:
            raise ValueError("num_sampled_clients cannot be greater than num_partitions.")

        if not (0.0 < self.p_min <= 1.0):
            raise ValueError("p_min must be in (0, 1].")

        if self.update_norm_mode not in {"holt", "newest", "real"}:
            raise ValueError("update_norm_mode must be one of: 'holt', 'newest', 'real'.")

        if not isinstance(self.probe_update_norm_initialization, bool):
            raise ValueError("probe_update_norm_initialization must be a boolean.")

        if self.update_norm_mode == "holt":
            if not (0.0 < self.holt_rho < 1.0):
                raise ValueError("holt_rho must be in (0, 1).")

            if not (0.0 < self.holt_beta < 1.0):
                raise ValueError("holt_beta must be in (0, 1).")

            if not (0.0 < self.holt_gamma <= 1.0):
                raise ValueError("holt_gamma must be in (0, 1].")

            if self.holt_epsilon <= 0.0:
                raise ValueError("holt_epsilon must be > 0.")

        if self.round_deadline_seconds <= 0:
            raise ValueError("round_deadline_seconds must be > 0.")
        if self.resource_block_budget <= 0:
            raise ValueError("resource_block_budget must be > 0.")

        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0.")

        if not (0.0 < self.test_ratio < 1.0):
            raise ValueError("test_ratio must be between 0 and 1.")