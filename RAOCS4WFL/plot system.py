from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from src.Models import FCNFashionModel, FCNCIFAR10Model, ResNet8CIFAR10Model
from src.PhysicalSystem import PhysicalSystem


# =========================
# User configuration
# =========================

NUM_ROUNDS = 50

# Moving-average window for noisy time-series plots.
# Use 1 to disable smoothing.
SMOOTHING_WINDOW = 1

# Optional horizontal reference line for each selected client.
# Use "median", "mean", or None.
REFERENCE_LINE_STATISTIC = "median"

# Chosen clients for temporal evolution plots.
# These are 0-indexed client IDs.
SELECTED_CLIENTS = [7, 23]

PLOTS_DIR = Path("../plots (FCN-MNIST)") / "experimental_system"

LOG_EPS = 1e-12

# CDF over time configuration.
# For each round, we compute a CDF across clients.
# Then we plot the median CDF over rounds with a 10%-90% band.
CDF_NUM_GRID_POINTS = 300
CDF_LOWER_PERCENTILE = 10
CDF_UPPER_PERCENTILE = 90


# =========================
# Model/system utilities
# =========================

def build_model_from_config(config: SimulationConfig):
    dataset_name = config.dataset_name.lower()
    model_name = config.model_name.lower()

    if dataset_name == "fashionmnist" and model_name == "fcn-fashion":
        return FCNFashionModel()

    if dataset_name == "cifar10" and model_name == "fcn-cifar10":
        return FCNCIFAR10Model()

    if dataset_name == "cifar10" and model_name == "resnet8-cifar10":
        return ResNet8CIFAR10Model()

    raise ValueError(
        f"Unsupported dataset/model pair: {dataset_name}/{model_name}"
    )


def compute_model_update_size_bits(model) -> int:
    model_parameters = model.get_model_parameters()
    update_size_bytes = sum(param.nbytes for param in model_parameters)
    return int(8 * update_size_bytes)


def simulate_physical_system(
    config: SimulationConfig,
    num_rounds: int,
) -> np.ndarray:
    physical_system = PhysicalSystem(
        num_clients=config.num_partitions,
        seed=config.seed,
        wireless_config=config.wireless_config,
    )

    uplink_rates = []

    for _ in range(num_rounds):
        state = physical_system(return_details=False)
        uplink_rates.append(state["communication_rates_bps"])

    return np.asarray(uplink_rates, dtype=float)


def compute_system_quantities(
    config: SimulationConfig,
    upload_size_bits: int,
    uplink_rates_bps: np.ndarray,
) -> dict[str, np.ndarray]:
    tau = float(config.round_deadline_seconds)
    B = float(config.resource_block_budget)

    upload_times_1rb = np.full_like(uplink_rates_bps, np.nan, dtype=float)
    b_min = np.full_like(uplink_rates_bps, np.nan, dtype=float)

    valid = uplink_rates_bps > 0.0

    upload_times_1rb[valid] = upload_size_bits / uplink_rates_bps[valid]
    b_min[valid] = upload_size_bits / (uplink_rates_bps[valid] * tau)

    feasible_by_total_budget = (
        np.isfinite(b_min)
        & (b_min > 0.0)
        & (b_min <= B)
    )

    return {
        "uplink_rates_bps": uplink_rates_bps,
        "upload_times_1rb_seconds": upload_times_1rb,
        "minimum_required_resource_blocks": b_min,
        "feasible_by_total_budget": feasible_by_total_budget,
    }


# =========================
# Plot helpers
# =========================

def apply_paper_grid() -> None:
    """
    Add major and minor grid lines for paper-readable plots.
    """
    plt.minorticks_on()
    plt.grid(True, which="major", linewidth=0.8, alpha=0.75)
    plt.grid(True, which="minor", linewidth=0.4, alpha=0.35)


def save_current_figure(output_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    NaN-aware moving average.

    If window <= 1, returns the original values.
    """
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values

    valid = np.isfinite(values).astype(float)
    clean = np.where(np.isfinite(values), values, 0.0)

    kernel = np.ones(window, dtype=float)

    numerator = np.convolve(clean, kernel, mode="same")
    denominator = np.convolve(valid, kernel, mode="same")

    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full_like(values, np.nan, dtype=float),
        where=denominator > 0.0,
    )

    return smoothed

def reference_value(values: np.ndarray, statistic: str | None) -> float:
    """
    Compute a horizontal reference value for a time series.

    statistic can be:
        - "median"
        - "mean"
        - None
    """
    clean_values = finite_values(values)

    if len(clean_values) == 0 or statistic is None:
        return math.nan

    if statistic == "median":
        return float(np.median(clean_values))

    if statistic == "mean":
        return float(np.mean(clean_values))

    raise ValueError("REFERENCE_LINE_STATISTIC must be 'median', 'mean', or None.")

def plot_temporal_cdf_band(
    values: np.ndarray,
    xlabel: str,
    output_name: str,
    x_log: bool = False,
) -> None:
    """
    Compute one CDF per round across clients, then summarize CDFs over time.

    For each round t:

        F_t(x) = number of clients with X_k^t <= x / number of valid clients

    Then we plot:
        - median_t F_t(x)
        - 10th-90th percentile band over rounds

    This avoids pooling all rounds and clients into one mixed distribution.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 2:
        raise ValueError("values must have shape (num_rounds, num_clients).")

    finite = values[np.isfinite(values)]

    if x_log:
        finite = finite[finite > 0.0]

    if len(finite) == 0:
        print(f"Skipping {output_name}: no finite values.")
        return

    x_min = float(np.min(finite))
    x_max = float(np.max(finite))

    if x_log:
        x_grid = np.logspace(
            np.log10(max(x_min, LOG_EPS)),
            np.log10(x_max),
            CDF_NUM_GRID_POINTS,
        )
    else:
        x_grid = np.linspace(x_min, x_max, CDF_NUM_GRID_POINTS)

    cdfs = []

    for round_values in values:
        round_values = round_values[np.isfinite(round_values)]

        if x_log:
            round_values = round_values[round_values > 0.0]

        if len(round_values) == 0:
            continue

        round_values = np.sort(round_values)

        # For each x, count how many client values are <= x.
        cdf = np.searchsorted(round_values, x_grid, side="right") / len(round_values)
        cdfs.append(cdf)

    if len(cdfs) == 0:
        print(f"Skipping {output_name}: no valid per-round CDFs.")
        return

    cdfs = np.asarray(cdfs, dtype=float)

    median_cdf = np.percentile(cdfs, 50, axis=0)
    lower_cdf = np.percentile(cdfs, CDF_LOWER_PERCENTILE, axis=0)
    upper_cdf = np.percentile(cdfs, CDF_UPPER_PERCENTILE, axis=0)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_grid,
        median_cdf,
        linewidth=2,
        label="Median CDF over rounds",
    )

    plt.fill_between(
        x_grid,
        lower_cdf,
        upper_cdf,
        alpha=0.20,
        label=f"{CDF_LOWER_PERCENTILE}%-{CDF_UPPER_PERCENTILE}% band",
    )

    if x_log:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.legend()
    apply_paper_grid()
    save_current_figure(output_name)


def plot_selected_clients_vs_round(
    values: np.ndarray,
    selected_clients: list[int],
    ylabel: str,
    output_name: str,
    y_log: bool = False,
) -> None:
    rounds = np.arange(1, values.shape[0] + 1)

    plt.figure(figsize=(8, 5))

    for client_id in selected_clients:
        if client_id < 0 or client_id >= values.shape[1]:
            continue

        raw_y = values[:, client_id]

        if y_log:
            raw_y = np.where(np.isfinite(raw_y) & (raw_y > 0.0), raw_y, np.nan)

        smoothed_y = moving_average(raw_y, SMOOTHING_WINDOW)

        line, = plt.plot(
            rounds,
            smoothed_y,
            linewidth=2,
            label=f"Client {client_id}",
        )

        ref = reference_value(raw_y, REFERENCE_LINE_STATISTIC)

        if np.isfinite(ref) and (not y_log or ref > 0.0):
            plt.axhline(
                y=ref,
                linestyle=":",
                linewidth=1.5,
                color=line.get_color(),
                alpha=0.9,
            )

    if y_log:
        plt.yscale("log")

    plt.xlabel("Round")
    plt.ylabel(ylabel)

    if REFERENCE_LINE_STATISTIC is None:
        legend_title = None
    else:
        legend_title = f"Solid: moving average, dotted: {REFERENCE_LINE_STATISTIC}"

    plt.legend(title=legend_title)
    apply_paper_grid()
    save_current_figure(output_name)


def describe(values: np.ndarray) -> dict[str, float]:
    values = finite_values(values)

    if len(values) == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p10": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "max": math.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def save_summary(
    config: SimulationConfig,
    upload_size_bits: int,
    quantities: dict[str, np.ndarray],
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    feasible_budget = quantities["feasible_by_total_budget"]

    payload: dict[str, Any] = {
        "num_rounds": NUM_ROUNDS,
        "smoothing_window": SMOOTHING_WINDOW,
        "selected_clients": SELECTED_CLIENTS,
        "num_clients": int(config.num_partitions),
        "tau_seconds": float(config.round_deadline_seconds),
        "resource_block_budget": float(config.resource_block_budget),
        "upload_size_bits": int(upload_size_bits),
        "uplink_rate_per_rb_bps": describe(quantities["uplink_rates_bps"]),
        "upload_time_with_1_rb_seconds": describe(
            quantities["upload_times_1rb_seconds"]
        ),
        "minimum_required_resource_blocks": describe(
            quantities["minimum_required_resource_blocks"]
        ),
        "avg_clients_feasible_by_resource_budget": float(
            np.mean(np.sum(feasible_budget, axis=1))
        ),
    }

    with (PLOTS_DIR / "physical_system_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


# =========================
# Main
# =========================

def main() -> None:
    config = SimulationConfig()
    config.validate()

    model = build_model_from_config(config)
    upload_size_bits = compute_model_update_size_bits(model)

    uplink_rates_bps = simulate_physical_system(
        config=config,
        num_rounds=NUM_ROUNDS,
    )

    quantities = compute_system_quantities(
        config=config,
        upload_size_bits=int(upload_size_bits),
        uplink_rates_bps=uplink_rates_bps,
    )

    uplink_rates_mbps = quantities["uplink_rates_bps"] / 1e6
    upload_times_1rb = quantities["upload_times_1rb_seconds"]
    b_min = quantities["minimum_required_resource_blocks"]

    feasible_by_total_budget = quantities["feasible_by_total_budget"]

    # -------------------------
    # 1. Uplink rate CDF
    # -------------------------
    plot_temporal_cdf_band(
        values=uplink_rates_mbps,
        xlabel="Uplink rate per RB (Mbps)",
        output_name="01_uplink_rate_per_rb_temporal_cdf.png",
        x_log=True,
    )


    # -------------------------
    # 2. Minimum required RBs CDF
    # -------------------------
    plot_temporal_cdf_band(
        values=b_min,
        xlabel="Minimum required RBs to satisfy deadline",
        output_name="02_minimum_required_resource_blocks_temporal_cdf.png",
        x_log=True,
    )

    # -------------------------
    # 3. Feasible clients over rounds
    # -------------------------
    rounds = np.arange(1, NUM_ROUNDS + 1)

    num_feasible_budget = np.sum(feasible_by_total_budget, axis=1)
    num_feasible_budget_smooth = moving_average(num_feasible_budget, SMOOTHING_WINDOW)

    plt.figure(figsize=(8, 5))

    line_budget, = plt.plot(
        rounds,
        num_feasible_budget_smooth,
        linewidth=2,
        label=r"$b_{\min} \leq B$",
    )

    ref_budget = reference_value(num_feasible_budget, REFERENCE_LINE_STATISTIC)

    if np.isfinite(ref_budget):
        plt.axhline(
            y=ref_budget,
            linestyle=":",
            linewidth=1.5,
            color=line_budget.get_color(),
            alpha=0.9,
        )

    plt.xlabel("Round")
    plt.ylabel("Number of feasible clients")
    plt.legend(
        title=f"Solid: moving average, dotted: {REFERENCE_LINE_STATISTIC}"
        if REFERENCE_LINE_STATISTIC is not None
        else None
    )
    apply_paper_grid()
    save_current_figure("03_feasible_clients_vs_round.png")

    # -------------------------
    # 4. Selected clients: uplink rate over rounds
    # -------------------------
    plot_selected_clients_vs_round(
        values=uplink_rates_mbps,
        selected_clients=SELECTED_CLIENTS,
        ylabel="Uplink rate per RB (Mbps)",
        output_name="04_selected_clients_uplink_rate_per_rb_vs_round.png",
        y_log=True,
    )


    # -------------------------
    # 5. Selected clients: minimum required RBs over rounds
    # -------------------------
    plot_selected_clients_vs_round(
        values=b_min,
        selected_clients=SELECTED_CLIENTS,
        ylabel="Minimum required RBs",
        output_name="05_selected_clients_minimum_required_rbs_vs_round.png",
        y_log=True,
    )

    save_summary(
        config=config,
        upload_size_bits=upload_size_bits,
        quantities=quantities,
    )

    print(f"Experimental system paper plots saved in: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()