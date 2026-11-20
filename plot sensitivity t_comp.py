from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# =========================
# User configuration
# =========================

MAX_ROUNDS: int | None = 800

# Smoothing used only for evaluation curves and noisy cost curves.
# SMOOTHING_ALPHA = 0.0 means no smoothing.
SMOOTHING_ALPHA = 0.75

LOG_EPS = 1e-12

# Optional per-client diagnostics. Usually False for paper figures.
PLOT_PER_CLIENT_DIAGNOSTICS = False

# Plot format.
EXTENSION = "PDF"  # "PDF" or "PNG"

# Computation-time variability experiment:
# fixed mu_comp = 3.5 s, fixed cutoff c = 4.5 s, fixed round deadline tau = 6 s.
# Only sigma_comp changes.
RESULTS_DIR = Path("RAOCSplus/results")

EXPERIMENTS = [
    {
        "path": RESULTS_DIR / "RAOCSplus(ResNet8-CIFAR10, 800_rounds, 5_RBs, 4.5+1.5_sec, 3.5+-0.25_sec, seed_42).json",
        "label": r"$\sigma_{\mathrm{comp}}=0.25$",
        "sigma_comp": 0.25,
    },
    {
        "path": RESULTS_DIR / "RAOCSplus(ResNet8-CIFAR10, 800_rounds, 5_RBs, 4.5+1.5_sec, 3.5+-0.5_sec, seed_42).json",
        "label": r"$\sigma_{\mathrm{comp}}=0.50$",
        "sigma_comp": 0.50,
    },
    {
        "path": RESULTS_DIR / "RAOCSplus(ResNet8-CIFAR10, 800_rounds, 5_RBs, 4.5+1.5_sec, 3.5+-1.0_sec, seed_42).json",
        "label": r"$\sigma_{\mathrm{comp}}=1.00$",
        "sigma_comp": 1.00,
    },
    {
        "path": RESULTS_DIR / "RAOCSplus(ResNet8-CIFAR10, 800_rounds, 5_RBs, 4.5+1.5_sec, 3.5+-2.0_sec, seed_42).json",
        "label": r"$\sigma_{\mathrm{comp}}=2.00$",
        "sigma_comp": 2.00,
    },
    {
        "path": RESULTS_DIR / "RAOCSplus(ResNet8-CIFAR10, 800_rounds, 5_RBs, 4.5+1.5_sec, 3.5+-4.0_sec, seed_42).json",
        "label": r"$\sigma_{\mathrm{comp}}=4.00$",
        "sigma_comp": 4.00,
    },
]

PLOTS_DIR = Path("plots (ResNet8-CIFAR10)") / "computation_time_variability"


# =========================
# Plot visual style
# =========================

FIGURE_SIZE = (8.0, 5.0)

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
    "lines.linewidth": 2.8,
    "lines.markersize": 5.5,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Loading utilities
# =========================

def resolve_result_path(path: str | Path) -> Path:
    """
    Resolve result paths robustly.

    The primary path is the project-relative path in EXPERIMENTS.
    For quick testing in a notebook/sandbox, this also falls back to /mnt/data/<filename>.
    """
    path = Path(path)

    if path.exists():
        return path

    fallback_path = Path("/mnt/data") / path.name
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Result file not found. Tried: {path} and {fallback_path}"
    )


def load_result(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_result_path(path)

    with resolved_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "metrics" not in result:
        raise KeyError("Result file must contain a 'metrics' field.")

    metrics = result["metrics"]

    if not isinstance(metrics, list):
        raise TypeError("'metrics' must be a list.")

    return metrics


def limit_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if MAX_ROUNDS is None:
        return metrics

    if MAX_ROUNDS <= 0:
        raise ValueError("MAX_ROUNDS must be a positive integer or None.")

    return metrics[:MAX_ROUNDS]


def get_limited_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    return limit_metrics(get_metrics(result))


# =========================
# Basic helpers
# =========================

def client_key(client_id: int) -> str:
    return f"client_{client_id}"


def get_num_clients(metrics: list[dict[str, Any]]) -> int:
    for row in metrics:
        rates = row.get("uplink_rates_per_resource_block_bps")
        if rates is not None:
            return len(rates)

    raise ValueError("Could not infer number of clients from uplink rates.")


def round_axis(metrics: list[dict[str, Any]]) -> list[int]:
    return [int(row["round"]) for row in metrics]


def clean_numeric_values(values: list[float]) -> list[float]:
    clean_values = []

    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(value):
            clean_values.append(value)

    return clean_values


def mean_std(values: list[float]) -> tuple[float, float]:
    clean_values = clean_numeric_values(values)

    if not clean_values:
        return math.nan, math.nan

    mean_value = sum(clean_values) / len(clean_values)
    variance = sum((v - mean_value) ** 2 for v in clean_values) / len(clean_values)

    return mean_value, math.sqrt(variance)


def percentile(values: list[float], q: float) -> float:
    clean_values = sorted(clean_numeric_values(values))

    if not clean_values:
        return math.nan

    if len(clean_values) == 1:
        return clean_values[0]

    position = q * (len(clean_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return clean_values[lower_index]

    lower_value = clean_values[lower_index]
    upper_value = clean_values[upper_index]
    weight = position - lower_index

    return lower_value * (1.0 - weight) + upper_value * weight


def summarize_distribution(values: list[float], prefix: str) -> dict[str, float]:
    clean_values = clean_numeric_values(values)

    if not clean_values:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_q25": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_q75": math.nan,
            f"{prefix}_max": math.nan,
        }

    mean_value, std_value = mean_std(clean_values)

    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": std_value,
        f"{prefix}_min": min(clean_values),
        f"{prefix}_q25": percentile(clean_values, 0.25),
        f"{prefix}_median": percentile(clean_values, 0.50),
        f"{prefix}_q75": percentile(clean_values, 0.75),
        f"{prefix}_max": max(clean_values),
    }


def last_finite(values: list[float]) -> float:
    clean_values = clean_numeric_values(values)

    if not clean_values:
        return math.nan

    return clean_values[-1]


def exponential_moving_average(values: list[float], smoothing_alpha: float) -> list[float]:
    """
    smoothed_t = (1 - alpha) * value_t + alpha * smoothed_{t-1}
    alpha = 0.0 gives no smoothing.
    """
    if not (0.0 <= smoothing_alpha < 1.0):
        raise ValueError("SMOOTHING_ALPHA must be in [0, 1).")

    if smoothing_alpha == 0.0:
        return values

    smoothed_values = []
    previous_value = None

    for value in values:
        value = float(value)

        if math.isnan(value):
            smoothed_values.append(math.nan)
            continue

        if previous_value is None:
            smoothed_value = value
        else:
            smoothed_value = (
                (1.0 - smoothing_alpha) * value
                + smoothing_alpha * previous_value
            )

        smoothed_values.append(smoothed_value)
        previous_value = smoothed_value

    return smoothed_values


# =========================
# Derived metric computation
# =========================

def compute_round_derived_metrics(row: dict[str, Any]) -> dict[str, Any]:
    selected_clients = [int(client_id) for client_id in row.get("selected_clients", [])]
    available_clients = [int(client_id) for client_id in row.get("available_clients", [])]
    precomputed_clients = [int(client_id) for client_id in row.get("precomputed_clients", [])]

    uplink_rates = row.get("uplink_rates_per_resource_block_bps", [])
    assigned_resource_blocks = row.get("assigned_resource_blocks", {})
    upload_bytes = row.get("upload_bytes", {})
    download_bytes = row.get("download_bytes", {})
    norm_report_bytes = row.get("norm_report_bytes", {})
    actual_energy = row.get("actual_energy_consumed_joules", [])
    computation_times = row.get("computation_times_seconds", [])

    num_clients = len(uplink_rates) if uplink_rates is not None else len(computation_times)

    upload_times_selected: dict[int, float] = {}

    for client_id in selected_clients:
        key = client_key(client_id)

        b_k = assigned_resource_blocks.get(key)
        upload_k = upload_bytes.get(key)
        nu_k = uplink_rates[client_id] if client_id < len(uplink_rates) else None

        if (
            b_k is None
            or upload_k is None
            or nu_k is None
            or float(b_k) <= 0.0
            or float(nu_k) <= 0.0
        ):
            upload_time = math.nan
        else:
            upload_bits = 8.0 * float(upload_k)
            upload_time = upload_bits / (float(nu_k) * float(b_k))

        upload_times_selected[client_id] = upload_time

    selected_upload_times = clean_numeric_values(list(upload_times_selected.values()))
    max_selected_upload_time = max(selected_upload_times) if selected_upload_times else 0.0

    used_resource_blocks = sum(
        float(value)
        for value in assigned_resource_blocks.values()
        if value is not None and math.isfinite(float(value))
    )

    upload_window = float(row.get("upload_window_seconds", math.nan))
    decision_time = float(row.get("server_decision_time_seconds", math.nan))

    if math.isfinite(decision_time):
        wall_clock_round_time = decision_time + max_selected_upload_time
    else:
        wall_clock_round_time = max_selected_upload_time

    computation_mean, computation_std = mean_std(computation_times)
    selected_upload_mean, selected_upload_std = mean_std(selected_upload_times)

    total_upload_bytes = sum(float(value) for value in upload_bytes.values())
    total_download_bytes = sum(float(value) for value in download_bytes.values())
    total_norm_report_bytes = sum(float(value) for value in norm_report_bytes.values())
    total_communication_bytes = (
        total_upload_bytes
        + total_download_bytes
        + total_norm_report_bytes
    )

    total_energy = sum(clean_numeric_values(actual_energy))

    computation_clean = clean_numeric_values(computation_times)

    return {
        # General training/network diagnostics.
        "used_resource_blocks": used_resource_blocks,
        "num_selected_clients": len(selected_clients),
        "upload_times_selected": upload_times_selected,
        "selected_upload_times": selected_upload_times,
        "selected_upload_time_mean": selected_upload_mean,
        "selected_upload_time_std": selected_upload_std,
        "max_selected_upload_time": max_selected_upload_time,
        "wall_clock_round_time": wall_clock_round_time,

        # Computation-time extension diagnostics.
        "num_clients": num_clients,
        "num_available_clients": len(available_clients),
        "num_unavailable_clients": max(num_clients - len(available_clients), 0),
        "available_fraction": (len(available_clients) / num_clients) if num_clients else math.nan,
        "num_precomputed_clients": len(precomputed_clients),
        "server_decision_time": decision_time,
        "upload_window": upload_window,
        "computation_time_mean": computation_mean,
        "computation_time_std": computation_std,
        "computation_time_min": min(computation_clean) if computation_clean else math.nan,
        "computation_time_max": max(computation_clean) if computation_clean else math.nan,

        # Cost diagnostics.
        "round_energy_joules": total_energy,
        "upload_bytes": total_upload_bytes,
        "download_bytes": total_download_bytes,
        "norm_report_bytes": total_norm_report_bytes,
        "communication_bytes": total_communication_bytes,
    }


def compute_all_derived_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = get_limited_metrics(result)
    return [compute_round_derived_metrics(row) for row in metrics]


# =========================
# Figure utilities
# =========================

def save_current_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if EXTENSION.upper() == "PNG":
        plt.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    elif EXTENSION.upper() == "PDF":
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    else:
        raise ValueError("EXTENSION must be either 'PNG' or 'PDF'.")


def apply_log_scales(x_log: bool = False, y_log: bool = False) -> None:
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")


def plot_eval_metric(metric_name: str, y_label: str, output_name: str) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    for experiment in EXPERIMENTS:
        result = load_result(experiment["path"])
        metrics = get_limited_metrics(result)

        x_values = round_axis(metrics)
        y_values = [float(row[metric_name]) for row in metrics]
        smoothed_y_values = exponential_moving_average(y_values, SMOOTHING_ALPHA)

        plt.plot(x_values, smoothed_y_values, marker="o", label=experiment["label"])

    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    save_current_figure(PLOTS_DIR / output_name)
    plt.close()


def plot_simple_derived_metric(
    derived_key: str,
    y_label: str,
    output_name: str,
    y_log: bool = False,
    smooth: bool = False,
) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    for experiment in EXPERIMENTS:
        result = load_result(experiment["path"])
        metrics = get_limited_metrics(result)
        derived_metrics = compute_all_derived_metrics(result)

        x_values = round_axis(metrics)
        y_values = [float(row[derived_key]) for row in derived_metrics]

        if smooth:
            y_values = exponential_moving_average(y_values, SMOOTHING_ALPHA)

        plt.plot(x_values, y_values, marker="o", label=experiment["label"])

    apply_log_scales(y_log=y_log)

    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    save_current_figure(PLOTS_DIR / output_name)
    plt.close()


def plot_mean_std_derived_metric(
    mean_key: str,
    std_key: str,
    y_label: str,
    output_name: str,
    y_log: bool = False,
) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    for experiment in EXPERIMENTS:
        result = load_result(experiment["path"])
        metrics = get_limited_metrics(result)
        derived_metrics = compute_all_derived_metrics(result)

        x_values = round_axis(metrics)
        means = [float(row[mean_key]) for row in derived_metrics]
        stds = [float(row[std_key]) for row in derived_metrics]

        lower = []
        upper = []

        for mean_value, std_value in zip(means, stds):
            if not math.isfinite(mean_value):
                lower.append(math.nan)
                upper.append(math.nan)
                continue

            low = mean_value - std_value
            high = mean_value + std_value

            if y_log:
                low = max(low, LOG_EPS)
                high = max(high, LOG_EPS)

            lower.append(low)
            upper.append(high)

        plt.plot(x_values, means, marker="o", label=experiment["label"])
        plt.fill_between(x_values, lower, upper, alpha=0.15)

    apply_log_scales(y_log=y_log)

    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    save_current_figure(PLOTS_DIR / output_name)
    plt.close()


# =========================
# Summary-over-sigma plots
# =========================

def plot_summary_metric_vs_sigma(
    summary_rows: list[dict[str, Any]],
    metric_key: str,
    y_label: str,
    output_name: str,
    value_multiplier: float = 1.0,
    y_log: bool = False,
) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    x_values = [float(row["sigma_comp"]) for row in summary_rows]
    y_values = [float(row[metric_key]) * value_multiplier for row in summary_rows]

    plt.plot(x_values, y_values, marker="o")

    apply_log_scales(y_log=y_log)

    plt.xlabel(r"$\sigma_{\mathrm{comp}}$ (s)")
    plt.ylabel(y_label)
    plt.grid(True, which="both")
    plt.tight_layout()

    save_current_figure(PLOTS_DIR / "summary_vs_sigma" / output_name)
    plt.close()


def plot_summary_mean_std_vs_sigma(
    summary_rows: list[dict[str, Any]],
    mean_key: str,
    std_key: str,
    y_label: str,
    output_name: str,
    value_multiplier: float = 1.0,
) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    x_values = [float(row["sigma_comp"]) for row in summary_rows]
    means = [float(row[mean_key]) * value_multiplier for row in summary_rows]
    stds = [float(row[std_key]) * value_multiplier for row in summary_rows]

    plt.errorbar(x_values, means, yerr=stds, marker="o", capsize=5)

    plt.xlabel(r"$\sigma_{\mathrm{comp}}$ (s)")
    plt.ylabel(y_label)
    plt.grid(True, which="both")
    plt.tight_layout()

    save_current_figure(PLOTS_DIR / "summary_vs_sigma" / output_name)
    plt.close()


# =========================
# Per-client diagnostics
# =========================

def build_per_client_values(
    result: dict[str, Any],
    metric_name: str,
    client_id: int,
) -> list[float]:
    metrics = get_limited_metrics(result)
    derived_metrics = compute_all_derived_metrics(result)

    values = []

    for row, derived_row in zip(metrics, derived_metrics):
        selected_clients = [int(client) for client in row.get("selected_clients", [])]
        available_clients = [int(client) for client in row.get("available_clients", [])]
        key = client_key(client_id)

        if metric_name == "uplink_rates_per_resource_block_bps":
            values.append(float(row["uplink_rates_per_resource_block_bps"][client_id]))

        elif metric_name == "computation_times_seconds":
            values.append(float(row["computation_times_seconds"][client_id]))

        elif metric_name == "availability_indicator":
            values.append(1.0 if client_id in available_clients else 0.0)

        elif metric_name == "assigned_resource_blocks":
            if client_id in selected_clients:
                values.append(float(row["assigned_resource_blocks"].get(key, math.nan)))
            else:
                values.append(math.nan)

        elif metric_name == "upload_times_seconds":
            if client_id in selected_clients:
                values.append(float(derived_row["upload_times_selected"].get(client_id, math.nan)))
            else:
                values.append(math.nan)

        else:
            raise ValueError(f"Unknown per-client metric: {metric_name}")

    return values


def plot_per_client_diagnostics(
    metric_name: str,
    y_label: str,
    y_log: bool = False,
) -> None:
    first_result = load_result(EXPERIMENTS[0]["path"])
    num_clients = get_num_clients(get_limited_metrics(first_result))

    output_dir = PLOTS_DIR / "diagnostics" / metric_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id in range(num_clients):
        plt.figure(figsize=FIGURE_SIZE)

        for experiment in EXPERIMENTS:
            result = load_result(experiment["path"])
            metrics = get_limited_metrics(result)

            x_values = round_axis(metrics)
            y_values = build_per_client_values(result, metric_name, client_id)

            plt.plot(x_values, y_values, marker="o", label=experiment["label"])

        apply_log_scales(y_log=y_log)

        plt.xlabel("Round")
        plt.ylabel(y_label)
        plt.title(f"Client {client_id}")
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()

        save_current_figure(output_dir / f"client_{client_id}")
        plt.close()


# =========================
# Summary statistics
# =========================

def infer_sigma_comp(result: dict[str, Any], experiment: dict[str, Any]) -> float:
    if "sigma_comp" in experiment:
        return float(experiment["sigma_comp"])

    config = result.get("config", {})
    computation_config = config.get("computation_config", {})
    std_range = computation_config.get("std_computation_time_range_seconds")

    if isinstance(std_range, (list, tuple)) and std_range:
        return float(std_range[0])

    return math.nan


def compute_experiment_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    result = load_result(experiment["path"])
    metrics = get_limited_metrics(result)
    derived_metrics = compute_all_derived_metrics(result)
    config = result.get("config", {})
    computation_config = config.get("computation_config", {})

    raw_acc = [float(row["fed_eval_acc"]) for row in metrics]
    raw_loss = [float(row["fed_eval_loss"]) for row in metrics]

    smoothed_acc = exponential_moving_average(raw_acc, SMOOTHING_ALPHA)
    smoothed_loss = exponential_moving_average(raw_loss, SMOOTHING_ALPHA)

    used_resource_blocks = [float(row["used_resource_blocks"]) for row in derived_metrics]
    num_selected_clients = [float(row["num_selected_clients"]) for row in derived_metrics]
    num_available_clients = [float(row["num_available_clients"]) for row in derived_metrics]
    available_fraction = [float(row["available_fraction"]) for row in derived_metrics]
    num_precomputed_clients = [float(row["num_precomputed_clients"]) for row in derived_metrics]
    upload_windows = [float(row["upload_window"]) for row in derived_metrics]
    decision_times = [float(row["server_decision_time"]) for row in derived_metrics]
    computation_means = [float(row["computation_time_mean"]) for row in derived_metrics]
    computation_stds = [float(row["computation_time_std"]) for row in derived_metrics]
    max_selected_upload_times = [float(row["max_selected_upload_time"]) for row in derived_metrics]
    selected_upload_time_means = [float(row["selected_upload_time_mean"]) for row in derived_metrics]

    round_energy = [float(row["round_energy_joules"]) for row in derived_metrics]
    upload_bytes = [float(row["upload_bytes"]) for row in derived_metrics]
    download_bytes = [float(row["download_bytes"]) for row in derived_metrics]
    norm_report_bytes = [float(row["norm_report_bytes"]) for row in derived_metrics]
    communication_bytes = [float(row["communication_bytes"]) for row in derived_metrics]

    resource_block_budget = float(config.get("resource_block_budget", math.nan))
    round_deadline_seconds = float(config.get("round_deadline_seconds", math.nan))
    computation_cutoff_seconds = float(config.get("computation_cutoff_seconds", math.nan))

    over_budget_rounds = (
        sum(
            1
            for value in used_resource_blocks
            if math.isfinite(value) and value > resource_block_budget + 1e-9
        )
        if math.isfinite(resource_block_budget)
        else math.nan
    )

    decision_at_cutoff_rounds = (
        sum(
            1
            for value in decision_times
            if math.isfinite(value)
            and math.isfinite(computation_cutoff_seconds)
            and abs(value - computation_cutoff_seconds) <= 1e-9
        )
        if math.isfinite(computation_cutoff_seconds)
        else math.nan
    )

    sigma_comp = infer_sigma_comp(result, experiment)

    mu_range = computation_config.get("mean_computation_time_range_seconds", [math.nan, math.nan])
    mu_comp = float(mu_range[0]) if isinstance(mu_range, (list, tuple)) and mu_range else math.nan

    summary: dict[str, Any] = {
        "setting": experiment["label"],
        "path": str(experiment["path"]),
        "num_rounds": len(metrics),
        "smoothing_alpha": SMOOTHING_ALPHA,

        "sigma_comp": sigma_comp,
        "mu_comp": mu_comp,
        "resource_block_budget_B": resource_block_budget,
        "round_deadline_tau_seconds": round_deadline_seconds,
        "computation_cutoff_seconds": computation_cutoff_seconds,

        "final_raw_accuracy": last_finite(raw_acc),
        "final_smoothed_accuracy": last_finite(smoothed_acc),
        "best_raw_accuracy": max(clean_numeric_values(raw_acc)),
        "best_smoothed_accuracy": max(clean_numeric_values(smoothed_acc)),

        "final_raw_loss": last_finite(raw_loss),
        "final_smoothed_loss": last_finite(smoothed_loss),
        "min_raw_loss": min(clean_numeric_values(raw_loss)),
        "min_smoothed_loss": min(clean_numeric_values(smoothed_loss)),

        "total_energy_joules": sum(clean_numeric_values(round_energy)),
        "total_communication_bytes": sum(clean_numeric_values(communication_bytes)),
        "total_upload_bytes": sum(clean_numeric_values(upload_bytes)),
        "total_download_bytes": sum(clean_numeric_values(download_bytes)),
        "total_norm_report_bytes": sum(clean_numeric_values(norm_report_bytes)),

        "over_budget_rounds": over_budget_rounds,
        "over_budget_round_fraction": over_budget_rounds / len(metrics) if len(metrics) else math.nan,
        "decision_at_cutoff_rounds": decision_at_cutoff_rounds,
        "decision_at_cutoff_fraction": decision_at_cutoff_rounds / len(metrics) if len(metrics) else math.nan,
    }

    summary.update(summarize_distribution(used_resource_blocks, "used_resource_blocks"))
    summary.update(summarize_distribution(num_selected_clients, "num_selected_clients"))
    summary.update(summarize_distribution(num_available_clients, "num_available_clients"))
    summary.update(summarize_distribution(available_fraction, "available_fraction"))
    summary.update(summarize_distribution(num_precomputed_clients, "num_precomputed_clients"))
    summary.update(summarize_distribution(upload_windows, "upload_window_seconds"))
    summary.update(summarize_distribution(decision_times, "server_decision_time_seconds"))
    summary.update(summarize_distribution(computation_means, "mean_computation_time_seconds"))
    summary.update(summarize_distribution(computation_stds, "std_computation_time_seconds"))
    summary.update(summarize_distribution(max_selected_upload_times, "max_selected_upload_time_seconds"))
    summary.update(summarize_distribution(selected_upload_time_means, "mean_selected_upload_time_seconds"))
    summary.update(summarize_distribution(round_energy, "round_energy_joules"))
    summary.update(summarize_distribution(communication_bytes, "round_communication_bytes"))

    return summary


def format_mean_std(
    row: dict[str, Any],
    mean_key: str,
    std_key: str,
    precision: int = 2,
) -> str:
    mean_value = float(row[mean_key])
    std_value = float(row[std_key])

    if not math.isfinite(mean_value) or not math.isfinite(std_value):
        return "--"

    return f"${mean_value:.{precision}f} \\pm {std_value:.{precision}f}$"


def save_latex_summary_table(rows: list[dict[str, Any]]) -> None:
    output_path = PLOTS_DIR / "computation_time_variability_table.tex"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity to computation-time variability on CIFAR-10 with ResNet-8. The mean computation time is fixed to $\mu_{\mathrm{comp}}=3.5$ s, while $\sigma_{\mathrm{comp}}$ is varied.}",
        r"\label{tab:computation_time_sensitivity}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$\sigma_{\mathrm{comp}}$ & Final Acc. (\%) & Upload Window (s) & Available Clients & Selected Clients \\",
        r"\midrule",
    ]

    for row in rows:
        sigma = float(row["sigma_comp"])
        final_acc = 100.0 * float(row["final_smoothed_accuracy"])

        upload_window = format_mean_std(
            row,
            "upload_window_seconds_mean",
            "upload_window_seconds_std",
            precision=2,
        )

        available_clients = format_mean_std(
            row,
            "num_available_clients_mean",
            "num_available_clients_std",
            precision=2,
        )

        selected_clients = format_mean_std(
            row,
            "num_selected_clients_mean",
            "num_selected_clients_std",
            precision=2,
        )

        lines.append(
            rf"${sigma:.2f}$ & ${final_acc:.2f}$ & {upload_window} & {available_clients} & {selected_clients} \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"LaTeX summary table saved in: {output_path.resolve()}")


def save_summary_statistics() -> list[dict[str, Any]]:
    rows = [compute_experiment_summary(experiment) for experiment in EXPERIMENTS]
    rows.sort(key=lambda row: float(row["sigma_comp"]))

    output_path = PLOTS_DIR / "computation_variability_summary_statistics.csv"
    fieldnames = list(rows[0].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_latex_summary_table(rows)

    print(f"Summary statistics saved in: {output_path.resolve()}")

    print("\nCompact computation-variability summary:")
    print(
        f"{'sigma':>7s} "
        f"{'Final acc':>10s} "
        f"{'Best acc':>10s} "
        f"{'Avail.':>15s} "
        f"{'Selected':>15s} "
        f"{'Upload win.':>15s} "
        f"{'Energy (kJ)':>12s} "
        f"{'Comm. (GB)':>11s} "
        f"{'Over-B':>8s}"
    )

    for row in rows:
        print(
            f"{float(row['sigma_comp']):7.2f} "
            f"{100.0 * float(row['final_smoothed_accuracy']):10.2f} "
            f"{100.0 * float(row['best_smoothed_accuracy']):10.2f} "
            f"{row['num_available_clients_mean']:7.2f}±{row['num_available_clients_std']:<6.2f} "
            f"{row['num_selected_clients_mean']:7.2f}±{row['num_selected_clients_std']:<6.2f} "
            f"{row['upload_window_seconds_mean']:7.2f}±{row['upload_window_seconds_std']:<6.2f} "
            f"{row['total_energy_joules'] / 1000.0:12.2f} "
            f"{row['total_communication_bytes'] / 1e9:11.2f} "
            f"{int(row['over_budget_rounds']):8d}"
        )

    return rows


# =========================
# Main
# =========================

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # General learning curves.
    plot_eval_metric(
        metric_name="fed_eval_loss",
        y_label="Evaluation loss",
        output_name="fed_eval_loss_vs_round",
    )

    plot_eval_metric(
        metric_name="fed_eval_acc",
        y_label="Evaluation accuracy",
        output_name="fed_eval_acc_vs_round",
    )

    # Core computation-time variability diagnostics.
    plot_simple_derived_metric(
        derived_key="num_available_clients",
        y_label="Available clients",
        output_name="available_clients_vs_round",
        smooth=False,
    )

    plot_simple_derived_metric(
        derived_key="available_fraction",
        y_label="Available client fraction",
        output_name="available_fraction_vs_round",
        smooth=False,
    )

    plot_simple_derived_metric(
        derived_key="upload_window",
        y_label="Upload window (s)",
        output_name="upload_window_vs_round",
        smooth=False,
    )

    plot_mean_std_derived_metric(
        mean_key="computation_time_mean",
        std_key="computation_time_std",
        y_label="Computation time across clients (s)",
        output_name="mean_std_computation_time_vs_round",
    )

    # General RAOCS diagnostics that remain important.
    plot_simple_derived_metric(
        derived_key="num_selected_clients",
        y_label="Selected clients",
        output_name="selected_clients_vs_round",
        smooth=False,
    )

    plot_simple_derived_metric(
        derived_key="used_resource_blocks",
        y_label="Used resource blocks",
        output_name="used_resource_blocks_vs_round",
        smooth=False,
    )

    plot_simple_derived_metric(
        derived_key="round_energy_joules",
        y_label="Round energy (J)",
        output_name="round_energy_vs_round",
        smooth=True,
    )

    plot_simple_derived_metric(
        derived_key="communication_bytes",
        y_label="Round communication (bytes)",
        output_name="round_communication_vs_round",
        smooth=True,
    )

    plot_mean_std_derived_metric(
        mean_key="selected_upload_time_mean",
        std_key="selected_upload_time_std",
        y_label="Selected-client upload time (s)",
        output_name="mean_std_selected_upload_time_vs_round",
        y_log=False,
    )

    # Save statistics before creating summary-over-sigma figures.
    summary_rows = save_summary_statistics()

    # Summary-over-sigma figures for paper/analysis.
    plot_summary_metric_vs_sigma(
        summary_rows,
        metric_key="final_smoothed_accuracy",
        y_label="Final accuracy (%)",
        output_name="final_accuracy_vs_sigma",
        value_multiplier=100.0,
    )

    plot_summary_metric_vs_sigma(
        summary_rows,
        metric_key="best_smoothed_accuracy",
        y_label="Best accuracy (%)",
        output_name="best_accuracy_vs_sigma",
        value_multiplier=100.0,
    )

    plot_summary_mean_std_vs_sigma(
        summary_rows,
        mean_key="num_available_clients_mean",
        std_key="num_available_clients_std",
        y_label="Available clients",
        output_name="available_clients_vs_sigma",
    )

    plot_summary_mean_std_vs_sigma(
        summary_rows,
        mean_key="num_selected_clients_mean",
        std_key="num_selected_clients_std",
        y_label="Selected clients",
        output_name="selected_clients_vs_sigma",
    )

    plot_summary_mean_std_vs_sigma(
        summary_rows,
        mean_key="upload_window_seconds_mean",
        std_key="upload_window_seconds_std",
        y_label="Upload window (s)",
        output_name="upload_window_vs_sigma",
    )

    plot_summary_metric_vs_sigma(
        summary_rows,
        metric_key="total_energy_joules",
        y_label="Total energy (kJ)",
        output_name="total_energy_vs_sigma",
        value_multiplier=1.0 / 1000.0,
    )

    plot_summary_metric_vs_sigma(
        summary_rows,
        metric_key="total_communication_bytes",
        y_label="Total communication (GB)",
        output_name="total_communication_vs_sigma",
        value_multiplier=1.0 / 1e9,
    )

    # Per-client diagnostics are mainly for debugging physical-system behavior.
    if PLOT_PER_CLIENT_DIAGNOSTICS:
        plot_per_client_diagnostics(
            metric_name="computation_times_seconds",
            y_label="Computation time (s)",
        )

        plot_per_client_diagnostics(
            metric_name="availability_indicator",
            y_label="Availability indicator",
        )

        plot_per_client_diagnostics(
            metric_name="uplink_rates_per_resource_block_bps",
            y_label="Uplink rate per RB (bps)",
        )

        plot_per_client_diagnostics(
            metric_name="assigned_resource_blocks",
            y_label="Assigned resource blocks",
        )

        plot_per_client_diagnostics(
            metric_name="upload_times_seconds",
            y_label="Upload time (s)",
        )

    print(f"Plots saved in: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()