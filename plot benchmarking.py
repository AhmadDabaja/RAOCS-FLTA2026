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

X_AXIS = "time"  # "round" or "time"

MAX_ROUNDS: int | None = 800

# Log-scale options
USE_LOG_TIME_AXIS = True
USE_LOG_ROUND_TIME_Y_AXIS = True
USE_LOG_UPLOAD_TIME_Y_AXIS = True

# SMOOTHING_ALPHA = 0.0 means no smoothing.
SMOOTHING_ALPHA = 0.75

LOG_EPS = 1e-12

PLOT_PER_CLIENT_DIAGNOSTICS = False


# EXPERIMENTS = [
#     {
#         "path": "FedAvg/results/FedAvg(FCN-Fashion, 500_rounds, 5_RBs, 5_sec, seed_42).json",
#         "label": "FedAvg",
#     },
#     {
#         "path": "UCS/results/UCS(FCN-MNIST, 500_rounds, 10%_sampling, 5_RBs, 5_sec, seed_42).json",
#         "label": "UCS (p=10%)",
#     },
#     # {
#     #     "path": "UCS/results/UCS(FCN-MNIST, 500_rounds, 20%_sampling, 5_RBs, 5_sec, seed_42).json",
#     #     "label": "Uniform Sampling (p=20%)",
#     # },
#     {
#         "path": "OCS/results/OCS(FCN-MNIST, 500_rounds, 5_clients,  5_RBs, 5_sec, seed_42).json",
#         "label": "OCS (m=5)",
#     },
#     # {
#     #     "path": "OCS/results/OCS(FCN-MNIST, 500_rounds, 10_clients, 5_RBs, 5_sec, seed_42).json",
#     #     "label": "OCS (m=10)",
#     # },
#     {
#         "path": "RAGreedySampling/results/RAGreedySampling(FCN-MNIST, 500_rounds, 5_RBs, 5_sec, seed_42).json",
#         "label": "RA-Greedy",
#     },
#     {
#         "path": "RAOCS4WFL/results/RAOCS4WFL(FCN-MNIST, 500_rounds, 5_RBs, 5_sec, seed_42).json",
#         "label": "RAOCS",
#     },
#     # {
#     #     "path": "RAOCSGreedy/results/RAOCSGreedy(FCN-MNIST, 500_rounds, 5_RBs, 5_sec, seed_42).json",
#     #     "label": "RA-Greedy-OCS (1/p_real)",
#     # },
#     {
#         "path": "RAOCSGreedy/results/RAOCSGreedy(FCN-MNIST, 500_rounds, 5_RBs, 5_sec, seed_42)*.json",
#         "label": "RAOCS-Greedy",
#     },
# ]
#
# PLOTS_DIR = Path("plots (FCN-MNIST)") / "baseline_comparison"


# EXPERIMENTS = [
#     {
#         "path": "FedAvg/results/FedAvg(ResNet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42).json",
#         "label": "FedAvg",
#     },
#     {
#         "path": "UCS/results/UCS(ResNet8-CIFAR10, 800_rounds, 10%_sampling, 5_RBs, 1.5_sec, seed_42).json",
#         "label": "UCS (p=10%)",
#     },
#     # {
#     #     "path": "UCS/results/UCS(ResNet8-CIFAR10, 800_rounds, 20%_sampling, 5_RBs, 1.5_sec, seed_42).json",
#     #     "label": "Uniform Sampling (p=20%)",
#     # },
#     {
#         "path": "OCS/results/OCS(ResNet8-CIFAR10, 800_rounds, 5_clients, 5_RBs, 1.5_sec, seed_42).json",
#         "label": "OCS (m=5)",
#     },
#     # {
#     #     "path": "OCS/results/OCS(ResNet8-CIFAR10, 800_rounds, 10_clients, 5_RBs, 1.5_sec, seed_42).json",
#     #     "label": "OCS (m=10)",
#     # },
#     {
#         "path": "RAGreedySampling/results/RAGreedySampling(Resnet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42).json",
#         "label": "RA-Greedy",
#     },
#     {
#         "path": "RAOCS4WFL/results/RAOCS4WFL(ResNet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42).json",
#         "label": "RAOCS",
#     },
#     # {
#     #     "path": "RAOCSGreedy/results/RAOCSGreedy(ResNet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42).json",
#     #     "label": "RA-Greedy-OCS (1/p_real)",
#     # },
#     {
#         "path": "RAOCSGreedy/results/RAOCSGreedy(ResNet8-CIFAR10, 800_rounds, 5_RBs, 1.5_sec, seed_42)*.json",
#         "label": "RAOCS-Greedy",
#     },
# ]
#
# PLOTS_DIR = Path("plots (ResNet8-CIFAR10)") / "baseline_comparison"


EXPERIMENTS = [
    {
        "path": "FedAvg/results/FedAvg(ResNet20-CIFAR100, 1200_rounds, 7_RBs, 4_sec, seed_42).json",
        "label": "FedAvg",
    },
    {
        "path": "UCS/results/UCS(ResNet20-CIFAR100, 1200_rounds, 10%_sampling, 7_RBs, 4_sec, seed_42).json",
        "label": "UCS (p=10%)",
    },
    {
        "path": "OCS/results/OCS(ResNet20-CIFAR100, 1200_rounds, 5_clients, 7_RBs, 4_sec, seed_42).json",
        "label": "OCS (m=5)",
    },
    {
        "path": "RAGreedySampling/results/RAGreedySampling(ResNet20-CIFAR100, 1200_rounds, 7_RBs, 4_sec, seed_42).json",
        "label": "RA-Greedy",
    },
    {
        "path": "RAOCS4WFL/results/RAOCS4WFL(ResNet20-CIFAR100, 1200_rounds, 7_RBs, 4_sec, seed_42).json",
        "label": "RAOCS",
    },
    {
        "path": "RAOCSGreedy/results/RAOCSGreedy(ResNet20-CIFAR100, 1200_rounds, 7_RBs, 4_sec, seed_42).json",
        "label": "RAOCS-Greedy",
    },
]

PLOTS_DIR = Path("plots (ResNet20-CIFAR100)") / "baseline_comparison"


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

EXTENSION = 'PDF'       # 'PDF' or 'PNG'


# =========================
# Loading utilities
# =========================

def load_result(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "metrics" not in result:
        raise KeyError("Result file must contain a 'metrics' field.")

    metrics = result["metrics"]

    if not isinstance(metrics, list):
        raise TypeError("'metrics' must be a list.")

    return metrics


def limit_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Limit metrics to the first MAX_ROUNDS entries.

    If MAX_ROUNDS is None, all rounds are used normally.
    """
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


# =========================
# Derived metric computation
# =========================

def compute_round_derived_metrics(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute upload-only derived metrics.

    Round time is the maximum upload time among selected clients:
        max_k Z_k / (nu_k b_k)
    """
    selected_clients = [int(client_id) for client_id in row.get("selected_clients", [])]

    uplink_rates = row["uplink_rates_per_resource_block_bps"]
    assigned_resource_blocks = row.get("assigned_resource_blocks", {})
    upload_bytes = row.get("upload_bytes", {})

    upload_times_selected: dict[int, float] = {}

    for client_id in selected_clients:
        key = client_key(client_id)

        b_k = assigned_resource_blocks.get(key)
        upload_k = upload_bytes.get(key)
        nu_k = uplink_rates[client_id]

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

    valid_upload_times = [
        value for value in upload_times_selected.values()
        if not math.isnan(value)
    ]

    round_time = max(valid_upload_times) if valid_upload_times else 0.0

    used_resource_blocks = sum(
        float(value)
        for value in assigned_resource_blocks.values()
        if value is not None
    )

    selected_upload_times = [
        value for value in upload_times_selected.values()
        if not math.isnan(value)
    ]

    return {
        "round_time": round_time,
        "used_resource_blocks": used_resource_blocks,
        "num_selected_clients": len(selected_clients),
        "upload_times_selected": upload_times_selected,
        "selected_upload_times": selected_upload_times,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    clean_values = [float(v) for v in values if not math.isnan(float(v))]

    if not clean_values:
        return math.nan, math.nan

    mean_value = sum(clean_values) / len(clean_values)

    variance = sum((v - mean_value) ** 2 for v in clean_values) / len(clean_values)
    std_value = math.sqrt(variance)

    return mean_value, std_value


def exponential_moving_average(
    values: list[float],
    smoothing_alpha: float,
) -> list[float]:
    """
    Apply exponential moving average smoothing.

    Convention:
        smoothed_t = (1 - alpha) * value_t + alpha * smoothed_{t-1}

    Therefore:
        alpha = 0.0 gives no smoothing.
        alpha close to 1.0 gives stronger smoothing.
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


def percentile(values: list[float], q: float) -> float:
    """
    Compute percentile using linear interpolation.

    q must be in [0, 1].
    """
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


def summarize_distribution(
    values: list[float],
    prefix: str,
) -> dict[str, float]:
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

    mean_value = sum(clean_values) / len(clean_values)
    variance = sum((v - mean_value) ** 2 for v in clean_values) / len(clean_values)

    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": math.sqrt(variance),
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


def compute_all_derived_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = get_limited_metrics(result)

    return [
        compute_round_derived_metrics(row)
        for row in metrics
    ]


# =========================
# Axis utilities
# =========================

def accumulated_time_axis(derived_metrics: list[dict[str, Any]]) -> list[float]:
    cumulative_time = 0.0
    x_values = []

    for row in derived_metrics:
        cumulative_time += float(row["round_time"])
        x_values.append(cumulative_time)

    return x_values


def get_x_values(
    metrics: list[dict[str, Any]],
    derived_metrics: list[dict[str, Any]],
) -> list[float]:
    if X_AXIS == "round":
        return round_axis(metrics)

    if X_AXIS == "time":
        return accumulated_time_axis(derived_metrics)

    raise ValueError("X_AXIS must be either 'round' or 'time'.")


def get_x_label() -> str:
    if X_AXIS == "round":
        return "Round"

    if X_AXIS == "time":
        return "Accumulated upload round time (s)"

    raise ValueError("X_AXIS must be either 'round' or 'time'.")

def apply_log_scales(
    x_log: bool = False,
    y_log: bool = False,
) -> None:
    """
    Apply log scale safely.

    We use log scale mainly for time quantities that can become very large
    because of weak uplink rates.
    """
    if x_log:
        plt.xscale("log")

    if y_log:
        plt.yscale("log")

# =========================
# Global plotting utilities
# =========================

def plot_eval_metric(metric_name: str, y_label: str, output_name: str) -> None:
    plt.figure(figsize=FIGURE_SIZE)

    for experiment in EXPERIMENTS:
        result = load_result(experiment["path"])
        metrics = get_limited_metrics(result)
        derived_metrics = compute_all_derived_metrics(result)

        x_values = get_x_values(metrics, derived_metrics)
        y_values = [float(row[metric_name]) for row in metrics]

        smoothed_y_values = exponential_moving_average(
            values=y_values,
            smoothing_alpha=SMOOTHING_ALPHA,
        )

        plt.plot(
            x_values,
            smoothed_y_values,
            marker="o",
            label=experiment["label"],
        )

    apply_log_scales(
        x_log=(X_AXIS == "time" and USE_LOG_TIME_AXIS),
        y_log=False,
    )

    plt.xlabel(get_x_label())
    plt.ylabel(y_label)

    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / output_name

    if EXTENSION.upper() == "PNG":
        plt.savefig(png_path, dpi=600, bbox_inches="tight")
    else:
        pdf_path = PLOTS_DIR / output_name.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()


def plot_simple_derived_metric(
    derived_key: str,
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
        y_values = [float(row[derived_key]) for row in derived_metrics]

        plt.plot(x_values, y_values, marker="o", label=experiment["label"])

    apply_log_scales(y_log=y_log)

    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / output_name

    if EXTENSION.upper() == "PNG":
        plt.savefig(png_path, dpi=600, bbox_inches="tight")
    else:
        pdf_path = PLOTS_DIR / output_name.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()


def plot_mean_std_derived_metric(
    selected_values_key: str,
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

        means = []
        lower = []
        upper = []

        for row in derived_metrics:
            mean_value, std_value = mean_std(row[selected_values_key])

            means.append(mean_value)

            if math.isnan(mean_value):
                lower.append(math.nan)
                upper.append(math.nan)
            else:
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

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / output_name

    if EXTENSION.upper() == "PNG":
        plt.savefig(png_path, dpi=600, bbox_inches="tight")
    else:
        pdf_path = PLOTS_DIR / output_name.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")

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
        key = client_key(client_id)

        if metric_name == "uplink_rates_per_resource_block_bps":
            values.append(float(row["uplink_rates_per_resource_block_bps"][client_id]))

        elif metric_name == "assigned_resource_blocks":
            if client_id in selected_clients:
                values.append(float(row["assigned_resource_blocks"].get(key, math.nan)))
            else:
                values.append(math.nan)

        elif metric_name == "upload_times_seconds":
            if client_id in selected_clients:
                values.append(
                    float(derived_row["upload_times_selected"].get(client_id, math.nan))
                )
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

        if EXTENSION.upper() == 'PNG':
            png_path = output_dir / f"client_{client_id}.png"
            plt.savefig(png_path, dpi=600, bbox_inches="tight")
        else:
            pdf_path = output_dir / f"client_{client_id}.pdf"
            plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()


def compute_experiment_summary(
    experiment: dict[str, str],
) -> dict[str, Any]:
    result = load_result(experiment["path"])
    metrics = get_limited_metrics(result)
    derived_metrics = compute_all_derived_metrics(result)

    config = result.get("config", {})

    raw_acc = [float(row["fed_eval_acc"]) for row in metrics]
    raw_loss = [float(row["fed_eval_loss"]) for row in metrics]

    smoothed_acc = exponential_moving_average(
        values=raw_acc,
        smoothing_alpha=SMOOTHING_ALPHA,
    )
    smoothed_loss = exponential_moving_average(
        values=raw_loss,
        smoothing_alpha=SMOOTHING_ALPHA,
    )

    round_times = [
        float(row["round_time"])
        for row in derived_metrics
    ]

    used_resource_blocks = [
        float(row["used_resource_blocks"])
        for row in derived_metrics
    ]

    num_selected_clients = [
        float(row["num_selected_clients"])
        for row in derived_metrics
    ]

    mean_selected_upload_times = []
    std_selected_upload_times = []

    for row in derived_metrics:
        mean_value, std_value = mean_std(row["selected_upload_times"])
        mean_selected_upload_times.append(mean_value)
        std_selected_upload_times.append(std_value)

    mean_selected_train_acc = []

    for row in metrics:
        train_acc_values = [
            float(value)
            for value in row.get("train_acc", {}).values()
            if value is not None
        ]

        if train_acc_values:
            mean_selected_train_acc.append(
                sum(train_acc_values) / len(train_acc_values)
            )
        else:
            mean_selected_train_acc.append(math.nan)

    resource_block_budget = float(config.get("resource_block_budget", math.nan))

    if math.isfinite(resource_block_budget):
        over_budget_rounds = sum(
            1
            for value in used_resource_blocks
            if math.isfinite(value) and value > resource_block_budget + 1e-9
        )
    else:
        over_budget_rounds = math.nan

    summary: dict[str, Any] = {
        "scheme": experiment["label"],
        "path": experiment["path"],
        "num_rounds": len(metrics),
        "smoothing_alpha": SMOOTHING_ALPHA,

        "final_raw_accuracy": last_finite(raw_acc),
        "final_smoothed_accuracy": last_finite(smoothed_acc),
        "best_raw_accuracy": max(clean_numeric_values(raw_acc)),
        "best_smoothed_accuracy": max(clean_numeric_values(smoothed_acc)),

        "final_raw_loss": last_finite(raw_loss),
        "final_smoothed_loss": last_finite(smoothed_loss),
        "min_raw_loss": min(clean_numeric_values(raw_loss)),
        "min_smoothed_loss": min(clean_numeric_values(smoothed_loss)),

        "final_mean_selected_train_accuracy": last_finite(mean_selected_train_acc),
        "mean_selected_train_accuracy": (
            sum(clean_numeric_values(mean_selected_train_acc))
            / len(clean_numeric_values(mean_selected_train_acc))
        ),

        "total_accumulated_upload_time_seconds": sum(
            clean_numeric_values(round_times)
        ),

        "resource_block_budget": resource_block_budget,
        "over_budget_rounds": over_budget_rounds,
    }

    summary.update(
        summarize_distribution(
            values=used_resource_blocks,
            prefix="used_resource_blocks",
        )
    )

    summary.update(
        summarize_distribution(
            values=round_times,
            prefix="max_selected_upload_time_seconds",
        )
    )

    summary.update(
        summarize_distribution(
            values=num_selected_clients,
            prefix="num_selected_clients",
        )
    )

    summary.update(
        summarize_distribution(
            values=mean_selected_upload_times,
            prefix="mean_selected_upload_time_seconds",
        )
    )

    summary.update(
        summarize_distribution(
            values=std_selected_upload_times,
            prefix="std_selected_upload_time_seconds",
        )
    )

    return summary


def save_summary_statistics() -> None:
    rows = [
        compute_experiment_summary(experiment)
        for experiment in EXPERIMENTS
    ]

    output_path = PLOTS_DIR / "summary_statistics.csv"

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary statistics saved in: {output_path.resolve()}")

    print(
        f"{'Scheme':35s} "
        f"{'Final acc':>10s} "
        f"{'Best acc':>10s} "
        f"{'Final loss':>12s} "
        f"{'Total time (s)':>15s} "
        f"{'Total time (h)':>15s} "
        f"{'Mean RB':>10s} "
        f"{'Std RB':>10s} "
        f"{'Over-B':>8s}"
    )

    for row in rows:
        total_time_seconds = row["total_accumulated_upload_time_seconds"]
        total_time_hours = total_time_seconds / 3600.0

        print(
            f"{row['scheme'][:35]:35s} "
            f"{row['final_smoothed_accuracy']:10.4f} "
            f"{row['best_smoothed_accuracy']:10.4f} "
            f"{row['final_smoothed_loss']:12.4f} "
            f"{total_time_seconds:15.4f} "
            f"{total_time_hours:15.4f} "
            f"{row['used_resource_blocks_mean']:10.4f} "
            f"{row['used_resource_blocks_std']:10.4f} "
            f"{int(row['over_budget_rounds']):8d}"
        )

# =========================
# Main
# =========================

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluation plots: round or accumulated time
    plot_eval_metric(
        metric_name="fed_eval_loss",
        y_label="Evaluation loss",
        output_name=f"fed_eval_loss_vs_{X_AXIS}.png",
    )

    plot_eval_metric(
        metric_name="fed_eval_acc",
        y_label="Evaluation accuracy",
        output_name=f"fed_eval_acc_vs_{X_AXIS}.png",
    )

    # Global diagnostics: always versus rounds
    plot_simple_derived_metric(
        derived_key="round_time",
        y_label="Maximum selected-client upload time (s)",
        output_name="max_selected_upload_time_vs_round.png",
        y_log=USE_LOG_ROUND_TIME_Y_AXIS,
    )

    plot_simple_derived_metric(
        derived_key="used_resource_blocks",
        y_label="Used resource blocks",
        output_name="used_resource_blocks_vs_round.png",
    )

    plot_simple_derived_metric(
        derived_key="num_selected_clients",
        y_label="Number of selected clients",
        output_name="num_selected_clients_vs_round.png",
    )

    plot_mean_std_derived_metric(
        selected_values_key="selected_upload_times",
        y_label="Selected-client upload time (s)",
        output_name="mean_std_selected_upload_time_vs_round.png",
        y_log=USE_LOG_UPLOAD_TIME_Y_AXIS,
    )

    # Per-client diagnostics are mainly for debugging physical-system behavior.
    # They are not always needed for the final comparison plots.
    if PLOT_PER_CLIENT_DIAGNOSTICS:
        plot_per_client_diagnostics(
            metric_name="uplink_rates_per_resource_block_bps",
            y_label="Uplink rate per resource block (bps)",
        )

        plot_per_client_diagnostics(
            metric_name="assigned_resource_blocks",
            y_label="Assigned resource blocks",
        )

        plot_per_client_diagnostics(
            metric_name="upload_times_seconds",
            y_label="Upload time (s)",
            y_log=USE_LOG_UPLOAD_TIME_Y_AXIS,
        )

    save_summary_statistics()

    print(f"Plots saved in: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()