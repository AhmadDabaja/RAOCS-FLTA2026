from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from config import SimulationConfig
from src.Data import prepare_federated_data


# =============================================================================
# User settings
# =============================================================================

RESULT_JSON_PATH: str | None = None
AUTO_USE_LATEST_RESULT = True

USE_TRAIN_SPLIT_ONLY = True

CLIENT_IDS_TO_PLOT: list[int] | None = None

# Options:
#   "client_id"
#   "participation_desc"
#   "data_share_desc"
ORDER_CLIENTS_BY = "client_id"

OUTPUT_DIR = Path("./../plots (FCN-MNIST, Stressed)") / "client_participation"
OUTPUT_BASENAME = "RAOCS"

SAVE_CSV = True
SAVE_PDF = False
SAVE_PNG = True


# =============================================================================
# Visual controls
# =============================================================================

FIGURE_HEIGHT = 6.2
FIGURE_WIDTH_PER_CLIENT = 0.5
MIN_FIGURE_WIDTH = 13.0

# color for data share
DATA_SHARE_COLOR = "#A10B56"

# Group layout:
# Each client group is centered at x_group.
# The three lollipops are placed around that center.
TICK_UNITS_BETWEEN_GROUPS = 15

CLASS_TICK_OFFSET = -3
PARTICIPATION_TICK_OFFSET = 0
DATA_SHARE_TICK_OFFSET = 3

CLASS_LINE_WIDTH = 5
PARTICIPATION_LINE_WIDTH = 3.5
DATA_SHARE_LINE_WIDTH = 3.5

CLASS_ALPHA = 0.90
PARTICIPATION_ALPHA = 0.95
DATA_SHARE_ALPHA = 0.90

PARTICIPATION_DOT_SIZE = 38
DATA_SHARE_DOT_SIZE = 32

X_TICK_ROTATION = 90


# =============================================================================
# Optional class names
# =============================================================================

DATASET_CLASS_NAMES = {
    "fashionmnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}


# =============================================================================
# Result loading
# =============================================================================

def resolve_result_path(config: SimulationConfig) -> Path:
    if RESULT_JSON_PATH is not None:
        path = Path(RESULT_JSON_PATH)
        if not path.exists():
            raise FileNotFoundError(f"RESULT_JSON_PATH does not exist: {path}")
        return path

    configured_path = Path(config.results_dir) / f"{config.run_name}.json"
    if configured_path.exists():
        return configured_path

    if not AUTO_USE_LATEST_RESULT:
        raise FileNotFoundError(
            f"Could not find configured result file: {configured_path}"
        )

    results_dir = Path(config.results_dir)
    candidates = sorted(
        results_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No JSON result files found in {results_dir.resolve()}"
        )

    print("Warning: configured result file was not found.")
    print(f"Using latest JSON result instead: {candidates[0]}")
    return candidates[0]


def load_metrics(result_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "metrics" in data:
        metrics = data["metrics"]
        result_config = data.get("config", {})
    elif isinstance(data, list):
        metrics = data
        result_config = {}
    else:
        raise ValueError(
            "Unsupported result JSON format. Expected either "
            "{'config': ..., 'metrics': [...]} or a list of round dictionaries."
        )

    if not isinstance(metrics, list):
        raise ValueError("The 'metrics' field must be a list.")

    return metrics, result_config


def parse_client_id(value: Any) -> int:
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        match = re.search(r"(\d+)$", value)
        if match:
            return int(match.group(1))

    raise ValueError(f"Cannot parse client id from: {value}")


def get_selected_client_ids_from_round(row: dict[str, Any]) -> list[int]:
    if "selected_clients" in row and row["selected_clients"] is not None:
        return [parse_client_id(x) for x in row["selected_clients"]]

    # Fallback for result files that do not explicitly store selected_clients.
    # Usually upload_bytes / train_loss keys correspond to participating clients.
    for key in ["upload_bytes", "train_loss", "train_acc"]:
        if key in row and isinstance(row[key], dict) and row[key]:
            return [parse_client_id(x) for x in row[key].keys()]

    return []


def compute_participation_rate(
    metrics: list[dict[str, Any]],
    num_clients: int,
    full_participation_if_empty: bool = False,
) -> np.ndarray:
    counts = np.zeros(num_clients, dtype=float)

    for row in metrics:
        selected_ids = get_selected_client_ids_from_round(row)

        if not selected_ids and full_participation_if_empty:
            selected_ids = list(range(num_clients))

        for client_id in selected_ids:
            if 0 <= client_id < num_clients:
                counts[client_id] += 1.0

    num_rounds = max(1, len(metrics))
    return counts / num_rounds


# =============================================================================
# Data statistics
# =============================================================================

def get_client_indices(data_manager, client_id: int) -> list[int]:
    if USE_TRAIN_SPLIT_ONLY:
        return list(data_manager.client_splits[client_id]["train"])

    return list(data_manager.partition_to_indices[client_id])


def compute_data_statistics(
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scheme_dir = Path(__file__).resolve().parent
    project_dir = scheme_dir.parent
    shared_datasets_dir = project_dir / "datasets"

    data_manager = prepare_federated_data(
        dataset_name=config.dataset_name,
        num_partitions=config.num_partitions,
        alpha=config.alpha,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
        data_root=str(shared_datasets_dir),
    )

    labels = np.asarray(data_manager.labels)
    classes = np.unique(labels)
    num_classes = len(classes)

    class_counts = np.zeros((config.num_partitions, num_classes), dtype=float)
    sample_counts = np.zeros(config.num_partitions, dtype=float)

    class_to_col = {int(class_id): idx for idx, class_id in enumerate(classes)}

    for client_id in range(config.num_partitions):
        indices = get_client_indices(data_manager, client_id)
        sample_counts[client_id] = len(indices)

        if len(indices) == 0:
            continue

        client_labels = labels[indices]

        for class_id in client_labels:
            col = class_to_col[int(class_id)]
            class_counts[client_id, col] += 1.0

    class_percentages = np.zeros_like(class_counts, dtype=float)

    non_empty = sample_counts > 0
    class_percentages[non_empty] = (
        class_counts[non_empty] / sample_counts[non_empty, None]
    )

    total_samples = max(1.0, float(np.sum(sample_counts)))
    data_share = sample_counts / total_samples

    dataset_name = config.dataset_name.lower()
    if (
        dataset_name in DATASET_CLASS_NAMES
        and len(DATASET_CLASS_NAMES[dataset_name]) == num_classes
    ):
        class_labels = DATASET_CLASS_NAMES[dataset_name]
    else:
        class_labels = [f"Class {int(c)}" for c in classes]

    return class_percentages, data_share, class_labels


# =============================================================================
# Client ordering / export
# =============================================================================

def choose_client_order(
    participation_rate: np.ndarray,
    data_share: np.ndarray,
) -> list[int]:
    client_ids = list(range(len(participation_rate)))

    if CLIENT_IDS_TO_PLOT is not None:
        allowed = set(CLIENT_IDS_TO_PLOT)
        client_ids = [cid for cid in client_ids if cid in allowed]

    if ORDER_CLIENTS_BY == "client_id":
        return client_ids

    if ORDER_CLIENTS_BY == "participation_desc":
        return sorted(client_ids, key=lambda cid: participation_rate[cid], reverse=True)

    if ORDER_CLIENTS_BY == "data_share_desc":
        return sorted(client_ids, key=lambda cid: data_share[cid], reverse=True)

    raise ValueError(
        "ORDER_CLIENTS_BY must be one of: "
        "'client_id', 'participation_desc', 'data_share_desc'."
    )


def save_statistics_csv(
    output_path: Path,
    client_ids: list[int],
    class_percentages: np.ndarray,
    data_share: np.ndarray,
    participation_rate: np.ndarray,
    class_labels: list[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["client_id", "participation_rate", "data_share"]
        header += [f"class_pct_{label}" for label in class_labels]
        writer.writerow(header)

        for cid in client_ids:
            row = [
                cid,
                float(participation_rate[cid]),
                float(data_share[cid]),
            ]
            row += [float(v) for v in class_percentages[cid]]
            writer.writerow(row)


# =============================================================================
# Plotting
# =============================================================================

def plot_client_lollipop(
    class_percentages: np.ndarray,
    participation_rate: np.ndarray,
    data_share: np.ndarray,
    class_labels: list[str],
    client_ids: list[int],
    output_dir: Path,
    title: str,
) -> None:
    selected_class_percentages = class_percentages[client_ids]
    selected_participation_rate = participation_rate[client_ids]
    selected_data_share = data_share[client_ids]

    num_clients = len(client_ids)
    num_classes = selected_class_percentages.shape[1]

    x_group = np.arange(num_clients) * TICK_UNITS_BETWEEN_GROUPS

    x_classes = x_group + CLASS_TICK_OFFSET
    x_participation = x_group + PARTICIPATION_TICK_OFFSET
    x_data_share = x_group + DATA_SHARE_TICK_OFFSET

    client_labels = [f"C{cid}" for cid in client_ids]

    figure_width = max(MIN_FIGURE_WIDTH, FIGURE_WIDTH_PER_CLIENT * num_clients)
    fig, ax_left = plt.subplots(figsize=(figure_width, FIGURE_HEIGHT))
    ax_right = ax_left.twinx()

    cmap_name = "tab10" if num_classes <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(num_classes)]

    # ------------------------------------------------------------------
    # Stacked colored lollipop: class distribution
    # ------------------------------------------------------------------
    bottom = np.zeros(num_clients, dtype=float)

    for class_idx in range(num_classes):
        top = bottom + selected_class_percentages[:, class_idx]

        ax_left.vlines(
            x_classes,
            bottom,
            top,
            linewidth=CLASS_LINE_WIDTH,
            color=colors[class_idx],
            alpha=CLASS_ALPHA,
            zorder=2,
        )

        bottom = top

    # No class-total dot: class distribution sums to 1 by construction.

    # ------------------------------------------------------------------
    # Black lollipop: real participation rate
    # Same axis as class distribution because both are in [0, 1].
    # ------------------------------------------------------------------
    ax_left.vlines(
        x_participation,
        0,
        selected_participation_rate,
        linewidth=PARTICIPATION_LINE_WIDTH,
        color="black",
        alpha=PARTICIPATION_ALPHA,
        zorder=3,
    )

    ax_left.scatter(
        x_participation,
        selected_participation_rate,
        color="black",
        s=PARTICIPATION_DOT_SIZE,
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Dark lollipop: data share
    # Separate right axis because values are usually around 1/K.
    # ------------------------------------------------------------------
    ax_right.vlines(
        x_data_share,
        0,
        selected_data_share,
        linewidth=DATA_SHARE_LINE_WIDTH,
        color=DATA_SHARE_COLOR,
        alpha=DATA_SHARE_ALPHA,
        zorder=1,
    )

    ax_right.scatter(
        x_data_share,
        selected_data_share,
        color=DATA_SHARE_COLOR,
        alpha=DATA_SHARE_ALPHA,
        s=DATA_SHARE_DOT_SIZE,
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax_left.set_ylim(0.0, 1.0)

    max_data_share = float(np.max(selected_data_share)) if num_clients > 0 else 0.0
    ax_right.set_ylim(0.0, max(0.01, max_data_share * 1.20))

    ax_left.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax_left.set_xticks(x_group)
    ax_left.set_xticklabels(client_labels, rotation=X_TICK_ROTATION)

    if num_clients > 0:
        left_margin = TICK_UNITS_BETWEEN_GROUPS * 0.70
        right_margin = TICK_UNITS_BETWEEN_GROUPS * 0.70
        ax_left.set_xlim(x_group[0] - left_margin, x_group[-1] + right_margin)

    ax_left.set_xlabel("Clients")
    ax_left.set_ylabel("Class distribution / Participation rate")
    ax_right.set_ylabel("Data share")

    ax_left.set_title(title)

    ax_left.grid(axis="y", which="major", alpha=0.30)
    ax_left.grid(axis="y", which="minor", alpha=0.15)
    ax_left.minorticks_on()

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------
    class_handles = [
        Patch(
            facecolor=colors[j],
            edgecolor="none",
            alpha=CLASS_ALPHA,
            label=class_labels[j],
        )
        for j in range(num_classes)
    ]

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linewidth=PARTICIPATION_LINE_WIDTH,
            markersize=6,
            label="Participation rate",
        ),
        Line2D(
            [0],
            [0],
            color=DATA_SHARE_COLOR,
            marker="o",
            linewidth=DATA_SHARE_LINE_WIDTH,
            markersize=6,
            alpha=DATA_SHARE_ALPHA,
            label="Data share",
        ),
    ]

    legend_classes = ax_left.legend(
        handles=class_handles,
        title="Classes",
        loc="upper left",
        bbox_to_anchor=(1.08, 1.00),
    )

    ax_left.add_artist(legend_classes)

    ax_left.legend(
        handles=metric_handles,
        title="Client metrics",
        loc="upper left",
        bbox_to_anchor=(1.08, 0.25),
    )

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(output_dir / f"{OUTPUT_BASENAME}.png", dpi=300, bbox_inches="tight")

    if SAVE_PDF:
        fig.savefig(output_dir / f"{OUTPUT_BASENAME}.pdf", bbox_inches="tight")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    config = SimulationConfig()
    config.validate()

    result_path = resolve_result_path(config)
    metrics, _ = load_metrics(result_path)

    participation_rate = compute_participation_rate(
        metrics=metrics,
        num_clients=config.num_partitions,
        full_participation_if_empty=False,
    )

    class_percentages, data_share, class_labels = compute_data_statistics(config)

    client_ids = choose_client_order(
        participation_rate=participation_rate,
        data_share=data_share,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_CSV:
        save_statistics_csv(
            output_path=OUTPUT_DIR / f"{OUTPUT_BASENAME}.csv",
            client_ids=client_ids,
            class_percentages=class_percentages,
            data_share=data_share,
            participation_rate=participation_rate,
            class_labels=class_labels,
        )

    scheme_name = Path(__file__).resolve().parent.name
    title = (
        f"{scheme_name}: client class distribution, "
        f"participation rate, and data share"
    )

    plot_client_lollipop(
        class_percentages=class_percentages,
        participation_rate=participation_rate,
        data_share=data_share,
        class_labels=class_labels,
        client_ids=client_ids,
        output_dir=OUTPUT_DIR,
        title=title,
    )

    print("Done.")
    print(f"Result file used: {result_path.resolve()}")
    print(f"Number of rounds: {len(metrics)}")
    print(f"Average participation: {participation_rate.sum():.4f} clients per round")
    print(f"Plot directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()