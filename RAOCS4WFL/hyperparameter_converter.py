from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =========================
# User configuration
# =========================

RESULT_FILE_NAME = "RAOCS4WFL(FCN-MNIST, 1000_rounds, 5_RBs, 5_sec, seed_42).json"


# =========================
# Loading utilities
# =========================

def load_result(path: str | Path) -> dict[str, Any]:
    """
    Load one RAOCS4WFL result JSON file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract the metrics list from the result file.
    """
    if "metrics" not in result:
        raise KeyError("Result file must contain a 'metrics' field.")

    metrics = result["metrics"]

    if not isinstance(metrics, list):
        raise TypeError("'metrics' must be a list.")

    if len(metrics) == 0:
        raise ValueError("'metrics' is empty.")

    return metrics


def get_num_clients(result: dict[str, Any]) -> int:
    """
    Get the total number of clients K from the saved config.
    """
    if "config" not in result:
        raise KeyError("Result file must contain a 'config' field.")

    config = result["config"]

    if "num_partitions" not in config:
        raise KeyError("Saved config must contain 'num_partitions'.")

    return int(config["num_partitions"])


# =========================
# Conversion logic
# =========================

def compute_average_selected_clients(
    metrics: list[dict[str, Any]],
) -> tuple[float, list[int]]:
    """
    Compute the average number of selected clients per round.

    m = average_t |A_t|
    """
    selected_counts = []

    for row in metrics:
        selected_clients = row.get("selected_clients", [])
        selected_counts.append(len(selected_clients))

    m = sum(selected_counts) / len(selected_counts)

    return m, selected_counts


def main() -> None:
    """
    Hyperparameter conversion according to Prof. Rachid's idea.

    We first run RAOCS4WFL.
    Then we compute the average number of selected clients per round:

        m = average_t |A_t|

    Then for UCS:

        p = m / K

    where K is the total number of clients.
    """
    scheme_dir = Path(__file__).resolve().parent
    result_path = scheme_dir / "results" / RESULT_FILE_NAME

    result = load_result(result_path)
    metrics = get_metrics(result)

    K = get_num_clients(result)

    m, selected_counts = compute_average_selected_clients(metrics)

    p = m / K

    print("\n====================================================")
    print("Rachid-style hyperparameter conversion")
    print("====================================================")
    print(f"Result file: {result_path}")
    print(f"Total clients K: {K}")
    print(f"Number of rounds: {len(metrics)}")

    print("\nSelected clients per round:")
    print(selected_counts)

    print("\nEquivalent hyperparameters:")
    print(f"  m = average selected clients = {m:.4f}")
    print(f"  p = m / K = {p:.6f}")

    print("\nSuggested config values:")

    print("\nUCS:")
    print(f"  sampling_fraction = {p:.6f}")

    print("\nOCS:")
    print(f"  expected_num_sampled_clients = {m:.4f}")

    print("\nNote:")
    print(
        "This is client-count matching. It matches UCS/OCS to the average "
        "number of clients selected by RAOCS4WFL."
    )


if __name__ == "__main__":
    main()