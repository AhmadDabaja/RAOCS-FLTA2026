from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from config import SimulationConfig
from src.Client import Client
from src.Data import prepare_federated_data
from src.Models import FCNFashionModel, FCNCIFAR10Model, ResNet8CIFAR10Model, ResNet20CIFAR100Model
from src.Server import Server
from src.PhysicalSystem import PhysicalSystem


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_model(dataset_name: str, model_name: str):
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()

    if dataset_name == "fashionmnist" and model_name == "fcn-fashion":
        return FCNFashionModel()

    if dataset_name == "cifar10" and model_name == "fcn-cifar10":
        return FCNCIFAR10Model()

    if dataset_name == "cifar10" and model_name == "resnet8-cifar10":
        return ResNet8CIFAR10Model()

    if dataset_name == "cifar100" and model_name == "resnet20-cifar100":
        return ResNet20CIFAR100Model()

    raise ValueError(
        f"Unsupported dataset/model pair: {dataset_name}/{model_name}"
    )


def build_clients(config: SimulationConfig) -> list[Client]:
    """
    Create all simulated clients.

    Each client corresponds to one dataset partition.
    UCS later samples a subset of these clients at each round.
    """
    clients: list[Client] = []

    for client_id in range(config.num_partitions):
        client = Client(
            client_id=client_id,
            dataset_name=config.dataset_name,
            model_name=config.model_name,
            num_partitions=config.num_partitions,
            alpha=config.alpha,
            batch_size=config.batch_size,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
        clients.append(client)

    return clients


def main() -> None:
    """
    Run the end-to-end federated learning simulation.
    """
    config = SimulationConfig()
    config.validate()

    set_seed(config.seed)

    # Prepare and cache the federated dataset once before simulation starts.
    scheme_dir = Path(__file__).resolve().parent
    project_dir = scheme_dir.parent
    shared_datasets_dir = project_dir / "datasets"
    prepare_federated_data(
        dataset_name=config.dataset_name,
        num_partitions=config.num_partitions,
        alpha=config.alpha,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
        data_root=str(shared_datasets_dir)
    )

    clients = build_clients(config)

    global_model = build_model(
        dataset_name=config.dataset_name,
        model_name=config.model_name,
    )

    physical_system = PhysicalSystem(
        num_clients=config.num_partitions,
        seed=config.seed,
        wireless_config=config.wireless_config,
    )

    server = Server(
        clients=clients,
        model=global_model,
        num_rounds=config.num_server_rounds,
        learning_rate=config.learning_rate,
        local_epochs=config.local_epochs,
        results_dir=config.results_dir,
        run_name=config.run_name,
        config=config,
        physical_system=physical_system,
    )

    history = server.run()

    print("\nSimulation finished.")
    if history:
        print("Last round summary:")
        print(history[-1])


if __name__ == "__main__":
    main()