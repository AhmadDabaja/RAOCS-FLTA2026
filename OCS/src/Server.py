from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union
from dataclasses import asdict, is_dataclass

import numpy as np
import torch

from .Client import Client
from .Message import Message
from .Models import FCNFashionModel, FCNCIFAR10Model, ResNet8CIFAR10Model, ResNet20CIFAR100Model
from .PhysicalSystem import PhysicalSystem


class Server:
    """
    Central server coordinating the federated learning process.

    The server:
      - computes OCS sampling probabilities,
      - samples clients from these probabilities,
      - sends the current global model to the sampled clients,
      - collects local updates,
      - aggregates updates using global data weights and inverse-probability correction,
      - sends the aggregated model for evaluation,
      - stores round history,
      - saves history and final model at the end.
    """

    def __init__(
        self,
        clients: Sequence[Client],
        model: Union[FCNFashionModel, FCNCIFAR10Model, ResNet8CIFAR10Model, ResNet20CIFAR100Model],
        num_rounds: int,
        learning_rate: float,
        local_epochs: int,
        results_dir: str | Path = "results",
        run_name: str = "simulation",
        server_id: str = "server",
        config: object | None = None,
        physical_system: PhysicalSystem | None = None,
    ) -> None:
        self.clients = list(clients)
        self.global_model = model
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.server_id = server_id
        self.run_name = run_name

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.config = asdict(config) if is_dataclass(config) else config
        self.communication_budget = (
            None if self.config is None else self.config.get("communication_budget")
        )

        self.used_communication_bytes = 0
        self.completed_rounds = 0

        self.physical_system = physical_system
        self.current_physical_state = None

        self.resource_block_budget = (
            len(self.clients)
            if self.config is None
            else int(self.config.get("resource_block_budget", len(self.clients)))
        )

        self.num_sampled_clients = (
            len(self.clients)
            if self.config is None
            else float(self.config.get("num_sampled_clients", len(self.clients)))
        )

        self.p_min = (
            0.1
            if self.config is None
            else float(self.config.get("p_min", 0.1))
        )

        self.probe_update_norm_initialization = (
            True
            if self.config is None
            else bool(self.config.get("probe_update_norm_initialization", True))
        )

        self.update_norm_mode = (
            "newest"
            if self.config is None
            else str(self.config.get("update_norm_mode", "newest")).lower()
        )

        if self.update_norm_mode not in {"holt", "newest", "real"}:
            raise ValueError("update_norm_mode must be one of: 'holt', 'newest', 'real'.")

        self.holt_rho = (
            0.5
            if self.config is None
            else float(self.config.get("holt_rho", 0.5))
        )

        self.holt_beta = (
            0.3
            if self.config is None
            else float(self.config.get("holt_beta", 0.3))
        )

        self.holt_gamma = (
            0.9
            if self.config is None
            else float(self.config.get("holt_gamma", 0.9))
        )

        self.holt_epsilon = (
            1e-12
            if self.config is None
            else float(self.config.get("holt_epsilon", 1e-12))
        )

        seed = (
            None if self.config is None else self.config.get("seed", None)
        )

        self.sampling_rng = np.random.default_rng(seed)

        # Client data weights alpha_k = n_k / n
        self.client_num_examples = np.array(
            [client.num_train_examples() for client in self.clients],
            dtype=float,
        )
        self.client_weights = self.client_num_examples / np.sum(self.client_num_examples)

        # Initial estimated update norm: sqrt(number of model parameters)
        num_model_parameters = sum(
            param.size for param in self.global_model.get_model_parameters()
        )
        initial_update_norm = np.sqrt(float(num_model_parameters))

        self.estimated_update_norms = np.full(
            len(self.clients),
            initial_update_norm,
            dtype=float,
        )

        self.holt_levels = np.log(self.estimated_update_norms + self.holt_epsilon)
        self.holt_trends = np.zeros(len(self.clients), dtype=float)

        self.history: List[dict] = []


    def compute_ocs_probabilities(self) -> np.ndarray:
        """
        Compute OCS probabilities using capped water-filling.

        Scores:
            a_k = alpha_k * estimated_update_norm_k

        Constraint:
            sum_k p_k = m
            0 < p_k <= 1
        """
        num_clients = len(self.clients)

        if num_clients == 0:
            return np.array([], dtype=float)

        m = self.num_sampled_clients

        scores = self.client_weights * self.estimated_update_norms
        scores = np.maximum(scores, 1e-12)

        probabilities = np.zeros(num_clients, dtype=float)

        remaining = np.ones(num_clients, dtype=bool)
        remaining_budget = m

        while True:
            remaining_indices = np.where(remaining)[0]

            if len(remaining_indices) == 0:
                break

            denominator = np.sum(scores[remaining_indices])

            if denominator <= 0:
                probabilities[remaining_indices] = remaining_budget / len(remaining_indices)
                break

            scale = remaining_budget / denominator
            candidate_probabilities = scale * scores[remaining_indices]

            saturated_mask = candidate_probabilities > 1.0

            if not np.any(saturated_mask):
                probabilities[remaining_indices] = candidate_probabilities
                break

            saturated_indices = remaining_indices[saturated_mask]
            probabilities[saturated_indices] = 1.0
            remaining[saturated_indices] = False

            remaining_budget = m - np.sum(probabilities[~remaining])

        probabilities = np.clip(probabilities, 0.0, 1.0)

        return probabilities

    def sample_clients_from_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> List[Client]:
        """
        Sample clients independently using Bernoulli probabilities.
        """
        random_values = self.sampling_rng.random(len(probabilities))
        selected_indices = np.where(random_values < probabilities)[0]

        # Avoid empty selected set.
        if len(selected_indices) == 0:
            selected_indices = np.array([int(np.argmax(probabilities))])

        selected_indices = sorted(int(idx) for idx in selected_indices)

        return [self.clients[idx] for idx in selected_indices]


    @staticmethod
    def compute_update_norm(
        global_parameters: list[np.ndarray],
        local_parameters: list[np.ndarray],
    ) -> float:
        """
        Compute ||local_parameters - global_parameters||_2.
        """
        squared_norm = 0.0

        for global_param, local_param in zip(global_parameters, local_parameters):
            diff = local_param - global_param
            squared_norm += float(np.sum(diff * diff))

        return float(np.sqrt(squared_norm))

    def set_observed_update_norms(
            self,
            observed_update_norms: dict[int, float],
            reset_holt_state: bool = False,
    ) -> None:
        """
        Store observed update norms.

        If reset_holt_state is True, also initialize Holt state from these norms.
        """
        for client_id, update_norm in observed_update_norms.items():
            client_id = int(client_id)
            safe_norm = max(float(update_norm), 0.0)

            self.estimated_update_norms[client_id] = safe_norm

            if reset_holt_state:
                self.holt_levels[client_id] = np.log(safe_norm + self.holt_epsilon)
                self.holt_trends[client_id] = 0.0

    def compute_selected_update_norms(
            self,
            global_parameters: list[np.ndarray],
            client_parameters: list[list[np.ndarray]],
            selected_client_ids: list[int],
    ) -> dict[int, float]:
        """
        Compute real update norms for selected clients from their uploaded updates.
        """
        selected_update_norms: dict[int, float] = {}

        for client_id, local_parameters in zip(selected_client_ids, client_parameters):
            selected_update_norms[int(client_id)] = self.compute_update_norm(
                global_parameters=global_parameters,
                local_parameters=local_parameters,
            )

        return selected_update_norms


    def ocs_aggregate_updates(
        self,
        global_parameters: list[np.ndarray],
        client_parameters: list[list[np.ndarray]],
        selected_client_ids: list[int],
        sampling_probabilities: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Aggregate selected client updates using inverse-probability weighting.

        Delta = sum_{k in A_t} (alpha_k / p_k) Delta_k
        w_next = w_current + Delta
        """
        aggregated_update = [
            np.zeros_like(param) for param in global_parameters
        ]

        for local_params, client_id in zip(client_parameters, selected_client_ids):
            alpha_k = self.client_weights[client_id]
            p_k = max(float(sampling_probabilities[client_id]), self.p_min)

            for layer_idx, (global_param, local_param) in enumerate(
                    zip(global_parameters, local_params)
            ):
                client_update = local_param - global_param
                aggregated_update[layer_idx] += (alpha_k / p_k) * client_update

        new_global_parameters = [
            global_param + update
            for global_param, update in zip(global_parameters, aggregated_update)
        ]

        return new_global_parameters

    def update_estimated_update_norms(
            self,
            round_id: int,
            global_parameters: list[np.ndarray],
            client_parameters: list[list[np.ndarray]],
            selected_client_ids: list[int],
    ) -> None:
        """
        Update stored update norms according to update_norm_mode.

        In all modes, selected clients are updated from their observed updates.

        For unselected clients:
          - "holt"   : Holt extrapolation;
          - "newest" : keep latest stored norm;
          - "real"   : query real current update norm.
        """
        selected_update_norms = self.compute_selected_update_norms(
            global_parameters=global_parameters,
            client_parameters=client_parameters,
            selected_client_ids=selected_client_ids,
        )

        if self.update_norm_mode == "newest":
            self.set_observed_update_norms(
                observed_update_norms=selected_update_norms,
                reset_holt_state=False,
            )
            return

        if self.update_norm_mode == "real":
            selected_set = set(int(client_id) for client_id in selected_client_ids)
            unselected_client_ids = [
                client_id for client_id in range(len(self.clients))
                if client_id not in selected_set
            ]

            unselected_update_norms = self._query_update_norms(
                round_id=round_id,
                client_ids=unselected_client_ids,
                global_parameters=global_parameters,
            )

            all_update_norms = dict(selected_update_norms)
            all_update_norms.update(unselected_update_norms)

            self.set_observed_update_norms(
                observed_update_norms=all_update_norms,
                reset_holt_state=False,
            )
            return

        if self.update_norm_mode == "holt":
            old_levels = self.holt_levels.copy()
            old_trends = self.holt_trends.copy()

            new_levels = old_levels.copy()
            new_trends = old_trends.copy()

            for client_id in range(len(self.clients)):
                old_level = old_levels[client_id]
                old_trend = old_trends[client_id]

                if client_id in selected_update_norms:
                    update_norm = selected_update_norms[client_id]
                    observed_log_norm = np.log(update_norm + self.holt_epsilon)

                    new_level = (
                            self.holt_rho * observed_log_norm
                            + (1.0 - self.holt_rho) * (old_level + old_trend)
                    )

                    new_trend = (
                            self.holt_beta * (new_level - old_level)
                            + (1.0 - self.holt_beta) * old_trend
                    )

                else:
                    new_level = old_level + old_trend
                    new_trend = self.holt_gamma * old_trend

                new_levels[client_id] = new_level
                new_trends[client_id] = new_trend

            self.holt_levels = new_levels
            self.holt_trends = new_trends

            self.estimated_update_norms = np.maximum(
                0.0,
                np.exp(self.holt_levels + self.holt_trends) - self.holt_epsilon,
            )
            return

        raise ValueError(f"Unsupported update_norm_mode: {self.update_norm_mode}")

    @staticmethod
    def weighted_average(values: List[float], weights: List[int]) -> float:
        """
        Compute a weighted average.
        """
        if not values or not weights or sum(weights) == 0:
            return 0.0
        return float(sum(v * w for v, w in zip(values, weights)) / sum(weights))

    def _send_train_messages(self, round_id: int, selected_clients: Sequence[Client]) -> List[Message]:
        """
        Send the current global model to selected clients for local training.
        """
        replies: List[Message] = []

        global_parameters = self.global_model.get_model_parameters()

        for client in selected_clients:
            message = Message(
                sender_id=self.server_id,
                receiver_id=f"client_{client.client_id}",
                message_type="train",
                round_id=round_id,
                parameters=global_parameters,
                metrics={},
                config={
                    "lr": self.learning_rate,
                    "local_epochs": self.local_epochs,
                },
            )
            reply = client.handle_message(message)
            replies.append(reply)

        return replies

    def _query_update_norms(
            self,
            round_id: int,
            client_ids: Sequence[int],
            global_parameters: list[np.ndarray],
    ) -> dict[int, float]:
        """
        Ask the given clients to train locally and return only update norms.
        """
        update_norms: dict[int, float] = {}

        for client_id in client_ids:
            client_id = int(client_id)
            client = self.clients[client_id]

            message = Message(
                sender_id=self.server_id,
                receiver_id=f"client_{client.client_id}",
                message_type="query_update_norm",
                round_id=round_id,
                parameters=global_parameters,
                metrics={},
                config={
                    "lr": self.learning_rate,
                    "local_epochs": self.local_epochs,
                },
            )

            reply = client.handle_message(message)
            update_norms[client_id] = float(reply.metrics["update_norm"])

        return update_norms

    def _send_evaluate_messages(self, round_id: int) -> List[Message]:
        """
        Send the current global model to all clients for local evaluation.
        """
        replies: List[Message] = []

        global_parameters = self.global_model.get_model_parameters()

        for client in self.clients:
            message = Message(
                sender_id=self.server_id,
                receiver_id=f"client_{client.client_id}",
                message_type="evaluate",
                round_id=round_id,
                parameters=global_parameters,
                metrics={},
                config={},
            )
            reply = client.handle_message(message)
            replies.append(reply)

        return replies

    def _save_history(self) -> None:
        """
        Save simulation configuration and round-by-round metrics as JSON.
        """
        history_path = self.results_dir / f"{self.run_name}.json"

        final_config = dict(self.config) if self.config is not None else {}

        final_config["num_server_rounds"] = int(self.completed_rounds)
        final_config["communication_budget"] = int(self.used_communication_bytes)

        output = {
            "config": final_config,
            "metrics": self.history,
        }

        history_path.write_text(json.dumps(output, indent=2))

    def _save_final_model(self) -> None:
        """
        Save the final global model as a PyTorch checkpoint.
        """
        model_path = self.results_dir / f"{self.run_name}_final_model.pt"
        torch.save(self.global_model.state_dict(), model_path)

    def initialize_update_norms_with_probe(self) -> None:
        """
        Perform a fake/probe round before round 1 to initialize all update norms.

        No full model updates are uploaded or aggregated.
        """
        global_parameters = self.global_model.get_model_parameters()
        all_client_ids = list(range(len(self.clients)))

        probe_update_norms = self._query_update_norms(
            round_id=0,
            client_ids=all_client_ids,
            global_parameters=global_parameters,
        )

        self.set_observed_update_norms(
            observed_update_norms=probe_update_norms,
            reset_holt_state=True,
        )

    def run(self) -> List[dict]:
        """
        Execute the full federated learning simulation.
        """
        if self.probe_update_norm_initialization:
            self.initialize_update_norms_with_probe()

        for round_id in range(1, self.num_rounds + 1):
            if (
                    self.communication_budget is not None
                    and self.used_communication_bytes >= self.communication_budget
            ):
                break

            # -------------------------
            # Physical-system update
            # -------------------------
            if self.physical_system is not None:
                self.current_physical_state = self.physical_system(return_details=False)

            # -------------------------
            # OCS probability computation
            # -------------------------
            estimated_update_norms_before_sampling = self.estimated_update_norms.copy()
            sampling_probabilities = self.compute_ocs_probabilities()

            # -------------------------
            # OCS client sampling
            # -------------------------
            selected_clients = self.sample_clients_from_probabilities(
                probabilities=sampling_probabilities,
            )
            selected_client_ids = [int(client.client_id) for client in selected_clients]

            # Save the current global model before local training
            global_parameters_before_training = self.global_model.get_model_parameters()

            # -------------------------
            # Training phase
            # -------------------------
            train_replies = self._send_train_messages(
                round_id=round_id,
                selected_clients=selected_clients,
            )

            local_parameters: List[List[np.ndarray]] = []
            local_num_examples: List[int] = []
            local_losses: Dict[str, float] = {}
            local_accs: Dict[str, float] = {}
            upload_bytes: Dict[str, int] = {}
            download_bytes: Dict[str, int] = {}


            for reply in train_replies:
                local_parameters.append(reply.parameters)
                n = int(reply.metrics["num_examples"])
                local_num_examples.append(n)

                cid = reply.sender_id
                local_losses[cid] = float(reply.metrics.get("train_loss", 0.0))
                local_accs[cid] = float(reply.metrics.get("train_acc", 0.0))

                upload_bytes[cid] = int(reply.metrics.get("upload_bytes", reply.message_size))
                download_bytes[cid] = int(reply.metrics.get("download_bytes", 0))

            round_communication = sum(upload_bytes.values()) + sum(download_bytes.values())
            self.used_communication_bytes += round_communication
            self.completed_rounds = round_id

            if local_parameters:
                aggregated_parameters = self.ocs_aggregate_updates(
                    global_parameters=global_parameters_before_training,
                    client_parameters=local_parameters,
                    selected_client_ids=selected_client_ids,
                    sampling_probabilities=sampling_probabilities,
                )

                self.global_model.set_model_parameters(aggregated_parameters)

                self.update_estimated_update_norms(
                    round_id=round_id,
                    global_parameters=global_parameters_before_training,
                    client_parameters=local_parameters,
                    selected_client_ids=selected_client_ids,
                )

            # -------------------------
            # Evaluation phase
            # -------------------------
            eval_replies = self._send_evaluate_messages(round_id=round_id)

            eval_losses: List[float] = []
            eval_accs: List[float] = []
            eval_num_examples: List[int] = []

            for reply in eval_replies:
                eval_losses.append(float(reply.metrics.get("eval_loss", 0.0)))
                eval_accs.append(float(reply.metrics.get("eval_acc", 0.0)))
                eval_num_examples.append(int(reply.metrics["num_examples"]))

            fed_eval_loss = self.weighted_average(eval_losses, eval_num_examples)
            fed_eval_acc = self.weighted_average(eval_accs, eval_num_examples)

            # -------------------------
            # Logging
            # -------------------------
            if self.current_physical_state is not None:
                communication_rates = self.current_physical_state["communication_rates_bps"].tolist()
            else:
                communication_rates = None

            # OCS is not resource-constrained. We assign equal RBs only for
            # communication-time logging and comparison with resource-aware schemes.
            assigned_blocks_per_client = (
                float(self.resource_block_budget) / len(selected_client_ids)
                if selected_client_ids
                else 0.0
            )

            assigned_resource_blocks = {
                f"client_{client_id}": assigned_blocks_per_client
                for client_id in selected_client_ids
            }

            row = {
                "round": round_id,
                "selected_clients": selected_client_ids,
                "assigned_resource_blocks": assigned_resource_blocks,
                "sampling_probabilities": sampling_probabilities.tolist(),
                "estimated_update_norms_before_sampling": estimated_update_norms_before_sampling.tolist(),
                "uplink_rates_per_resource_block_bps": communication_rates,
                "train_loss": local_losses,
                "train_acc": local_accs,
                "fed_eval_loss": fed_eval_loss,
                "fed_eval_acc": fed_eval_acc,
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
            }

            self.history.append(row)
            print(row)

        self._save_history()
        self._save_final_model()

        return self.history