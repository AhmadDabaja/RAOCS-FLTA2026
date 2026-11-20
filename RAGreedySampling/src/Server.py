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
    Central server coordinating Resource-Aware Greedy Sampling.

    RAGreedySampling:
      - computes b_k,min from the physical system,
      - sorts feasible clients by increasing b_k,min,
      - selects clients greedily until the resource-block budget B is filled,
      - estimates each client's participation probability from historical participation,
      - aggregates selected updates using global data weights and inverse-probability correction,
      - evaluates the global model,
      - stores round history.
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

        self.round_deadline_seconds = (
            10.0
            if self.config is None
            else float(self.config.get("round_deadline_seconds", 10.0))
        )

        self.resource_block_budget = (
            len(self.clients)
            if self.config is None
            else int(self.config.get("resource_block_budget", len(self.clients)))
        )

        self.p_min = (
            0.1
            if self.config is None
            else float(self.config.get("p_min", 0.1))
        )

        self.epsilon = 1e-12

        # Number of times each client has participated up to the current round.
        # Used to estimate p_k^t for deterministic greedy selection.
        self.client_participation_counts = np.zeros(len(self.clients), dtype=float)

        # Client data weights alpha_k = n_k / n
        self.client_num_examples = np.array(
            [client.num_train_examples() for client in self.clients],
            dtype=float,
        )
        self.client_weights = self.client_num_examples / np.sum(self.client_num_examples)

        model_parameters = self.global_model.get_model_parameters()

        self.model_update_size_bits = int(
            8 * sum(param.nbytes for param in model_parameters)
        )

        self.history: List[dict] = []

    def compute_minimum_required_resource_blocks(self) -> np.ndarray:
        """
        Compute upload-only minimum required resource blocks.

        Upload constraint:
            Z / (nu_k^t b_k^t) <= tau

        Therefore:
            b_{k,min}^t = Z / (nu_k^t tau)

        A client is infeasible only if its uplink rate is non-positive or if
        b_{k,min}^t exceeds the total resource-block budget.
        """
        num_clients = len(self.clients)
        minimum_required_blocks = np.full(num_clients, np.inf, dtype=float)

        if self.current_physical_state is None:
            return minimum_required_blocks

        uplink_rates = np.asarray(
            self.current_physical_state["communication_rates_bps"],
            dtype=float,
        )

        feasible = uplink_rates > 0.0

        required_blocks = (
                self.model_update_size_bits
                / (uplink_rates[feasible] * self.round_deadline_seconds)
        )

        required_blocks = np.maximum(required_blocks, 1e-12)
        minimum_required_blocks[feasible] = required_blocks

        minimum_required_blocks[
            minimum_required_blocks > float(self.resource_block_budget)
            ] = np.inf

        return minimum_required_blocks

    def select_clients_greedily(
            self,
            minimum_required_blocks: np.ndarray,
    ) -> List[Client]:
        """
        Select clients greedily according to the smallest b_k,min.

        The rule is:
            1. keep only feasible clients with finite b_k,min,
            2. sort them by increasing b_k,min,
            3. add clients while the total assigned resource blocks stays <= B.

        This is resource-aware but deterministic and biased toward clients that
        require fewer resource blocks.
        """
        feasible_indices = np.where(
            np.isfinite(minimum_required_blocks)
            & (minimum_required_blocks > 0.0)
        )[0]

        if len(feasible_indices) == 0:
            return []

        sorted_indices = sorted(
            [int(idx) for idx in feasible_indices],
            key=lambda idx: (float(minimum_required_blocks[idx]), idx),
        )

        selected_clients: List[Client] = []
        used_resource_blocks = 0.0
        total_budget = float(self.resource_block_budget)

        for client_id in sorted_indices:
            required_blocks = float(minimum_required_blocks[client_id])

            if used_resource_blocks + required_blocks <= total_budget + self.epsilon:
                selected_clients.append(self.clients[client_id])
                used_resource_blocks += required_blocks
            else:
                break

        return selected_clients

    def compute_empirical_participation_probabilities(
            self,
            round_id: int,
    ) -> np.ndarray:
        """
        Estimate p_k^t from historical greedy participation.

        p_k^t = number of participations of client k up to round t / t

        The current round must already be included in
        self.client_participation_counts before this function is called.
        """
        denominator = max(1, int(round_id))
        return self.client_participation_counts / float(denominator)

    def cap_participation_probabilities(
            self,
            probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Floor all probabilities by p_min before using them in aggregation.
        """
        return np.maximum(probabilities, self.p_min)

    def ragreedy_aggregate_updates(
            self,
            client_parameters: list[list[np.ndarray]],
            selected_client_ids: list[int],
            global_parameters_before_training: list[np.ndarray],
            capped_participation_probabilities: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Aggregate selected client updates using common importance-corrected aggregation.

        W^{t+1} =
            W^t + sum_{k in A_t} (n_k / n) * (1 / p_k^t) * (W_k^t - W^t)

        where p_k^t is the empirical greedy participation probability, floored by p_min.
        """
        if not client_parameters or not selected_client_ids:
            return self.global_model.get_model_parameters()

        aggregated_updates = [
            np.zeros_like(param) for param in global_parameters_before_training
        ]

        for local_params, client_id in zip(client_parameters, selected_client_ids):
            alpha_k = float(self.client_weights[client_id])
            p_k = float(capped_participation_probabilities[client_id])

            correction_weight = alpha_k / p_k

            for layer_idx, local_param in enumerate(local_params):
                client_update = local_param - global_parameters_before_training[layer_idx]
                aggregated_updates[layer_idx] += correction_weight * client_update

        aggregated_parameters = [
            global_param + update
            for global_param, update in zip(
                global_parameters_before_training,
                aggregated_updates,
            )
        ]

        return aggregated_parameters

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

    def run(self) -> List[dict]:
        """
        Execute the full federated learning simulation.
        """
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
            # RAGreedySampling client selection
            # -------------------------
            minimum_required_blocks = self.compute_minimum_required_resource_blocks()

            selected_clients = self.select_clients_greedily(
                minimum_required_blocks=minimum_required_blocks,
            )

            selected_client_ids = [
                int(client.client_id) for client in selected_clients
            ]

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
            local_client_ids: List[int] = []
            local_num_examples: List[int] = []
            local_losses: Dict[str, float] = {}
            local_accs: Dict[str, float] = {}
            upload_bytes: Dict[str, int] = {}
            download_bytes: Dict[str, int] = {}

            for reply in train_replies:
                local_parameters.append(reply.parameters)

                client_id = int(str(reply.sender_id).split("_")[-1])
                local_client_ids.append(client_id)

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

            # Update empirical participation counts using the current round.
            for client_id in local_client_ids:
                self.client_participation_counts[client_id] += 1.0

            empirical_participation_probabilities = (
                self.compute_empirical_participation_probabilities(round_id=round_id)
            )

            capped_participation_probabilities = (
                self.cap_participation_probabilities(empirical_participation_probabilities)
            )

            if local_parameters:
                aggregated_parameters = self.ragreedy_aggregate_updates(
                    client_parameters=local_parameters,
                    selected_client_ids=local_client_ids,
                    global_parameters_before_training=global_parameters_before_training,
                    capped_participation_probabilities=capped_participation_probabilities,
                )

                self.global_model.set_model_parameters(aggregated_parameters)


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

            assigned_resource_blocks = {
                f"client_{client_id}": float(minimum_required_blocks[client_id])
                for client_id in selected_client_ids
                if np.isfinite(minimum_required_blocks[client_id])
            }

            row = {
                "round": round_id,
                "selected_clients": selected_client_ids,
                "assigned_resource_blocks": assigned_resource_blocks,
                "sampling_probabilities": empirical_participation_probabilities.tolist(),
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