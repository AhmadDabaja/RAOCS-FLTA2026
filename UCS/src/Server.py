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
      - samples clients uniformly without replacement,
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

        self.sampling_fraction = (
            1.0 if self.config is None else float(self.config.get("sampling_fraction", 1.0))
        )

        self.p_min = (
            0.1 if self.config is None else float(self.config.get("p_min", 0.1))
        )

        # Global data weights alpha_k = n_k / n.
        self.client_num_examples = np.asarray(
            [client.get_num_train_examples() for client in self.clients],
            dtype=float,
        )
        self.total_num_examples = float(np.sum(self.client_num_examples))

        if self.total_num_examples <= 0:
            raise ValueError("Total number of client training examples must be positive.")

        self.client_weights = self.client_num_examples / self.total_num_examples

        seed = (
            None if self.config is None else self.config.get("seed", None)
        )

        self.sampling_rng = np.random.default_rng(seed)

        self.history: List[dict] = []


    def get_num_selected_clients(self) -> int:
        """
        Return the actual number of clients sampled by UCS.

        This uses the implemented rule floor(sampling_fraction * K).
        """
        num_clients = len(self.clients)

        if num_clients == 0:
            return 0

        num_selected = max(1, int(np.floor(self.sampling_fraction * num_clients)))
        return min(num_selected, num_clients)

    def get_sampling_probability(self) -> float:
        """
        Return the actual inclusion probability used by fixed-size UCS.

        Since UCS samples m clients uniformly without replacement,
        each client has probability p = m / K.
        """
        num_clients = len(self.clients)

        if num_clients == 0:
            return 0.0

        return float(self.get_num_selected_clients()) / float(num_clients)

    def uniform_sample_clients(self) -> List[Client]:
        """
        Uniformly sample a fraction of clients without replacement.
        """
        num_clients = len(self.clients)

        if num_clients == 0:
            return []

        num_selected = self.get_num_selected_clients()

        selected_indices = self.sampling_rng.choice(
            num_clients,
            size=num_selected,
            replace=False,
        )

        selected_indices = sorted(int(idx) for idx in selected_indices)

        return [self.clients[idx] for idx in selected_indices]

    def ucs_aggregate_updates(
            self,
            client_parameters: List[List[np.ndarray]],
            selected_client_ids: List[int],
            global_parameters_before_training: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Aggregate selected client updates using common importance-corrected aggregation.

        W^{t+1} =
            W^t + sum_{k in A_t} (n_k / n) * (1 / p_k^t) * (W_k^t - W^t)

        For UCS, p_k^t = m / K for every client, floored by p_min.
        """
        if not client_parameters or not selected_client_ids:
            return self.global_model.get_model_parameters()

        raw_sampling_probability = self.get_sampling_probability()
        capped_sampling_probability = max(raw_sampling_probability, self.p_min)

        aggregated_updates = [
            np.zeros_like(param) for param in global_parameters_before_training
        ]

        for local_params, client_id in zip(client_parameters, selected_client_ids):
            alpha_k = float(self.client_weights[client_id])
            correction_weight = alpha_k / capped_sampling_probability

            for layer_idx, local_param in enumerate(local_params):
                client_update = (
                        local_param - global_parameters_before_training[layer_idx]
                )
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
            # Uniform client sampling
            # -------------------------
            selected_clients = self.uniform_sample_clients()
            selected_client_ids = [int(client.client_id) for client in selected_clients]

            # -------------------------
            # Training phase
            # -------------------------
            global_parameters_before_training = self.global_model.get_model_parameters()

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
                aggregated_parameters = self.ucs_aggregate_updates(
                    client_parameters=local_parameters,
                    selected_client_ids=selected_client_ids,
                    global_parameters_before_training=global_parameters_before_training,
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

            assigned_blocks_per_client = (
                float(self.resource_block_budget) / len(selected_client_ids)
                if selected_client_ids
                else 0.0
            )

            assigned_resource_blocks = {
                f"client_{client_id}": assigned_blocks_per_client
                for client_id in selected_client_ids
            }

            sampling_probability = self.get_sampling_probability()
            sampling_probabilities = [
                sampling_probability for _ in range(len(self.clients))
            ]

            row = {
                "round": round_id,
                "selected_clients": selected_client_ids,
                "assigned_resource_blocks": assigned_resource_blocks,
                "sampling_probabilities": sampling_probabilities,
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