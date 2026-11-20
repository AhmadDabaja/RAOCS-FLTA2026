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
      - sends the current global model to the sampled clients,
      - collects local updates,
      - aggregates selected updates using global data weights and inverse-probability correction,
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

        self.probe_update_norm_initialization = (
            True
            if self.config is None
            else bool(self.config.get("probe_update_norm_initialization", True))
        )

        self.refresh_probability = (
            0.0
            if self.config is None
            else float(self.config.get("refresh_probability", 0.0))
        )

        if not (0.0 <= self.refresh_probability <= 1.0):
            raise ValueError("refresh_probability must be in [0, 1].")

        seed = (
            None if self.config is None else self.config.get("seed", None)
        )

        self.sampling_rng = np.random.default_rng(seed)
        self.refresh_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]) if seed is not None else None)

        # Client data weights alpha_k = n_k / n
        self.client_num_examples = np.array(
            [client.num_train_examples() for client in self.clients],
            dtype=float,
        )
        self.client_weights = self.client_num_examples / np.sum(self.client_num_examples)

        model_parameters = self.global_model.get_model_parameters()

        num_model_parameters = sum(param.size for param in model_parameters)
        initial_update_norm = np.sqrt(float(num_model_parameters))

        self.model_update_size_bits = int(
            8 * sum(param.nbytes for param in model_parameters)
        )

        self.estimated_update_norms = np.full(
            len(self.clients),
            initial_update_norm,
            dtype=float,
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

    def compute_energy_probability_caps(
            self,
            minimum_required_blocks: np.ndarray,
    ) -> dict:
        """
        Compute energy caps for non-selected norm refreshing with probability q.

        With L_k = download + computation and V_k = upload:
            p_k (L_k + V_k) + (1 - p_k) q L_k <= E_k
            bar_p_k = min(1, (E_k - q L_k) / ((1-q) L_k + V_k)).

        Non-positive caps exclude clients from selection, exactly as before.
        """
        num_clients = len(self.clients)

        raw_caps = np.zeros(num_clients, dtype=float)
        final_caps = np.zeros(num_clients, dtype=float)

        if self.current_physical_state is None:
            return {
                "raw_energy_probability_caps": raw_caps,
                "final_energy_probability_caps": final_caps,
                "energy_feasible_mask": np.zeros(num_clients, dtype=bool),
                "removed_by_energy_mask": np.zeros(num_clients, dtype=bool),
                "num_clients_removed_by_energy": 0,
                "num_clients_with_cap_below_pmin": 0,
            }

        required_keys = [
            "energy_budget_joules",
            "download_energy_joules",
            "computation_energy_joules",
            "upload_energy_joules",
        ]

        for key in required_keys:
            if key not in self.current_physical_state:
                raise KeyError(
                    f"Physical state is missing '{key}'. "
                    "Make sure PhysicalSystem returns energy quantities."
                )

        energy_budget = np.asarray(
            self.current_physical_state["energy_budget_joules"], dtype=float,
        )
        download_energy = np.asarray(
            self.current_physical_state["download_energy_joules"], dtype=float,
        )
        computation_energy = np.asarray(
            self.current_physical_state["computation_energy_joules"], dtype=float,
        )
        upload_energy = np.asarray(
            self.current_physical_state["upload_energy_joules"], dtype=float,
        )

        resource_feasible_mask = (np.isfinite(minimum_required_blocks) & (minimum_required_blocks > 0.0))

        local_training_energy = download_energy + computation_energy
        denominator = (1.0 - self.refresh_probability) * local_training_energy + upload_energy
        valid_mask = resource_feasible_mask & (denominator > 0.0)

        raw_caps[valid_mask] = (
                    (energy_budget[valid_mask] - self.refresh_probability * local_training_energy[valid_mask])
                    / denominator[valid_mask]
        )
        energy_feasible_mask = valid_mask & (raw_caps > 0.0)

        final_caps[energy_feasible_mask] = np.minimum(
            1.0, raw_caps[energy_feasible_mask],
        )

        removed_by_energy_mask = resource_feasible_mask & (final_caps <= 0.0)

        cap_below_pmin_mask = (
            energy_feasible_mask
            & (final_caps > 0.0)
            & (final_caps < self.p_min)
        )

        return {
            "raw_energy_probability_caps": raw_caps,
            "final_energy_probability_caps": final_caps,
            "energy_feasible_mask": energy_feasible_mask,
            "removed_by_energy_mask": removed_by_energy_mask,
            "num_clients_removed_by_energy": int(np.sum(removed_by_energy_mask)),
            "num_clients_with_cap_below_pmin": int(np.sum(cap_below_pmin_mask)),
        }

    def compute_reaocs_probabilities(
            self,
            minimum_required_blocks: np.ndarray,
            probability_caps: np.ndarray,
    ) -> np.ndarray:
        """
        Compute REAOCS probabilities using capped water-filling.

        Objective constraint:
            sum_k p_k b_{k,min} <= B

        Energy cap:
            0 < p_k <= bar_p_k

        Difference from RAOCS:
            saturation is at bar_p_k instead of 1.
        """
        num_clients = len(self.clients)
        probabilities = np.zeros(num_clients, dtype=float)

        probability_caps = np.asarray(probability_caps, dtype=float)

        # Make sure the cap used in the formulation is:
        #   bar_p_k = min(1, max(0, hat_p_k))
        probability_caps = np.where(
            np.isfinite(probability_caps),
            np.minimum(1.0, np.maximum(0.0, probability_caps)),
            0.0,
        )

        finite_clients = (
                np.isfinite(minimum_required_blocks)
                & (minimum_required_blocks > 0.0)
                & np.isfinite(probability_caps)
                & (probability_caps > 0.0)
        )

        if not np.any(finite_clients):
            return probabilities

        scores = self.client_weights * self.estimated_update_norms
        scores = np.maximum(scores, 1e-12)

        remaining = finite_clients.copy()
        remaining_budget = float(self.resource_block_budget)

        while True:
            remaining_indices = np.where(remaining)[0]

            if len(remaining_indices) == 0 or remaining_budget <= 0.0:
                break

            b_remaining = minimum_required_blocks[remaining_indices]
            scores_remaining = scores[remaining_indices]

            denominator = np.sum(scores_remaining * np.sqrt(b_remaining))

            if denominator <= 0.0:
                break

            scale = remaining_budget / denominator

            candidate_probabilities = (
                    scale * scores_remaining / np.sqrt(b_remaining)
            )

            candidate_caps = probability_caps[remaining_indices]

            saturated_mask = candidate_probabilities >= candidate_caps

            if not np.any(saturated_mask):
                probabilities[remaining_indices] = candidate_probabilities
                break

            saturated_indices = remaining_indices[saturated_mask]
            saturated_caps = probability_caps[saturated_indices]

            probabilities[saturated_indices] = saturated_caps
            remaining[saturated_indices] = False

            # Saturated clients consume cap_k * b_{k,min}, not b_{k,min}.
            remaining_budget -= float(
                np.sum(saturated_caps * minimum_required_blocks[saturated_indices])
            )

        probabilities = np.minimum(probabilities, probability_caps)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        return probabilities


    def sample_clients_from_probabilities(
            self,
            probabilities: np.ndarray,
    ) -> List[Client]:
        """
        Sample clients independently using Bernoulli probabilities.

        If all probabilities are zero, no client is selected.
        If probabilities are positive but Bernoulli sampling selects no client,
        force-select the client with the largest probability.
        """
        if len(probabilities) == 0 or float(np.sum(probabilities)) <= 0.0:
            return []

        random_values = self.sampling_rng.random(len(probabilities))
        selected_indices = np.where(random_values < probabilities)[0]

        if len(selected_indices) == 0:
            selected_indices = np.array([int(np.argmax(probabilities))])

        selected_indices = sorted(int(idx) for idx in selected_indices)

        return [self.clients[idx] for idx in selected_indices]

    def compute_actual_energy_consumed(
            self,
            selected_client_ids: list[int],
            refreshed_client_ids: list[int],
    ) -> np.ndarray:
        """
        Charge energy for operations that actually occurred this round.

        Selected: download + local computation + upload.
        Refreshed but not selected: download + local computation.
        Others: zero. E_k is an expected-energy threshold, not a battery meter.
        """
        num_clients = len(self.clients)
        actual_energy = np.zeros(num_clients, dtype=float)

        if self.current_physical_state is None:
            return actual_energy

        download_energy = np.asarray(
            self.current_physical_state["download_energy_joules"], dtype=float,
        )
        computation_energy = np.asarray(
            self.current_physical_state["computation_energy_joules"], dtype=float,
        )
        upload_energy = np.asarray(
            self.current_physical_state["upload_energy_joules"], dtype=float,
        )

        actual_energy[refreshed_client_ids] = (download_energy[refreshed_client_ids]
                                               + computation_energy[refreshed_client_ids])

        actual_energy[selected_client_ids] = (download_energy[selected_client_ids]
                                              + computation_energy[selected_client_ids]
                                              + upload_energy[selected_client_ids])

        return actual_energy

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
    ) -> None:
        """
        Store observed update norms.
        """
        for client_id, update_norm in observed_update_norms.items():
            client_id = int(client_id)
            safe_norm = max(float(update_norm), 0.0)

            self.estimated_update_norms[client_id] = safe_norm

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


    def reaocs_aggregate_updates(
            self,
            global_parameters: list[np.ndarray],
            client_parameters: list[list[np.ndarray]],
            selected_client_ids: list[int],
            sampling_probabilities: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Aggregate selected client updates using inverse-probability weighting.

        Delta = sum_{k in A_t} (alpha_k / max(p_k, p_min)) Delta_k
        w_next = w_current + Delta

        Here p_min is only an aggregation-stability floor. It does not change
        the true sampling probability and does not affect energy feasibility.
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
    ) -> list[int]:
        """
        Renew selected clients' norms from their uploaded updates. Independently
        refresh each energy-eligible non-selected client with probability q;
        the refreshed norms are stored for the next round's sampling decision.

        Returns IDs of non-selected clients that actually refreshed.
        """
        selected_update_norms = self.compute_selected_update_norms(
            global_parameters=global_parameters,
            client_parameters=client_parameters,
            selected_client_ids=selected_client_ids,
        )

        selected_set = set(int(client_id) for client_id in selected_client_ids)
        unselected_client_ids = [client_id for client_id in range(len(self.clients)) if client_id not in selected_set]

        if self.refresh_probability == 0.0:
            refreshed_client_ids = []
        else:
            if self.current_physical_state is None:
                raise ValueError("REAOCS requires energy data to check refreshing eligibility.")
            energy_budget = np.asarray(self.current_physical_state["energy_budget_joules"], dtype=float,)
            local_training_energy = (np.asarray(self.current_physical_state["download_energy_joules"], dtype=float)
                                     + np.asarray(self.current_physical_state["computation_energy_joules"], dtype=float)
                                     )

            # Refresh eligibility depends on expected local-training energy,
            # not uplink resource-block feasibility.
            refresh_candidates = [client_id for client_id in unselected_client_ids
                                  if self.refresh_probability * local_training_energy[client_id]
                                  <= energy_budget[client_id]
                                  ]


            if self.refresh_probability == 1.0:
                refreshed_client_ids = refresh_candidates
            else:
                draws = self.refresh_rng.random(len(refresh_candidates))
                refreshed_client_ids = [client_id for client_id, draw in zip(refresh_candidates, draws)
                                        if draw < self.refresh_probability
                                        ]

        # Always renew selected clients' cached norms, including at q=0 and q=1.
        refreshed_update_norms = self._query_update_norms(round_id = round_id,
                                                          client_ids = refreshed_client_ids,
                                                          global_parameters = global_parameters,
                                                          ) if refreshed_client_ids else {}

        all_update_norms = dict(selected_update_norms)
        all_update_norms.update(refreshed_update_norms)
        self.set_observed_update_norms(observed_update_norms=all_update_norms)

        return refreshed_client_ids

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
            # REAOCS probability computation
            # -------------------------
            estimated_update_norms_before_sampling = self.estimated_update_norms.copy()

            minimum_required_blocks = self.compute_minimum_required_resource_blocks()

            energy_cap_info = self.compute_energy_probability_caps(
                minimum_required_blocks=minimum_required_blocks,
            )

            final_energy_probability_caps = energy_cap_info["final_energy_probability_caps"]

            sampling_probabilities = self.compute_reaocs_probabilities(
                minimum_required_blocks=minimum_required_blocks,
                probability_caps=final_energy_probability_caps,
            )

            # -------------------------
            # REAOCS client sampling
            # -------------------------
            selected_clients = self.sample_clients_from_probabilities(
                probabilities=sampling_probabilities,
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
                aggregated_parameters = self.reaocs_aggregate_updates(
                    global_parameters=global_parameters_before_training,
                    client_parameters=local_parameters,
                    selected_client_ids=selected_client_ids,
                    sampling_probabilities=sampling_probabilities,
                )

                self.global_model.set_model_parameters(aggregated_parameters)

            refreshed_client_ids = self.update_estimated_update_norms(
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

            assigned_resource_blocks = {
                f"client_{client_id}": float(minimum_required_blocks[client_id])
                for client_id in selected_client_ids
                if np.isfinite(minimum_required_blocks[client_id])
            }

            actual_energy_consumed_joules = self.compute_actual_energy_consumed(
                selected_client_ids=selected_client_ids,
                refreshed_client_ids=refreshed_client_ids,
            )

            row = {
                "round": round_id,
                "selected_clients": selected_client_ids,
                "refreshed_clients": refreshed_client_ids,
                "assigned_resource_blocks": assigned_resource_blocks,
                "sampling_probabilities": sampling_probabilities.tolist(),
                "energy_probability_caps": final_energy_probability_caps.tolist(),
                "actual_energy_consumed_joules": actual_energy_consumed_joules.tolist(),
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