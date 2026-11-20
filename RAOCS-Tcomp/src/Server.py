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

        self.computation_cutoff_seconds = (
            0.5 * self.round_deadline_seconds
            if self.config is None
            else float(
                self.config.get(
                    "computation_cutoff_seconds",
                    0.5 * self.round_deadline_seconds,
                )
            )
        )

        if not (0.0 < self.computation_cutoff_seconds < self.round_deadline_seconds):
            raise ValueError(
                "computation_cutoff_seconds must satisfy "
                "0 < computation_cutoff_seconds < round_deadline_seconds."
            )

        # Computation-aware timing diagnostics for the current round.
        self.last_computation_times_seconds = np.zeros(len(self.clients), dtype=float)
        self.last_server_decision_time_seconds = 0.0
        self.last_upload_window_seconds = self.round_deadline_seconds
        self.last_available_client_mask = np.ones(len(self.clients), dtype=bool)

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

        model_parameters = self.global_model.get_model_parameters()

        num_model_parameters = sum(param.size for param in model_parameters)
        initial_update_norm = np.sqrt(float(num_model_parameters))

        self.model_update_size_bits = int(
            8 * sum(param.nbytes for param in model_parameters)
        )

        # Holt estimator state
        initial_log_norm = np.log(initial_update_norm + self.holt_epsilon)

        self.holt_levels = np.full(
            len(self.clients),
            initial_log_norm,
            dtype=float,
        )

        self.holt_trends = np.zeros(
            len(self.clients),
            dtype=float,
        )

        self.estimated_update_norms = np.full(
            len(self.clients),
            initial_update_norm,
            dtype=float,
        )

        self.history: List[dict] = []

    def compute_minimum_required_resource_blocks(self) -> np.ndarray:
        """
        Compute computation-aware minimum required resource blocks.

        The round is divided into two stages:

            1. Computation and norm-reporting stage.
            2. Upload stage after the server decision.

        The server waits until:

            T_dec^t = min(c, max_k T_k^{comp,t})

        where c is the computation cutoff.

        The available client set is:

            R^t = {k : T_k^{comp,t} <= T_dec^t}

        The remaining common upload window is:

            tau_up^t = tau - T_dec^t

        Therefore, for k in R^t:

            b_{k,min}^t = Z / (nu_k^t tau_up^t)

        Clients outside R^t are marked infeasible by assigning infinity.
        """
        num_clients = len(self.clients)
        minimum_required_blocks = np.full(num_clients, np.inf, dtype=float)

        # Default diagnostics in case the physical state is missing.
        self.last_computation_times_seconds = np.full(num_clients, np.nan, dtype=float)
        self.last_server_decision_time_seconds = 0.0
        self.last_upload_window_seconds = self.round_deadline_seconds
        self.last_available_client_mask = np.zeros(num_clients, dtype=bool)

        if self.current_physical_state is None:
            return minimum_required_blocks

        uplink_rates = np.asarray(
            self.current_physical_state["communication_rates_bps"],
            dtype=float,
        )

        computation_times = np.asarray(
            self.current_physical_state.get(
                "computation_times_seconds",
                np.zeros(num_clients, dtype=float),
            ),
            dtype=float,
        )

        self.last_computation_times_seconds = computation_times.copy()

        valid_computation_times = (
                np.isfinite(computation_times)
                & (computation_times >= 0.0)
        )

        if not np.any(valid_computation_times):
            return minimum_required_blocks

        max_computation_time = float(
            np.max(computation_times[valid_computation_times])
        )

        server_decision_time = min(
            self.computation_cutoff_seconds,
            max_computation_time,
        )

        upload_window_seconds = self.round_deadline_seconds - server_decision_time

        self.last_server_decision_time_seconds = float(server_decision_time)
        self.last_upload_window_seconds = float(upload_window_seconds)

        if upload_window_seconds <= 0.0:
            return minimum_required_blocks

        available_client_mask = (
                valid_computation_times
                & (computation_times <= server_decision_time)
        )

        self.last_available_client_mask = available_client_mask.copy()

        feasible = (
                available_client_mask
                & np.isfinite(uplink_rates)
                & (uplink_rates > 0.0)
        )

        if not np.any(feasible):
            return minimum_required_blocks

        required_blocks = (
                self.model_update_size_bits
                / (uplink_rates[feasible] * upload_window_seconds)
        )

        required_blocks = np.maximum(required_blocks, 1e-12)
        minimum_required_blocks[feasible] = required_blocks

        minimum_required_blocks[
            minimum_required_blocks > float(self.resource_block_budget)
            ] = np.inf

        return minimum_required_blocks

    def get_energy_accounting_mode(self) -> str:
        """
        Derive the energy-accounting mode from update_norm_mode.

        update_norm_mode == "real":
            original/exact-norm accounting:
            resource-feasible candidate clients pay download + computation,
            and selected clients additionally pay upload.

        update_norm_mode in {"holt", "newest"}:
            cached-update accounting:
            only selected clients pay download + computation + upload.
        """
        if self.update_norm_mode == "real":
            return "exact_norm"

        return "cached_update"


    def compute_raocs4wfl_probabilities(
            self,
            minimum_required_blocks: np.ndarray,
    ) -> np.ndarray:
        """
        Compute RAOCS4WFL probabilities using resource-aware capped water-filling.

        Objective constraint:
            sum_k p_k b_{k,min} <= B

        Solution for non-saturated clients:
            p_k proportional to alpha_k * G_hat_k / sqrt(b_{k,min})
        """
        num_clients = len(self.clients)
        probabilities = np.zeros(num_clients, dtype=float)

        finite_clients = (
                np.isfinite(minimum_required_blocks)
                & (minimum_required_blocks > 0.0)
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

            saturated_mask = candidate_probabilities >= 1.0

            if not np.any(saturated_mask):
                probabilities[remaining_indices] = candidate_probabilities
                break

            saturated_indices = remaining_indices[saturated_mask]
            probabilities[saturated_indices] = 1.0
            remaining[saturated_indices] = False

            remaining_budget -= float(
                np.sum(minimum_required_blocks[saturated_indices])
            )

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
            minimum_required_blocks: np.ndarray,
    ) -> np.ndarray:
        """
        Compute actual per-client energy consumed in the current round.

        RAOCS does not use energy to constrain sampling. This function only measures
        the energy consumed under the same accounting model used for REAOCS.

        exact_norm/original:
            zeta_down + zeta_comp + I_k^t zeta_up
            for resource-feasible candidate clients.

        cached_update:
            I_k^t (zeta_down + zeta_comp + zeta_up)
        """
        num_clients = len(self.clients)
        actual_energy = np.zeros(num_clients, dtype=float)

        if self.current_physical_state is None:
            return actual_energy

        required_keys = [
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

        download_energy = np.asarray(
            self.current_physical_state["download_energy_joules"],
            dtype=float,
        )
        computation_energy = np.asarray(
            self.current_physical_state["computation_energy_joules"],
            dtype=float,
        )
        upload_energy = np.asarray(
            self.current_physical_state["upload_energy_joules"],
            dtype=float,
        )

        selected_mask = np.zeros(num_clients, dtype=bool)
        selected_mask[selected_client_ids] = True

        resource_feasible_mask = (
                np.isfinite(minimum_required_blocks)
                & (minimum_required_blocks > 0.0)
        )

        energy_accounting_mode = self.get_energy_accounting_mode()

        if energy_accounting_mode == "exact_norm":
            # In exact computation-aware mode, all available clients have already
            # downloaded the model and computed their local update before sampling.
            available_mask = self.last_available_client_mask.astype(bool)

            actual_energy[available_mask] = (
                    download_energy[available_mask]
                    + computation_energy[available_mask]
            )

            # Only selected clients upload the full model update.
            actual_energy[selected_mask] += upload_energy[selected_mask]

        elif energy_accounting_mode == "cached_update":
            actual_energy[selected_mask] = (
                    download_energy[selected_mask]
                    + computation_energy[selected_mask]
                    + upload_energy[selected_mask]
            )

        else:
            raise ValueError(
                f"Unsupported energy accounting mode: {energy_accounting_mode}"
            )

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
            reset_holt_state: bool = False,
    ) -> None:
        """
        Store observed update norms.

        If reset_holt_state is True, also initialize the Holt state from
        the observed norms. This is used after the probe round.
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


    def raocs4wfl_aggregate_updates(
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
            available_client_mask: np.ndarray | None = None,
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

            if available_client_mask is None:
                candidate_client_ids = list(range(len(self.clients)))
            else:
                candidate_client_ids = [
                    client_id for client_id in range(len(self.clients))
                    if bool(available_client_mask[client_id])
                ]

            unselected_client_ids = [
                client_id for client_id in candidate_client_ids
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

    def _precompute_available_client_updates(
            self,
            round_id: int,
            available_client_ids: Sequence[int],
            global_parameters: list[np.ndarray],
    ) -> tuple[dict[int, Message], dict[int, float]]:
        """
        Exact computation-aware pre-sampling step.

        Clients in R^t already finished computation before T_dec.
        They compute their local update once and report the current update norm.

        The server stores the full local update internally. After OCS sampling,
        selected clients upload/use these already-computed updates. This avoids
        double training.
        """
        available_clients = [
            self.clients[int(client_id)]
            for client_id in available_client_ids
        ]

        replies = self._send_train_messages(
            round_id=round_id,
            selected_clients=available_clients,
        )

        precomputed_replies: dict[int, Message] = {}
        precomputed_update_norms: dict[int, float] = {}

        for reply in replies:
            client_id = int(reply.sender_id.split("_")[-1])

            if reply.parameters is None:
                raise ValueError(
                    f"Precomputed update for client {client_id} has no parameters."
                )

            precomputed_replies[client_id] = reply

            precomputed_update_norms[client_id] = self.compute_update_norm(
                global_parameters=global_parameters,
                local_parameters=reply.parameters,
            )

        return precomputed_replies, precomputed_update_norms

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
        if self.probe_update_norm_initialization and self.update_norm_mode != "real":
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

            # Save the current global model before any local computation.
            global_parameters_before_training = self.global_model.get_model_parameters()

            # -------------------------
            # Computation-aware feasibility
            # -------------------------
            minimum_required_blocks = self.compute_minimum_required_resource_blocks()

            # -------------------------
            # Exact pre-sampling computation and norm reporting
            # -------------------------
            precomputed_replies_by_client_id: dict[int, Message] = {}
            precomputed_update_norms: dict[int, float] = {}

            if self.update_norm_mode == "real":
                available_client_ids = np.where(self.last_available_client_mask)[0].astype(int).tolist()

                precomputed_replies_by_client_id, precomputed_update_norms = (
                    self._precompute_available_client_updates(
                        round_id=round_id,
                        available_client_ids=available_client_ids,
                        global_parameters=global_parameters_before_training,
                    )
                )

                # These are the current exact norms used by OCS in this round.
                self.set_observed_update_norms(
                    observed_update_norms=precomputed_update_norms,
                    reset_holt_state=False,
                )

            # -------------------------
            # RAOCS4WFL probability computation
            # -------------------------
            estimated_update_norms_before_sampling = self.estimated_update_norms.copy()

            sampling_probabilities = self.compute_raocs4wfl_probabilities(
                minimum_required_blocks=minimum_required_blocks,
            )

            # -------------------------
            # RAOCS4WFL client sampling
            # -------------------------
            selected_clients = self.sample_clients_from_probabilities(
                probabilities=sampling_probabilities,
            )

            selected_client_ids = [
                int(client.client_id) for client in selected_clients
            ]

            # -------------------------
            # Training/upload phase
            # -------------------------
            if self.update_norm_mode == "real":
                # Selected clients upload/use their already-computed updates.
                train_replies = [
                    precomputed_replies_by_client_id[int(client_id)]
                    for client_id in selected_client_ids
                ]
            else:
                # Cached-norm modes train only the selected clients after sampling.
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
            norm_report_bytes: Dict[str, int] = {}

            if self.update_norm_mode == "real":
                norm_size = 8 if self.config is None else int(self.config.get("norm_report_bytes", 8))

                # In exact mode, every available client downloads the global model and reports a norm.
                for client_id, reply in precomputed_replies_by_client_id.items():
                    cid = f"client_{client_id}"
                    download_bytes[cid] = int(reply.metrics.get("download_bytes", 0))
                    norm_report_bytes[cid] = norm_size

            for reply in train_replies:
                if reply.parameters is None:
                    raise ValueError(f"Training reply from {reply.sender_id} has no parameters.")

                local_parameters.append(reply.parameters)
                n = int(reply.metrics["num_examples"])
                local_num_examples.append(n)

                cid = reply.sender_id
                local_losses[cid] = float(reply.metrics.get("train_loss", 0.0))
                local_accs[cid] = float(reply.metrics.get("train_acc", 0.0))

                # Only selected clients upload the full model update.
                upload_bytes[cid] = int(reply.metrics.get("upload_bytes", reply.message_size))

                # In cached modes, only selected clients downloaded the model.
                # In exact mode, downloads for all available clients were already recorded above.
                if self.update_norm_mode != "real":
                    download_bytes[cid] = int(reply.metrics.get("download_bytes", 0))

            round_communication = (
                    sum(upload_bytes.values())
                    + sum(download_bytes.values())
                    + sum(norm_report_bytes.values())
            )
            self.used_communication_bytes += round_communication
            self.completed_rounds = round_id

            if local_parameters:
                aggregated_parameters = self.raocs4wfl_aggregate_updates(
                    global_parameters=global_parameters_before_training,
                    client_parameters=local_parameters,
                    selected_client_ids=selected_client_ids,
                    sampling_probabilities=sampling_probabilities,
                )

                self.global_model.set_model_parameters(aggregated_parameters)

            if self.update_norm_mode != "real":
                self.update_estimated_update_norms(
                    round_id=round_id,
                    global_parameters=global_parameters_before_training,
                    client_parameters=local_parameters,
                    selected_client_ids=selected_client_ids,
                    available_client_mask=self.last_available_client_mask,
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
                computation_times = self.last_computation_times_seconds.tolist()
                available_clients = np.where(self.last_available_client_mask)[0].astype(int).tolist()
            else:
                communication_rates = None
                computation_times = None
                available_clients = []

            assigned_resource_blocks = {
                f"client_{client_id}": float(minimum_required_blocks[client_id])
                for client_id in selected_client_ids
                if np.isfinite(minimum_required_blocks[client_id])
            }

            actual_energy_consumed_joules = self.compute_actual_energy_consumed(
                selected_client_ids=selected_client_ids,
                minimum_required_blocks=minimum_required_blocks,
            )

            row = {
                "round": round_id,
                "selected_clients": selected_client_ids,
                "assigned_resource_blocks": assigned_resource_blocks,
                "sampling_probabilities": sampling_probabilities.tolist(),
                "actual_energy_consumed_joules": actual_energy_consumed_joules.tolist(),
                "estimated_update_norms_before_sampling": estimated_update_norms_before_sampling.tolist(),
                "uplink_rates_per_resource_block_bps": communication_rates,
                # Computation-aware timing diagnostics.
                "computation_times_seconds": computation_times,
                "server_decision_time_seconds": float(self.last_server_decision_time_seconds),
                "upload_window_seconds": float(self.last_upload_window_seconds),
                "available_clients": available_clients,



                # Exact-mode diagnostics.
                "precomputed_clients": (
                    sorted(int(client_id) for client_id in precomputed_replies_by_client_id.keys())
                    if self.update_norm_mode == "real"
                    else []
                ),
                "num_precomputed_clients": (
                    len(precomputed_replies_by_client_id)
                    if self.update_norm_mode == "real"
                    else 0
                ),
                "norm_report_bytes": norm_report_bytes,


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