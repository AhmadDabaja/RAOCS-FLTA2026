from __future__ import annotations

from typing import Optional

import torch

from .Data import load_data
from .Message import Message
from .Models import FCNFashionModel, FCNCIFAR10Model, ResNet8CIFAR10Model, ResNet20CIFAR100Model


class Client:
    """
    Simulated federated learning client.

    Each client owns one local partition of the dataset. The client can:
      - receive a training message from the server,
      - train locally,
      - receive an evaluation message from the server,
      - evaluate locally,
      - reply with metrics and, for training, updated parameters.
    """

    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        model_name: str,
        num_partitions: int,
        alpha: float,
        batch_size: int,
        test_ratio: float,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_partitions = num_partitions
        self.alpha = alpha
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._trainloader = None
        self._testloader = None

        # Persistent local model.
        # Important for FedBN: BatchNorm affine parameters and running statistics
        # must remain local to each client across rounds.
        self._local_model = None

    def _ensure_data_loaded(self) -> None:
        """
        Load the local train/test loaders once and keep them cached.
        """
        if self._trainloader is None or self._testloader is None:
            self._trainloader, self._testloader = load_data(
                dataset_name=self.dataset_name,
                partition_id=self.client_id,
                num_partitions=self.num_partitions,
                alpha=self.alpha,
                test_ratio=self.test_ratio,
                batch_size=self.batch_size,
                seed=self.seed,
            )

    def num_train_examples(self) -> int:
        """
        Return the number of local training examples owned by this client.
        """
        self._ensure_data_loaded()
        return len(self._trainloader.dataset)

    def _create_model(self):
        """
        Create the correct local model according to the selected dataset/model pair.
        """
        dataset_name = self.dataset_name.lower()
        model_name = self.model_name.lower()

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

    def _get_local_model(self):
        """
        Return this client's persistent local model.

        This is essential for FedBN:
          - the server sends global non-BN parameters;
          - the client loads those non-BN parameters;
          - BatchNorm parameters/statistics remain stored locally across rounds.
        """
        if self._local_model is None:
            self._local_model = self._create_model()
            self._local_model.to(self.device)

        return self._local_model

    def handle_message(self, message: Message) -> Message:
        """
        Dispatch an incoming message to the appropriate client routine.
        """
        if message.message_type == "train":
            return self.train(message)

        if message.message_type == "query_update_norm":
            return self.query_update_norm(message)

        if message.message_type == "evaluate":
            return self.evaluate(message)

        raise ValueError(f"Unsupported message type: {message.message_type}")

    @staticmethod
    def compute_update_norm(
            global_parameters: list,
            local_parameters: list,
    ) -> float:
        """
        Compute ||local_parameters - global_parameters||_2.
        """
        squared_norm = 0.0

        for global_param, local_param in zip(global_parameters, local_parameters):
            diff = local_param - global_param
            squared_norm += float((diff * diff).sum())

        return float(squared_norm ** 0.5)

    def _train_local_model_from_message(self, message: Message):
        """
        Train a local model initialized from the parameters carried by message.
        """
        if not message.has_parameters():
            raise ValueError(f"{message.message_type} message does not contain model parameters.")

        self._ensure_data_loaded()

        model = self._get_local_model()
        model.set_model_parameters(message.parameters)

        criterion = torch.nn.CrossEntropyLoss()

        lr = float(message.config["lr"])
        local_epochs = int(message.config.get("local_epochs", 1))

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        losses = []
        accs = []

        for _ in range(local_epochs):
            loss, acc = model.train_epoch(
                train_loader=self._trainloader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
            )
            losses.append(loss)
            accs.append(acc)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc = sum(accs) / len(accs) if accs else 0.0

        return model, avg_loss, avg_acc

    def train(self, message: Message) -> Message:
        """
        Perform local training and return updated model parameters.
        """
        model, avg_loss, avg_acc = self._train_local_model_from_message(message)

        reply = Message(
            sender_id=f"client_{self.client_id}",
            receiver_id=message.sender_id,
            message_type="reply_train",
            round_id=message.round_id,
            parameters=model.get_model_parameters(),
            metrics={
                "train_loss": float(avg_loss),
                "train_acc": float(avg_acc),
                "num_examples": int(len(self._trainloader.dataset)),
            },
            config={},
        )

        reply.metrics["upload_bytes"] = reply.message_size
        reply.metrics["download_bytes"] = message.message_size

        return reply

    def query_update_norm(self, message: Message) -> Message:
        """
        Perform local training and return only the update norm.
        """
        model, _, _ = self._train_local_model_from_message(message)

        local_parameters = model.get_model_parameters()
        update_norm = self.compute_update_norm(
            global_parameters=message.parameters,
            local_parameters=local_parameters,
        )

        return Message(
            sender_id=f"client_{self.client_id}",
            receiver_id=message.sender_id,
            message_type="reply_update_norm",
            round_id=message.round_id,
            parameters=None,
            metrics={
                "update_norm": float(update_norm),
            },
            config={},
        )

    def evaluate(self, message: Message) -> Message:
        """
        Evaluate the received global model on the client's local test data.
        """
        if not message.has_parameters():
            raise ValueError("Evaluation message does not contain model parameters.")

        self._ensure_data_loaded()

        model = self._get_local_model()
        model.set_model_parameters(message.parameters)

        criterion = torch.nn.CrossEntropyLoss()

        eval_loss, eval_acc = model.test_epoch(
            test_loader=self._testloader,
            criterion=criterion,
            device=self.device,
        )

        reply = Message(
            sender_id=f"client_{self.client_id}",
            receiver_id=message.sender_id,
            message_type="reply_evaluate",
            round_id=message.round_id,
            parameters=None,
            metrics={
                "eval_loss": float(eval_loss),
                "eval_acc": float(eval_acc),
                "num_examples": int(len(self._testloader.dataset)),
            },
            config={},
        )
        return reply