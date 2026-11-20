from __future__ import annotations

from typing import List, Tuple
from itertools import islice
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FCNFashionModel(nn.Module):
    """
    Fully connected neural network for FashionMNIST classification.

    Design choices:
    - Input images are flattened from shape [B, 1, 28, 28] to [B, 784].
    - Local training uses exactly one randomly selected batch per call.
    - Evaluation iterates over the entire test loader.
    - Model parameter exchange is done through explicit NumPy copies.
    """

    def __init__(self) -> None:
        """Initialize the network layers."""
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images with shape [B, 1, 28, 28].

        Returns
        -------
        torch.Tensor
            Output logits with shape [B, 10].
        """
        # Flatten each image into a 1D feature vector.
        x = x.view(x.size(0), -1)

        # Pass the features through the hidden layers.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Produce class logits.
        x = self.fc4(x)
        return x

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Train the model on exactly one randomly selected batch.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing the local training data.
        criterion : nn.Module
            Loss function used for optimization.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        device : torch.device
            Target device for computation.

        Returns
        -------
        Tuple[float, float]
            Batch loss and batch accuracy.
        """
        self.to(device)
        self.train()

        # Safeguard against empty dataloaders.
        if len(train_loader) == 0:
            return 0.0, 0.0

        # Randomly choose one batch index from the local dataloader.
        random_batch_idx = random.randrange(len(train_loader))

        # Extract only the selected batch.
        x, y = next(islice(iter(train_loader), random_batch_idx, None))

        # Move batch tensors to the target device.
        x = x.to(device)
        y = y.to(device)

        # Standard optimization step.
        optimizer.zero_grad()
        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Compute batch accuracy.
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

        batch_loss = loss.item()
        batch_acc = correct / total if total > 0 else 0.0

        return batch_loss, batch_acc

    @torch.no_grad()
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on the full test dataset.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader containing the local test data.
        criterion : nn.Module
            Loss function used for evaluation.
        device : torch.device
            Target device for computation.

        Returns
        -------
        Tuple[float, float]
            Average loss and overall accuracy across the full test loader.
        """
        self.to(device)
        self.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Evaluate every batch in the test loader.
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = self(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        return avg_loss, acc

    def get_model_parameters(self) -> List[np.ndarray]:
        """
        Export model parameters as independent NumPy copies.

        Returns
        -------
        List[np.ndarray]
            A list of parameter arrays copied from the current model state.
        """
        return [
            tensor.detach().cpu().numpy().copy()
            for _, tensor in self.state_dict().items()
        ]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        """
        Load model parameters from NumPy arrays using explicit copies.

        Parameters
        ----------
        params : List[np.ndarray]
            Parameter values to load into the model.

        Raises
        ------
        ValueError
            If the number of provided arrays does not match the number of
            tensors in the model state dictionary.
        """
        state = self.state_dict()

        if len(params) != len(state):
            raise ValueError(
                f"Parameter length mismatch: got {len(params)} arrays, expected {len(state)}."
            )

        new_state = {}
        for (key, old_tensor), arr in zip(state.items(), params):
            # Create a safe NumPy copy before converting back to a tensor.
            arr_copy = np.array(arr, copy=True)

            # Rebuild the tensor while preserving dtype and device.
            new_state[key] = torch.tensor(
                arr_copy,
                dtype=old_tensor.dtype,
                device=old_tensor.device,
            )

        self.load_state_dict(new_state, strict=True)



class FCNCIFAR10Model(nn.Module):
    """
    Fully connected neural network for CIFAR10 classification.

    Design choices:
    - Input images are flattened from shape [B, 3, 32, 32] to [B, 3072].
    - Local training uses exactly one randomly selected batch per call.
    - Evaluation iterates over the entire test loader.
    - Model parameter exchange is done through explicit NumPy copies.
    """

    def __init__(self) -> None:
        """Initialize the network layers."""
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images with shape [B, 3, 32, 32].

        Returns
        -------
        torch.Tensor
            Output logits with shape [B, 10].
        """
        # Flatten each image into a 1D feature vector.
        x = x.view(x.size(0), -1)

        # Pass the features through the hidden layers.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Produce class logits.
        x = self.fc4(x)
        return x

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Train the model on exactly one randomly selected batch.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader containing the local training data.
        criterion : nn.Module
            Loss function used for optimization.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        device : torch.device
            Target device for computation.

        Returns
        -------
        Tuple[float, float]
            Batch loss and batch accuracy.
        """
        self.to(device)
        self.train()

        # Safeguard against empty dataloaders.
        if len(train_loader) == 0:
            return 0.0, 0.0

        # Randomly choose one batch index from the local dataloader.
        random_batch_idx = random.randrange(len(train_loader))

        # Extract only the selected batch.
        x, y = next(islice(iter(train_loader), random_batch_idx, None))

        # Move batch tensors to the target device.
        x = x.to(device)
        y = y.to(device)

        # Standard optimization step.
        optimizer.zero_grad()
        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Compute batch accuracy.
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

        batch_loss = loss.item()
        batch_acc = correct / total if total > 0 else 0.0

        return batch_loss, batch_acc

    @torch.no_grad()
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on the full test dataset.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader containing the local test data.
        criterion : nn.Module
            Loss function used for evaluation.
        device : torch.device
            Target device for computation.

        Returns
        -------
        Tuple[float, float]
            Average loss and overall accuracy across the full test loader.
        """
        self.to(device)
        self.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Evaluate every batch in the test loader.
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = self(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        return avg_loss, acc

    def get_model_parameters(self) -> List[np.ndarray]:
        """
        Export model parameters as independent NumPy copies.

        Returns
        -------
        List[np.ndarray]
            A list of parameter arrays copied from the current model state.
        """
        return [
            tensor.detach().cpu().numpy().copy()
            for _, tensor in self.state_dict().items()
        ]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        """
        Load model parameters from NumPy arrays using explicit copies.

        Parameters
        ----------
        params : List[np.ndarray]
            Parameter values to load into the model.

        Raises
        ------
        ValueError
            If the number of provided arrays does not match the number of
            tensors in the model state dictionary.
        """
        state = self.state_dict()

        if len(params) != len(state):
            raise ValueError(
                f"Parameter length mismatch: got {len(params)} arrays, expected {len(state)}."
            )

        new_state = {}
        for (key, old_tensor), arr in zip(state.items(), params):
            # Create a safe NumPy copy before converting back to a tensor.
            arr_copy = np.array(arr, copy=True)

            # Rebuild the tensor while preserving dtype and device.
            new_state[key] = torch.tensor(
                arr_copy,
                dtype=old_tensor.dtype,
                device=old_tensor.device,
            )

        self.load_state_dict(new_state, strict=True)

class BasicBlockCIFARResNet(nn.Module):
    """
    Basic residual block for CIFAR-style ResNet.

    Shortcut follows the same Option-A logic as your old PLT model:
    - spatial downsampling with avg_pool2d when stride=2
    - zero-padding channels when in_channels < out_channels
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_bn,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_bn,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        self.downsample = (stride != 1) or (in_channels != out_channels)

    def _downsample_identity(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.stride != 1:
            identity = F.avg_pool2d(identity, kernel_size=1, stride=self.stride)

        in_c = identity.shape[1]
        out_c = self.out_channels

        if in_c < out_c:
            pad_c = out_c - in_c
            identity = F.pad(identity, (0, 0, 0, 0, 0, pad_c))

        return identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self._downsample_identity(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet8CIFAR10Model(nn.Module):
    """
    ResNet-8 for CIFAR-10.

    Architecture:
        stem conv: 3 -> 16
        stage1: 1 residual block, 16 channels
        stage2: 1 residual block, 32 channels, stride 2
        stage3: 1 residual block, 64 channels, stride 2
        global average pooling
        linear classifier: 64 -> 10

    FedBN behavior:
        - Conv/Linear parameters are exchanged and aggregated.
        - BatchNorm affine parameters and running statistics remain local.
    """

    def __init__(self, use_bn: bool = True) -> None:
        super().__init__()

        self.use_bn = use_bn

        self.stem_conv = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_bn,
        )
        self.stem_bn = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = BasicBlockCIFARResNet(
            in_channels=16,
            out_channels=16,
            stride=1,
            use_bn=use_bn,
        )

        self.layer2 = BasicBlockCIFARResNet(
            in_channels=16,
            out_channels=32,
            stride=2,
            use_bn=use_bn,
        )

        self.layer3 = BasicBlockCIFARResNet(
            in_channels=32,
            out_channels=64,
            stride=2,
            use_bn=use_bn,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Train the model on exactly one randomly selected local batch.
        This matches your current FCN training behavior.
        """
        self.to(device)
        self.train()

        if len(train_loader) == 0:
            return 0.0, 0.0

        random_batch_idx = random.randrange(len(train_loader))
        x, y = next(islice(iter(train_loader), random_batch_idx, None))

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

        batch_loss = loss.item()
        batch_acc = correct / total if total > 0 else 0.0

        return batch_loss, batch_acc

    @torch.no_grad()
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        self.to(device)
        self.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = self(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        return avg_loss, acc

    def _is_batchnorm_parameter(self, name: str) -> bool:
        """
        Detect BatchNorm affine parameters.

        Running mean/variance are buffers, not named_parameters(),
        so they are already excluded from parameter exchange.
        """
        return (
            ".bn" in name
            or "stem_bn" in name
            or name.endswith("bn.weight")
            or name.endswith("bn.bias")
        )

    def _iter_non_bn_named_parameters(self):
        """
        Yield only parameters that should be globally exchanged.

        FedBN:
            - Aggregate Conv/Linear parameters.
            - Keep BatchNorm affine parameters local.
        """
        for name, param in self.named_parameters():
            if self._is_batchnorm_parameter(name):
                continue
            yield name, param

    def get_model_parameters(self) -> List[np.ndarray]:
        """
        Export non-BatchNorm model parameters as NumPy arrays.

        This implements FedBN-style parameter exchange.
        """
        return [
            param.detach().cpu().numpy().copy()
            for _, param in self._iter_non_bn_named_parameters()
        ]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        """
        Load non-BatchNorm model parameters from NumPy arrays.

        BatchNorm parameters and running statistics remain local.
        """
        non_bn_params = list(self._iter_non_bn_named_parameters())

        if len(params) != len(non_bn_params):
            raise ValueError(
                f"Parameter length mismatch: got {len(params)} arrays, "
                f"expected {len(non_bn_params)} non-BN parameters."
            )

        for (_, param), arr in zip(non_bn_params, params):
            arr_copy = np.array(arr, copy=True)
            param.data = torch.tensor(
                arr_copy,
                dtype=param.dtype,
                device=param.device,
            )


class ResNet20CIFAR100Model(nn.Module):
    """
    ResNet-20 for CIFAR-100.

    Architecture:
        stem conv: 3 -> 16
        stage1: 3 residual blocks, 16 channels
        stage2: 3 residual blocks, 32 channels, first block stride 2
        stage3: 3 residual blocks, 64 channels, first block stride 2
        global average pooling
        linear classifier: 64 -> 100

    ResNet depth:
        6n + 2 with n = 3 gives ResNet-20.

    FedBN behavior:
        - Conv/Linear parameters are exchanged and aggregated.
        - BatchNorm affine parameters and running statistics remain local.
    """

    def __init__(self, use_bn: bool = True) -> None:
        super().__init__()

        self.use_bn = use_bn

        self.stem_conv = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_bn,
        )
        self.stem_bn = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            BasicBlockCIFARResNet(
                in_channels=16,
                out_channels=16,
                stride=1,
                use_bn=use_bn,
            ),
            BasicBlockCIFARResNet(
                in_channels=16,
                out_channels=16,
                stride=1,
                use_bn=use_bn,
            ),
            BasicBlockCIFARResNet(
                in_channels=16,
                out_channels=16,
                stride=1,
                use_bn=use_bn,
            ),
        )

        self.layer2 = nn.Sequential(
            BasicBlockCIFARResNet(
                in_channels=16,
                out_channels=32,
                stride=2,
                use_bn=use_bn,
            ),
            BasicBlockCIFARResNet(
                in_channels=32,
                out_channels=32,
                stride=1,
                use_bn=use_bn,
            ),
            BasicBlockCIFARResNet(
                in_channels=32,
                out_channels=32,
                stride=1,
                use_bn=use_bn,
            ),
        )

        self.layer3 = nn.Sequential(
            BasicBlockCIFARResNet(
                in_channels=32,
                out_channels=64,
                stride=2,
                use_bn=use_bn,
            ),
            BasicBlockCIFARResNet(
                in_channels=64,
                out_channels=64,
                stride=1,
                use_bn=use_bn,
            ),
            BasicBlockCIFARResNet(
                in_channels=64,
                out_channels=64,
                stride=1,
                use_bn=use_bn,
            ),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Train the model on exactly one randomly selected local batch.
        This matches your current FCN training behavior.
        """
        self.to(device)
        self.train()

        if len(train_loader) == 0:
            return 0.0, 0.0

        random_batch_idx = random.randrange(len(train_loader))
        x, y = next(islice(iter(train_loader), random_batch_idx, None))

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

        batch_loss = loss.item()
        batch_acc = correct / total if total > 0 else 0.0

        return batch_loss, batch_acc

    @torch.no_grad()
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        self.to(device)
        self.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = self(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        return avg_loss, acc

    def _is_batchnorm_parameter(self, name: str) -> bool:
        """
        Detect BatchNorm affine parameters.

        Running mean/variance are buffers, not named_parameters(),
        so they are already excluded from parameter exchange.
        """
        return (
            ".bn" in name
            or "stem_bn" in name
            or name.endswith("bn.weight")
            or name.endswith("bn.bias")
        )

    def _iter_non_bn_named_parameters(self):
        """
        Yield only parameters that should be globally exchanged.

        FedBN:
            - Aggregate Conv/Linear parameters.
            - Keep BatchNorm affine parameters local.
        """
        for name, param in self.named_parameters():
            if self._is_batchnorm_parameter(name):
                continue
            yield name, param

    def get_model_parameters(self) -> List[np.ndarray]:
        """
        Export non-BatchNorm model parameters as NumPy arrays.

        This implements FedBN-style parameter exchange.
        """
        return [
            param.detach().cpu().numpy().copy()
            for _, param in self._iter_non_bn_named_parameters()
        ]

    def set_model_parameters(self, params: List[np.ndarray]) -> None:
        """
        Load non-BatchNorm model parameters from NumPy arrays.

        BatchNorm parameters and running statistics remain local.
        """
        non_bn_params = list(self._iter_non_bn_named_parameters())

        if len(params) != len(non_bn_params):
            raise ValueError(
                f"Parameter length mismatch: got {len(params)} arrays, "
                f"expected {len(non_bn_params)} non-BN parameters."
            )

        for (_, param), arr in zip(non_bn_params, params):
            arr_copy = np.array(arr, copy=True)
            param.data = torch.tensor(
                arr_copy,
                dtype=param.dtype,
                device=param.device,
            )