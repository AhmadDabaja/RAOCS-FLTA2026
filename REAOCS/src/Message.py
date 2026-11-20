from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy
import numpy as np


@dataclass(slots=True)
class Message:
    """
    Message exchanged between the server and clients during simulation.

    Attributes
    ----------
    sender_id : str
        Identifier of the component sending the message.
    receiver_id : str
        Identifier of the component receiving the message.
    message_type : str
        Type of communication, such as "train", "evaluate",
        "reply_train", or "reply_evaluate".
    round_id : int
        Federated learning round index.
    parameters : Optional[List[np.ndarray]]
        Model parameters carried by the message.
    metrics : Dict[str, Any]
        Metrics carried by the message, such as loss, accuracy,
        number of examples, upload bytes, or download bytes.
    config : Dict[str, Any]
        Configuration values carried by the message, such as learning rate,
        local epochs, batch size, or evaluation settings.
    message_size : int
        Estimated size in bytes of the model parameters carried by the
        message. This counts only NumPy array storage using `nbytes`; it does
        not include Python object overhead, metadata, or serialization overhead.
    """

    sender_id: str
    receiver_id: str
    message_type: str
    round_id: int
    parameters: Optional[List[np.ndarray]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    message_size: int = 0

    def __post_init__(self) -> None:
        """
        Estimate the size of the parameter payload after initialization.

        This counts only NumPy array storage using `nbytes`.
        It does not include Python object overhead, metadata, or serialization
        overhead.
        """
        self.message_size = self.compute_message_size()

    def compute_message_size(self) -> int:
        """
        Return the estimated size, in bytes, of the parameters in the message.
        """
        if self.parameters is None:
            return 0

        return int(sum(param.nbytes for param in self.parameters))

    def copy(self) -> "Message":
        """
        Return a safe copy of the message.

        Parameter arrays, metrics, and configuration values are copied to avoid
        accidental in-place modification during simulation.
        """
        copied_parameters = None
        if self.parameters is not None:
            copied_parameters = [np.array(param, copy=True) for param in self.parameters]

        return Message(
            sender_id=self.sender_id,
            receiver_id=self.receiver_id,
            message_type=self.message_type,
            round_id=self.round_id,
            parameters=copied_parameters,
            metrics=copy.deepcopy(self.metrics),
            config=copy.deepcopy(self.config),
            message_size=self.message_size,
        )

    def has_parameters(self) -> bool:
        """Return True if the message contains model parameters."""
        return self.parameters is not None

    def has_metrics(self) -> bool:
        """Return True if the message contains metrics."""
        return bool(self.metrics)

    def has_config(self) -> bool:
        """Return True if the message contains configuration values."""
        return bool(self.config)