"""
Abstract base class for all Mechanical MNIST Cahn-Hilliard models.

Every model must:
  - Implement forward(x) returning a dict whose keys match get_target_keys().
  - Implement from_config(config) as a classmethod constructor.
  - Optionally override get_target_keys() and get_input_transform() to declare
    which data the model needs and any preprocessing it requires.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MechMNISTModel(ABC, nn.Module):
    """
    Base class for multi-task FE surrogate models on Mechanical MNIST CH.

    Forward contract
    ----------------
    forward(x) must return a dict containing exactly the keys declared by
    get_target_keys(), e.g.:
        {"psi": (B, 7), "force": (B, 28), "disp": (B, 2, H, W)}

    This contract lets train.py and dataset.py remain model-agnostic.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)

        Returns
        -------
        dict with keys matching get_target_keys()
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config) -> "MechMNISTModel":
        """Construct model from a Config instance."""
        ...

    def get_target_keys(self) -> list:
        """
        Which output targets this model predicts.

        Used by create_dataloaders() to decide which targets to load from disk
        (avoiding loading e.g. 6 GB of displacement data for a scalars-only
        model).  Override in subclasses that don't predict all three targets.
        """
        return ["psi", "force", "disp"]

    def get_input_transform(self, train: bool):
        """
        Optional torchvision-style callable applied to each image tensor after
        resizing, inside Dataset.__getitem__.

        Parameters
        ----------
        train : bool
            True during training (augmentations enabled), False at val/test.

        Returns
        -------
        callable or None
            Callable signature: (img_tensor: Tensor) -> Tensor.
            Return None (default) for no transform.
        """
        return None
