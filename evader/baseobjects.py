# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Missile evader model object interface and base classes."""
from abc import abstractmethod
from typing import Protocol, Any, Optional
import logging

import numpy as np

import evader.math

module_logger = logging.getLogger(__name__)


class IModelObject(Protocol):
    """Interface of model objects."""

    @abstractmethod
    def next(
            self,
            t: float,
            dt: float
    ) -> None:
        """Performs next modeling step.

        Args:
            t: Current model time.
            dt: Model time step.
        """
        raise NotImplementedError

    @abstractmethod
    def draw(
            self,
            ax: Any
    ) -> None:
        """Draws the object.

        Args:
            ax: Axes object to draw in.
        """
        raise NotImplementedError

    @abstractmethod
    def erase(self) -> None:
        """Erases the object."""
        raise NotImplementedError


class IPhysicalBody(Protocol):
    """Interface of physical bodies."""

    @property
    @abstractmethod
    def x(self) -> Optional[np.ndarray]:
        """Returns body position."""
        raise NotImplementedError

    @property
    @abstractmethod
    def v(self) -> Optional[np.ndarray]:
        """Returns body velocity."""
        raise NotImplementedError


class IMissile(IModelObject, IPhysicalBody):
    """Interface of missiles."""

    @property
    @abstractmethod
    def destroyed(self) -> bool:
        """Returns True, if the object is destroyed."""
        raise NotImplementedError


class ModelObjectBase(IModelObject, IPhysicalBody):
    """Base class for model objects.

    Implements various functionality common to model objects.

    Attributes:
        model: Parent model.

    Args:
        model: Parent model.
        name: Object name.
        logger: Logger object.
        x: Object position vector.
        v: Object velocity vector.
    """
    x: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None

    def __init__(
            self,
            model,
            name,
            logger,
            x,
            v
    ):
        """Initialize."""
        self.model = model
        self.name = name
        if logger is None:
            logger = module_logger
        self.logger = logger.getChild(name)
        self.x = evader.math.float_array(x)
        self.v = evader.math.float_array(v)

    def _str_dict(self):
        """Makes a dictionary for ``__str__()`` representation."""
        return {'x': self.x, 'v': f'{self.v} ({np.linalg.norm(self.v)})'}

    def __str__(self):
        """Makes an informal string representation of this object.

        Calls :meth:`_str_dict` to obtain a dictionary of values to be printed
        """
        params = ', '.join(f'{k}={v}' for k, v in self._str_dict().items())
        return f'{self.name}({params})'
