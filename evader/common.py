# SPDX-FileCopyrightText: 2021 Kirill Pushkaryov <kpushkaryov@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Various common utilities."""
from abc import abstractmethod
from typing import Any, Iterable, Protocol


class Removable(Protocol):
    """Interface for removable entities."""

    @abstractmethod
    def remove() -> Any:
        """Remove the entity."""
        raise NotImplementedError


def remove_artists(artists: Iterable[Removable]) -> None:
    """Removes artists in an iterable."""
    for a in artists:
        if a:
            a.remove()
