
"""Measurement transcript representation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Transcript:
    """Measurement result of a single keyed linear-tag block.

    Attributes
    ----------
    i : int
        Measured index-register outcome.
    r : int
        Measured tag-register outcome.
    y : int
        Measured system-register outcome.

    Notes
    -----
    The transcript represents Z = (I, R, Y).

    Its fields are nonnegative integer representations
    of binary measurement outcomes.

    Register-width validation is performed by the
    circuit model, which knows the dimensions m, s, n.

    Instances are immutable and hashable, making them
    suitable as dictionary keys and histogram outcomes.
    """

    i: int
    r: int
    y: int

    def __post_init__(self) -> None:
        """Validate and normalize measurement outcomes."""
        for name in ("i", "r", "y"):
            value = getattr(self, name)

            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(
                    f"{name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{name} must be nonnegative."
                )

            # Normalize NumPy integer scalars to Python integers.
            object.__setattr__(self, name, int(value))
