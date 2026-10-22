
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

    def as_bitstrings(
            self,
            *,
            m: int,
            s: int,
            n: int,
    ) -> dict[str, str]:
        """Return measurement outcomes as fixed-width binary strings.

        Parameters
        ----------
        m : int
            Number of index-register qubits.
        s : int
            Number of tag-register qubits.
        n : int
            Number of system-register qubits.

        Returns
        -------
        dict[str, str]
            Binary representations of the three outcomes, using
            the most significant bit on the left.

        Notes
        -----
        This method does not modify the stored integer outcomes.

        Register widths must be explicitly provided because a
        Transcript does not store the circuit configuration.
        """
        for width in (m, s, n):
            if not isinstance(width, int) or width < 1:
                raise ValueError("Register widths must be positive integers.")

        if not (
                0 <= self.i < 2 ** m
                and 0 <= self.r < 2 ** s
                and 0 <= self.y < 2 ** n
        ):
            raise ValueError("Transcript outcomes exceed register widths.")

        return {
            "i": f"{self.i:0{m}b}",
            "r": f"{self.r:0{s}b}",
            "y": f"{self.y:0{n}b}",
        }
