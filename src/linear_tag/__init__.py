"""Public API for the keyed linear-tag package.

This package provides:
- Binary linear algebra utilities over GF(2).
- Measurement transcript representations.
- The configurable keyed linear-tag circuit model.
"""

from .gf2 import (
    int_to_bits,
    bits_to_int,
    gf2_rank,
    tag_value,
)

from .transcript import Transcript
from .model import KeyedLinearTagModel

__all__ = [
    "int_to_bits",
    "bits_to_int",
    "gf2_rank",
    "tag_value",
    "Transcript",
    "KeyedLinearTagModel",
]