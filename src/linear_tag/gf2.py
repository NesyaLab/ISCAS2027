
"""Binary linear algebra utilities for the keyed linear-tag construction.

This module implements the operations over GF(2) required to construct
and analyze binary linear tags.

Bit vectors use little-endian indexing: element q corresponds to bit
2**q of the associated nonnegative integer.
"""

import numpy as np


def _require_int(value: int, name: str) -> int:
    """Validate and normalize an integer argument."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")

    return int(value)


def _as_binary_array(
    values,
    *,
    ndim: int,
    name: str,
) -> np.ndarray:
    """Return a validated, independent uint8 binary array.

    Parameters
    ----------
    values : array-like
        Input vector or matrix.
    ndim : int
        Required number of dimensions.
    name : str
        Argument name used in error messages.

    Returns
    -------
    np.ndarray
        A copy containing only binary values, with dtype uint8.

    Raises
    ------
    TypeError
        If the input is not integer- or boolean-valued.
    ValueError
        If the shape or binary values are invalid.
    """
    array = np.asarray(values)

    if array.ndim != ndim:
        raise ValueError(
            f"{name} must be a {ndim}-dimensional array."
        )

    # Reject values that would otherwise be silently truncated
    # or converted when casting to an unsigned integer.
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(
            f"{name} must contain integer or boolean values."
        )

    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only 0 or 1.")

    return array.astype(np.uint8, copy=True)


def int_to_bits(value: int, width: int) -> np.ndarray:
    """Represent a nonnegative integer as a binary vector.

    The returned vector is little-endian: position q contains
    the coefficient of 2**q.

    Parameters
    ----------
    value : int
        Integer to convert.
    width : int
        Number of bits in the output vector.

    Returns
    -------
    np.ndarray
        Binary vector of shape (width,) and dtype uint8.

    Raises
    ------
    TypeError
        If value or width is not an integer.
    ValueError
        If width is not positive or value does not fit.

    Examples
    --------
    >>> int_to_bits(5, 4).tolist()
    [1, 0, 1, 0]
    """
    value = _require_int(value, "value")
    width = _require_int(width, "width")

    if width < 1:
        raise ValueError("width must be positive.")

    if not 0 <= value < (1 << width):
        raise ValueError(
            "value must be representable using width bits."
        )

    return np.fromiter(
        ((value >> q) & 1 for q in range(width)),
        dtype=np.uint8,
        count=width,
    )


def bits_to_int(bits) -> int:
    """Convert a little-endian binary vector to an integer.

    Parameters
    ----------
    bits : array-like
        One-dimensional binary vector.

    Returns
    -------
    int
        Integer represented by the input bits.

    Examples
    --------
    >>> bits_to_int([1, 0, 1, 0])
    5
    """
    vector = _as_binary_array(
        bits, ndim=1, name="bits"
    )

    if vector.size == 0:
        raise ValueError("bits must not be empty.")

    return sum(
        int(bit) << q
        for q, bit in enumerate(vector)
    )


def gf2_rank(matrix: np.ndarray) -> int:
    """Compute the rank of a binary matrix over GF(2).

    The algorithm uses Gaussian elimination with XOR row
    operations. All arithmetic is performed over GF(2).

    The input matrix is not modified.

    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix of shape (rows, columns).

    Returns
    -------
    int
        Exact rank over GF(2).

    Notes
    -----
    Floating-point matrix rank is not generally equivalent
    to rank over GF(2). In particular, subtraction and addition
    of rows are both XOR operations in this field.
    """
    A = _as_binary_array(
        matrix, ndim=2, name="matrix"
    )

    rows, cols = A.shape
    rank = 0

    for col in range(cols):

        # Find a nonzero pivot in the current column.
        pivot = next(
            (r for r in range(rank, rows) if A[r, col]),
            None,
        )

        if pivot is None:
            continue

        # Move the pivot row into the current rank position.
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]

        # Eliminate entries below the pivot using GF(2) addition.
        for r in range(rank + 1, rows):
            if A[r, col]:
                A[r] ^= A[rank]

        rank += 1

        if rank == rows:
            break

    return rank


def tag_value(B: np.ndarray, t: int, m: int) -> int:
    """Compute the binary linear tag r = B t over GF(2).

    Parameters
    ----------
    B : np.ndarray
        Binary key matrix of shape (s, m).
    t : int
        Integer representation of the m-bit index label.
    m : int
        Number of index bits, equal to the number of
        columns of B.

    Returns
    -------
    int
        Integer representation of the s-bit tag.

    Notes
    -----
    In the paper-exact construction, m = n and B has
    shape (s, n).

    Matrix multiplication and reduction modulo 2 implement

        r_ell = XOR_q (B[ell, q] AND t[q]).
    """
    m = _require_int(m, "m")

    if m < 1:
        raise ValueError("m must be positive.")

    matrix = _as_binary_array(
        B, ndim=2, name="B"
    )

    if matrix.shape[0] < 1 or matrix.shape[1] != m:
        raise ValueError(
            "B must have shape (s, m), with s >= 1."
        )

    t_bits = int_to_bits(t, m)

    # Perform the binary matrix-vector multiplication.
    r_bits = (matrix @ t_bits) % 2

    return bits_to_int(r_bits)
