
"""Keyed linear-tag construction and experimental tools.

Reference
---------
"Circuit-Level Inversion Security for All-Outcomes Linear
Combinations of Unitaries", Section IV.

The paper-exact construction has:
    - n index qubits;
    - s tag qubits;
    - n system qubits;
    - nu independently keyed blocks;
    - permutations pi_t(x) = x XOR t;
    - secret matrices B_j in GF(2)^(s x n).

The implementation also supports exploratory configurations with
m <= n and distinct user-defined XOR masks. The paper's decoding,
security, and Table I formulas apply only to the exact construction.

A block is a separate circuit execution. Multiple blocks do not
form one coherent quantum circuit.

The classical sampler implements Proposition 4. It produces the
complete ideal measurement transcript without quantum simulation.

Limitations
-----------
- Only XOR permutations are supported.
- Overlapping output fibers are not supported.
- Statevector and exhaustive enumeration are for small instances.
- Quantum execution is ideal unless a different backend is added.
- Random generation is pseudorandom and is not a cryptographic KGen.
"""

from collections import Counter
from dataclasses import InitVar, dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit.circuit.library import XGate
from qiskit.quantum_info import Statevector

from .gf2 import (
    bits_to_int,
    gf2_rank,
    int_to_bits,
    tag_value,
)
from .transcript import Transcript


def _integer(value: int, name: str) -> int:
    """Validate an integer parameter, rejecting boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")

    return int(value)


def _binary_matrix(values, shape: tuple[int, int]) -> np.ndarray:
    """Create a validated, read-only binary matrix.

    The input is copied so that later changes to a user-provided
    matrix cannot silently modify an initialized model.
    """
    B = np.asarray(values)

    if B.shape != shape:
        raise ValueError(f"Expected matrix shape {shape}.")

    if B.dtype.kind not in ("b", "i", "u"):
        raise TypeError("Matrix entries must be integers or booleans.")

    if np.any((B != 0) & (B != 1)):
        raise ValueError("Matrix entries must belong to GF(2).")

    B = B.astype(np.uint8, copy=True)
    B.flags.writeable = False
    return B


def _tag_depth(B: np.ndarray) -> int:
    """Compute the optimal all-to-all tag-network CNOT depth.

    The CNOT dependency graph is bipartite. Its maximum vertex
    degree equals the minimum number of disjoint-edge layers.
    """
    return int(max(
        np.sum(B, axis=0).max(),
        np.sum(B, axis=1).max(),
    ))


def _solve_gf2(A: np.ndarray, c: np.ndarray) -> Optional[np.ndarray]:
    """Solve Ax = c over GF(2), requiring a unique solution.

    Returns None if A lacks full column rank or the system
    is inconsistent.

    Gaussian elimination uses XOR row operations and never
    modifies the caller's arrays.
    """
    rows, n = A.shape

    augmented = np.column_stack((
        A.astype(np.uint8, copy=True),
        np.asarray(c, dtype=np.uint8),
    ))

    pivot_row = 0

    for col in range(n):
        pivot = next(
            (r for r in range(pivot_row, rows)
             if augmented[r, col]),
            None,
        )

        if pivot is None:
            return None

        if pivot != pivot_row:
            augmented[[pivot_row, pivot]] = augmented[
                [pivot, pivot_row]
            ]

        # Reduced-row-echelon elimination over GF(2).
        for r in range(rows):
            if r != pivot_row and augmented[r, col]:
                augmented[r] ^= augmented[pivot_row]

        pivot_row += 1

    # Full column rank gives an identity block in the first
    # n rows. Any remaining nonzero RHS indicates inconsistency.
    if np.any(augmented[n:, -1]):
        return None

    return augmented[:n, -1].copy()


@dataclass(frozen=True, repr=False)
class KeyedLinearTagModel:
    """Complete configuration of a keyed linear-tag experiment.

    Parameters
    ----------
    n : int
        System-register width. Also the index width in the paper.
    s : int
        Tag-register width, 1 <= s <= n.
    nu : int, default=1
        Number of independently keyed blocks.
    m : int, optional
        Index-register width. Defaults to n.
    masks : sequence of int, optional
        Custom XOR mask for each label t. Must contain 2^m
        distinct n-bit masks. Defaults to mask[t] = t.
    key_matrices : sequence of arrays, optional
        Explicit matrices B_j of shape (s, m). If omitted,
        independent uniform binary matrices are generated.
    seed : int or None, default=42
        NumPy pseudorandom seed for reproducible experiments.
        None uses nondeterministic initialization.

    Notes
    -----
    The instance keeps fixed parameters and fixed secret keys.
    Repeated evaluations reuse those keys.

    To create a fresh-key challenge, instantiate a new model
    with newly generated keys.

    The stored matrices are read-only; get_key() returns copies.
    """

    n: int
    s: int
    nu: int = 1
    m: Optional[int] = None
    masks: Optional[Sequence[int]] = None
    key_matrices: InitVar[Optional[Sequence[np.ndarray]]] = None
    seed: Optional[int] = 42

    _keys: tuple[np.ndarray, ...] = field(
        init=False, repr=False, compare=False
    )
    _rng: np.random.Generator = field(
        init=False, repr=False, compare=False
    )
    _standard_xor: bool = field(init=False, repr=False)
    random_keys_generated: bool = field(init=False, compare=False)

    def __post_init__(
        self,
        key_matrices: Optional[Sequence[np.ndarray]],
    ) -> None:
        """Validate and freeze the model configuration."""
        n = _integer(self.n, "n")
        s = _integer(self.s, "s")
        nu = _integer(self.nu, "nu")
        m = n if self.m is None else _integer(self.m, "m")

        if not 1 <= m <= n:
            raise ValueError("Require 1 <= m <= n.")
        if not 1 <= s <= n:
            raise ValueError("Require 1 <= s <= n.")
        if nu < 1:
            raise ValueError("nu must be positive.")

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "m", m)

        if self.seed is not None:
            seed = _integer(self.seed, "seed")
            if seed < 0:
                raise ValueError("seed must be nonnegative.")
            object.__setattr__(self, "seed", seed)

        # Validate and normalize the XOR permutation family.
        if self.masks is None:
            standard_xor = True
        else:
            masks = tuple(
                _integer(v, "mask") for v in self.masks
            )

            if len(masks) != 2**m:
                raise ValueError("Exactly 2^m masks are required.")
            if any(not 0 <= v < 2**n for v in masks):
                raise ValueError("Every mask must fit into n bits.")
            if len(set(masks)) != len(masks):
                raise ValueError("XOR masks must be distinct.")

            standard_xor = all(
                mask == t for t, mask in enumerate(masks)
            )
            object.__setattr__(self, "masks", masks)

        object.__setattr__(self, "_standard_xor", standard_xor)

        # One generator is used for reproducible experiment draws.
        rng = np.random.default_rng(self.seed)
        object.__setattr__(self, "_rng", rng)

        generated = key_matrices is None
        object.__setattr__(self, "random_keys_generated", generated)

        if generated:
            keys = tuple(
                _binary_matrix(
                    rng.integers(
                        0, 2, size=(s, m), dtype=np.uint8
                    ),
                    (s, m),
                )
                for _ in range(nu)
            )
        else:
            if len(key_matrices) != nu:
                raise ValueError("Exactly nu key matrices are required.")

            keys = tuple(
                _binary_matrix(B, (s, m))
                for B in key_matrices
            )

        object.__setattr__(self, "_keys", keys)

    # =============================================================
    # 1. MODEL CONFIGURATION
    # =============================================================

    @property
    def paper_exact(self) -> bool:
        """Whether the circuit architecture matches Section IV.

        This does not certify secrecy, fresh keys, or the
        assumptions of the security experiment.
        """
        return self.m == self.n and self._standard_xor

    @property
    def num_labels(self) -> int:
        """Number of index labels K = 2^m."""
        return 2**self.m

    @property
    def num_qubits(self) -> int:
        """Qubit width of a single block."""
        return self.m + self.s + self.n

    @property
    def keys(self) -> tuple[np.ndarray, ...]:
        """Return copies of all block key matrices."""
        return tuple(B.copy() for B in self._keys)

    def _check_block(self, block: int) -> int:
        """Validate a zero-based block index."""
        block = _integer(block, "block")

        if not 0 <= block < self.nu:
            raise IndexError("Block index out of range.")

        return block

    def _check_input(self, x: int) -> int:
        """Validate an n-bit computational-basis input."""
        x = _integer(x, "x")

        if not 0 <= x < 2**self.n:
            raise ValueError("Input x does not fit into n bits.")

        return x

    def get_key(self, block: int = 0) -> np.ndarray:
        """Return an independent copy of B_j."""
        return self._keys[self._check_block(block)].copy()

    @property
    def stacked_matrix(self) -> np.ndarray:
        """Return A = [B_1; ...; B_nu] for the paper model.

        The stacked matrix determines whether authorized
        decoding has a unique solution.
        """
        if not self.paper_exact:
            raise ValueError(
                "The paper's stacked decoding system requires m=n "
                "and the standard XOR permutation family."
            )

        return np.vstack(self._keys)

    # =============================================================
    # 2. PERMUTATION FAMILY
    # =============================================================

    def xor_mask(self, t: int) -> int:
        """Return the XOR mask associated with label t."""
        t = _integer(t, "t")

        if not 0 <= t < self.num_labels:
            raise ValueError("Invalid permutation label.")

        return t if self.masks is None else self.masks[t]

    def permute(self, t: int, x: int) -> int:
        """Evaluate pi_t(x) = x XOR mask[t]."""
        x = self._check_input(x)
        return x ^ self.xor_mask(t)

    def permutation_table(
        self,
        input_x: int = 0,
        full: bool = False,
        max_labels: int = 16,
    ) -> pd.DataFrame:
        """Inspect selected permutations in a readable table.

        The complete mapping for each permutation can be shown
        only when n <= 4, to avoid exponentially large output.
        """
        input_x = self._check_input(input_x)
        max_labels = _integer(max_labels, "max_labels")

        if max_labels < 1:
            raise ValueError("max_labels must be positive.")
        if full and self.n > 4:
            raise ValueError("Full mappings require n <= 4.")

        rows = []

        for t in range(min(self.num_labels, max_labels)):
            row = {
                "Label t": f"{t:0{self.m}b}",
                "Mask": f"{self.xor_mask(t):0{self.n}b}",
            }

            if full:
                row["Mapping"] = ", ".join(
                    f"{a:0{self.n}b}->{self.permute(t, a):0{self.n}b}"
                    for a in range(2**self.n)
                )
            else:
                row["Output"] = (
                    f"{self.permute(t, input_x):0{self.n}b}"
                )

            rows.append(row)

        return pd.DataFrame(rows)

    # =============================================================
    # 3. QUANTUM CIRCUIT CONSTRUCTION
    # =============================================================

    def circuit(
        self,
        x: int,
        block: int = 0,
        measure: bool = False,
        barriers: bool = True,
    ) -> QuantumCircuit:
        """Construct one block using the prescribed gate sequence.

        Stages
        ------
        1. Prepare the computational-basis system state |x>.
        2. Apply H to each index qubit.
        3. Compute the secret tag B_j t using CNOTs.
        4. Apply XOR SELECT.
        5. Apply H to each index qubit.
        6. Optionally measure all registers.

        For the paper-exact construction, this produces the
        state given by Eq. (9) before measurement.

        Barriers are optional visualization aids.
        """
        x = self._check_input(x)
        block = self._check_block(block)

        I = QuantumRegister(self.m, "I")
        R = QuantumRegister(self.s, "R")
        S = QuantumRegister(self.n, "S")

        qc = QuantumCircuit(I, R, S, name=f"Block_{block + 1}")

        # Prepare |x> in the system register.
        for q in range(self.n):
            if (x >> q) & 1:
                qc.x(S[q])

        if barriers:
            qc.barrier()

        # Create a uniform superposition of permutation labels.
        qc.h(I)

        if barriers:
            qc.barrier()

        # Secret tag operation: |t>|0> -> |t>|B_j t>.
        B = self._keys[block]

        for ell in range(self.s):
            for q in range(self.m):
                if B[ell, q]:
                    qc.cx(I[q], R[ell])

        if barriers:
            qc.barrier()

        # Apply the permutation selected by the index register.
        if self._standard_xor:
            # Standard XOR SELECT uses pairwise CNOT gates.
            for q in range(self.m):
                qc.cx(I[q], S[q])
        else:
            # Custom masks use label-controlled X gates.
            # This is an exploratory, potentially expensive path.
            for t in range(self.num_labels):
                mask = self.xor_mask(t)

                if mask == 0:
                    continue

                controlled_x = XGate().control(
                    num_ctrl_qubits=self.m,
                    ctrl_state=t,
                )

                for q in range(self.n):
                    if (mask >> q) & 1:
                        qc.append(controlled_x, [*I, S[q]])

        if barriers:
            qc.barrier()

        # Final Hadamard layer produces the index-dependent signs.
        qc.h(I)

        if measure:
            # Qiskit displays counts in reverse register order:
            # cS cR cI, with each register's MSB on the left.
            cI = ClassicalRegister(self.m, "cI")
            cR = ClassicalRegister(self.s, "cR")
            cS = ClassicalRegister(self.n, "cS")

            qc.add_register(cI, cR, cS)

            qc.measure(I, cI)
            qc.measure(R, cR)
            qc.measure(S, cS)

        return qc

    def circuits(
        self,
        x: int,
        measure: bool = True,
        barriers: bool = True,
    ) -> tuple[QuantumCircuit, ...]:
        """Construct one separate circuit for every keyed block."""
        return tuple(
            self.circuit(
                x=x,
                block=j,
                measure=measure,
                barriers=barriers,
            )
            for j in range(self.nu)
        )

    def parse_outcome(self, bitstring: str) -> Transcript:
        """Parse the Qiskit cS cR cI register ordering.

        Spaces between classical registers are permitted.
        The same bit ordering is used by Statevector bitstrings
        for the quantum registers ordered as I, R, S.
        """
        if not isinstance(bitstring, str):
            raise TypeError("bitstring must be a string.")

        z = "".join(bitstring.split())

        if len(z) != self.num_qubits or any(
            bit not in "01" for bit in z
        ):
            raise ValueError("Invalid measurement bitstring.")

        return Transcript(
            i=int(z[-self.m:], 2),
            r=int(z[-(self.m + self.s):-self.m], 2),
            y=int(z[:self.n], 2),
        )

    # =============================================================
    # 4. PROPOSITION 4: EFFICIENT CLASSICAL SAMPLER
    # =============================================================

    def classical_sample(
        self,
        x: int,
        block: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> Transcript:
        """Generate one transcript using Proposition 4.

        Algorithm (paper-exact mode)
        ----------------------------
        1. Draw independent uniform T, I in GF(2)^n.
        2. Compute Y = x XOR T.
        3. Compute R = B_j T over GF(2).
        4. Return Z = (I, R, Y).

        No quantum circuit or statevector is involved.

        Complexity
        ----------
        O(s*m) binary operations per sample, excluding random
        bit generation and output representation.

        Parameters
        ----------
        x : int
            Computational-basis input.
        block : int
            Zero-based index of the keyed block.
        rng : np.random.Generator, optional
            External generator for reproducible independent runs.
            Defaults to the model's internal generator.

        Returns
        -------
        Transcript
            A complete ideal measurement record.

        Notes
        -----
        This method also reproduces the ideal distribution for
        distinct custom XOR masks, but Proposition 4 is stated
        in the paper for m=n and pi_t(x)=x XOR t.
        """
        x = self._check_input(x)
        block = self._check_block(block)
        rng = self._rng if rng is None else rng

        # Step 1: independent uniform binary vectors.
        t_bits = rng.integers(
            0, 2, size=self.m, dtype=np.uint8
        )
        i_bits = rng.integers(
            0, 2, size=self.m, dtype=np.uint8
        )

        T = bits_to_int(t_bits)
        I = bits_to_int(i_bits)

        # Step 2: selected permutation output.
        Y = self.permute(T, x)

        # Step 3: secret binary linear tag.
        R = tag_value(self._keys[block], T, self.m)

        # Step 4: complete measured transcript.
        return Transcript(i=I, r=R, y=Y)

    def classical_counts(
        self,
        x: int,
        shots: int,
        block: int = 0,
        seed: Optional[int] = None,
    ) -> dict[Transcript, int]:
        """Repeat the efficient classical sampler for finite shots.

        Unlike classical_probabilities(), this method samples
        individual transcripts without enumerating the support.
        """
        shots = _integer(shots, "shots")

        if shots < 1:
            raise ValueError("shots must be positive.")

        rng = (
            np.random.default_rng(seed)
            if seed is not None
            else self._rng
        )

        return dict(Counter(
            self.classical_sample(x, block=block, rng=rng)
            for _ in range(shots)
        ))

    def sample_all_blocks(self, x: int) -> list[Transcript]:
        """Sample exactly one execution of every keyed block.

        All blocks receive the same x, use their respective
        fixed keys, and independently draw their labels.

        Returns the complete transcript Z_nu.
        """
        self._check_input(x)

        return [
            self.classical_sample(x, block=j)
            for j in range(self.nu)
        ]

    # =============================================================
    # 5. QUANTUM EXPERIMENTS
    # =============================================================

    def quantum_counts(
        self,
        x: int,
        shots: int,
        block: int = 0,
        seed: Optional[int] = None,
    ) -> dict[Transcript, int]:
        """Execute one block with finite shots using Qiskit Aer.

        The default Aer simulator is ideal (no noise model).
        A transpiled copy is executed without modifying the
        original logical circuit or its resource accounting.

        Returns
        -------
        dict[Transcript, int]
            Observed quantum measurement counts.
        """
        from qiskit_aer import AerSimulator

        shots = _integer(shots, "shots")

        if shots < 1:
            raise ValueError("shots must be positive.")

        qc = self.circuit(x, block=block, measure=True)
        simulator = AerSimulator()
        executable = transpile(qc, simulator, optimization_level=0)

        options = {"shots": shots}

        if seed is not None:
            options["seed_simulator"] = _integer(seed, "seed")

        result = simulator.run(executable, **options).result()

        return {
            self.parse_outcome(bitstring): int(count)
            for bitstring, count in result.get_counts().items()
        }

    # =============================================================
    # 6. EXACT PROBABILITY DISTRIBUTIONS
    # =============================================================

    def quantum_probabilities(
        self,
        x: int,
        block: int = 0,
    ) -> dict[Transcript, float]:
        """Compute the ideal distribution via the Qiskit statevector.

        This method evolves the quantum amplitudes and then
        applies the Born rule to all computational-basis outcomes.

        It does not perform finite-shot sampling.

        Limited to at most 18 qubits to avoid excessive memory
        requirements in educational experiments.
        """
        self._check_input(x)
        self._check_block(block)

        if self.num_qubits > 18:
            raise ValueError(
                "Statevector analysis is limited to 18 qubits."
            )

        qc = self.circuit(x, block=block, measure=False)
        state = Statevector.from_instruction(qc)

        return {
            self.parse_outcome(z): float(p)
            for z, p in state.probabilities_dict().items()
            if p > 1e-14
        }

    def classical_probabilities(
        self,
        x: int,
        block: int = 0,
    ) -> dict[Transcript, float]:
        """Enumerate the exact classical transcript distribution.

        This is an exhaustive reference distribution, not the
        efficient classical sampler of Proposition 4.

        Each pair (T, I) has probability 2^(-2m).
        Distinct XOR masks guarantee unique system outputs
        for different labels at fixed x.

        Enumeration is limited to m <= 8.
        """
        x = self._check_input(x)
        block = self._check_block(block)

        if self.m > 8:
            raise ValueError(
                "Exact enumeration is limited to m <= 8."
            )

        probabilities = {}
        weight = 1.0 / self.num_labels**2

        for T in range(self.num_labels):
            Y = self.permute(T, x)
            R = tag_value(self._keys[block], T, self.m)

            for I in range(self.num_labels):
                probabilities[Transcript(I, R, Y)] = weight

        return probabilities

    # =============================================================
    # 7. AUTHORIZED DECODING
    # =============================================================

    def decode(
        self,
        records: Sequence[Transcript],
    ) -> Optional[int]:
        """Recover x from one ideal transcript per keyed block.

        Authorized decoding uses the secret matrices B_j and

            c_j = r_j XOR B_j y_j = B_j x.

        Stacking all equations gives Ax = c.

        Returns None when unique decoding is impossible or the
        observed equations are inconsistent.

        This method requires the paper-exact construction.
        """
        if not self.paper_exact:
            raise ValueError(
                "The paper's decoder requires the exact XOR model."
            )

        if len(records) != self.nu:
            raise ValueError("Expected one transcript per block.")

        c_parts = []

        for j, record in enumerate(records):
            if not isinstance(record, Transcript):
                raise TypeError("Expected Transcript objects.")

            if not (
                0 <= record.i < 2**self.n
                and 0 <= record.r < 2**self.s
                and 0 <= record.y < 2**self.n
            ):
                raise ValueError("Transcript outcome out of range.")

            r_bits = int_to_bits(record.r, self.s)
            y_bits = int_to_bits(record.y, self.n)

            # The secret key removes the random XOR label.
            c_j = r_bits ^ ((self._keys[j] @ y_bits) % 2)
            c_parts.append(c_j)

        solution = _solve_gf2(
            self.stacked_matrix,
            np.concatenate(c_parts),
        )

        return None if solution is None else bits_to_int(solution)

    # =============================================================
    # 8. TABLE I: LOGICAL RESOURCE ACCOUNTING
    # =============================================================

    @property
    def resources(self) -> dict:
        """Return per-block and aggregate logical resource metrics.

        Only the paper-exact circuit is supported here.

        Logical depth assumes all-to-all connectivity and optimal
        scheduling of the tag CNOTs. Physical routing, input
        preparation, error correction, and noise are excluded.

        Qubit count is not multiplied by nu because the blocks
        are executed sequentially and registers may be reused.
        """
        if not self.paper_exact:
            raise ValueError(
                "Table I applies only to the paper-exact model."
            )

        per_block = []

        for j, B in enumerate(self._keys):
            weight = int(np.count_nonzero(B))
            delta = _tag_depth(B)

            per_block.append({
                "block": j + 1,
                "rank_B": gf2_rank(B),
                "executions": 1,
                "qubits": 2 * self.n + self.s,
                "hadamards": 2 * self.n,
                "cnot": self.n + weight,
                "expected_cnot": self.n + self.s * self.n / 2,
                "logical_two_qubit_depth": 1 + delta,
                "measured_bits": 2 * self.n + self.s,
                "secret_key_bits": self.s * self.n,
                "tag_cnot": weight,
                "tag_depth": delta,
            })

        total = {
            "executions": self.nu,
            "qubits": 2 * self.n + self.s,
            "hadamards": 2 * self.nu * self.n,
            "cnot": sum(b["cnot"] for b in per_block),
            "expected_cnot": self.nu * (
                self.n + self.s * self.n / 2
            ),
            "logical_two_qubit_depth": sum(
                b["logical_two_qubit_depth"]
                for b in per_block
            ),
            "measured_bits": self.nu * (2 * self.n + self.s),
            "secret_key_bits": self.nu * self.s * self.n,
            "classical_decoding": "O(nu * s * n^2)",
        }

        # These probabilistic bounds concern the ensemble of
        # independent uniform keys, not the realized fixed key.
        correctness_bound = None
        inversion_bound = None

        if self.random_keys_generated:
            correctness_bound = min(
                1.0,
                (2**self.n - 1) / 2**(self.nu * self.s),
            )

            if self.nu < 2**self.n:
                inversion_bound = min(
                    1.0,
                    2**self.s / (2**self.n - self.nu)
                    + self.nu * (self.nu - 1) / 2**(self.n + 1),
                )

        rank_A = gf2_rank(self.stacked_matrix)

        return {
            "per_block": per_block,
            "total": total,
            "rank_A": rank_A,
            "full_column_rank": rank_A == self.n,
            "correctness_failure_bound": correctness_bound,
            "inversion_success_bound": inversion_bound,
        }

    def resource_table(self) -> pd.DataFrame:
        """Return a Table I-style resource comparison."""
        report = self.resources
        blocks = report["per_block"]
        total = report["total"]

        names = [
            ("Executions", "executions"),
            ("Qubits", "qubits"),
            ("Hadamard gates", "hadamards"),
            ("CNOT gates", "cnot"),
            ("Expected CNOT gates", "expected_cnot"),
            ("Logical two-qubit depth", "logical_two_qubit_depth"),
            ("Measured bits", "measured_bits"),
            ("Secret-key bits", "secret_key_bits"),
        ]

        table = []

        for label, key in names:
            row = {"Resource": label}

            for j, block in enumerate(blocks):
                row[f"Block {j + 1}"] = block[key]

            row["Total"] = total[key]
            table.append(row)

        table.append({
            "Resource": "Classical decoding",
            "Total": total["classical_decoding"],
        })

        return pd.DataFrame(table).set_index("Resource")

    # =============================================================
    # 9. MODEL INSPECTION
    # =============================================================

    def summary(self, show_keys: bool = True) -> str:
        """Build a human-readable model configuration report.

        No output is produced during initialization.
        The caller explicitly decides whether to print keys.

        Never publish this key-bearing report as an adversarial
        challenge transcript.
        """
        lines = [
            "=" * 54,
            "KEYED LINEAR-TAG MODEL",
            "=" * 54,
            f"Paper-exact construction : {self.paper_exact}",
            f"System qubits (n)        : {self.n}",
            f"Index qubits (m)         : {self.m}",
            f"Tag qubits (s)           : {self.s}",
            f"Keyed blocks (nu)        : {self.nu}",
            f"Permutation labels (K)  : {self.num_labels}",
            f"Qubits per block         : {self.num_qubits}",
            f"Seed                     : {self.seed}",
            "Key source               : "
            + ("Pseudorandom" if self.random_keys_generated
               else "User-specified"),
            "Permutation family       : "
            + ("pi_t(x) = x XOR t" if self._standard_xor
               else "Custom XOR masks"),
        ]

        if show_keys:
            for j, B in enumerate(self._keys):
                lines.extend([
                    "",
                    f"Secret key B_{j + 1}:",
                    np.array2string(B),
                    f"rank(B_{j + 1}) = {gf2_rank(B)}",
                ])

        if self.paper_exact:
            report = self.resources

            lines.extend([
                "",
                "DECODING INFORMATION",
                f"rank(A)                  : {report['rank_A']}",
                "Full column rank         : "
                f"{report['full_column_rank']}",
                "Random-key failure bound : "
                f"{report['correctness_failure_bound']}",
                "Random-key inversion bound: "
                f"{report['inversion_success_bound']}",
                "",
                "TOTAL LOGICAL RESOURCES",
            ])

            for name, value in report["total"].items():
                lines.append(f"{name:25s}: {value}")
        else:
            lines.extend([
                "",
                "Exploratory configuration:",
                "Table I and the paper's decoder do not apply.",
            ])

        return "\n".join(lines)

    def describe(self, show_keys: bool = True) -> None:
        """Print the current model's complete configuration."""
        print(self.summary(show_keys=show_keys))
