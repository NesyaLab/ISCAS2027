"""Measured linear-tag model from the ISCAS 2027 draft. Bit order is little-endian."""

import math
from statistics import NormalDist

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


BASIS = ["h", "x", "cx", "s", "sdg", "z"]


def binary(value, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"Expected a {ndim}-dimensional binary array.")
    return a.astype(np.uint8, copy=True)


def int_bits(value, n):
    if n < 1 or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**n:
        raise ValueError("Require n >= 1 and integer 0 <= value < 2**n.")
    return np.array([(int(value) >> q) & 1 for q in range(n)], dtype=np.uint8)


def bits_int(bits):
    return sum(int(b) << q for q, b in enumerate(binary(bits, 1)))


def solve_gf2(A, rhs):
    """Return (unique solution or None, coefficient rank, status), also for matrix RHS."""
    A = binary(A, 2)
    b = np.asarray(rhs)
    vector = b.ndim == 1
    b = binary(b, 1 if vector else 2)
    if vector:
        b = b[:, None]
    m, n = A.shape
    if n < 1 or b.shape[0] != m:
        raise ValueError("Incompatible system dimensions.")
    aug = np.concatenate([A, b], axis=1)
    pivots = []
    for col in range(n):
        row = len(pivots)
        candidates = np.flatnonzero(aug[row:, col])
        if not len(candidates):
            continue
        pivot = row + candidates[0]
        aug[[row, pivot]] = aug[[pivot, row]]
        others = np.flatnonzero(aug[:, col])
        others = others[others != row]
        aug[others] ^= aug[row]
        pivots.append(col)
    rank = len(pivots)
    if np.any(aug[rank:, n:]):
        return None, rank, "inconsistent"
    if rank < n:
        return None, rank, "rank_deficient"
    solution = aug[:n, n:]
    return (solution[:, 0] if vector else solution), rank, "unique"


def rank_gf2(A):
    A = binary(A, 2)
    return solve_gf2(A, np.zeros(A.shape[0], dtype=np.uint8))[1]


def rank_failure_exact(n, rows):
    if n < 1 or rows < 0:
        raise ValueError("Require n >= 1 and rows >= 0.")
    if rows < n:
        return 1.0
    # log1p/expm1 retain small failure probabilities instead of subtracting near-equals.
    return -math.expm1(sum(math.log1p(-2.0 ** (i - rows)) for i in range(n)))


def wilson(successes, trials, confidence=0.95):
    if not 0 <= successes <= trials or trials < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid binomial count or confidence.")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / den
    return max(0.0, center - half), min(1.0, center + half)


def sample_transcript(B, x, rng):
    """Exact ideal sampler; returns the complete (I, R, Y) transcript."""
    B, x = binary(B, 2), binary(x, 1)
    if B.shape[1] != len(x):
        raise ValueError("B and x dimensions differ.")
    t = rng.integers(0, 2, len(x), dtype=np.uint8)
    i = rng.integers(0, 2, len(x), dtype=np.uint8)
    return i, (B @ t) % 2, x ^ t


def linear_tag_circuit(B, x, measure=True):
    B, x = binary(B, 2), binary(x, 1)
    s, n = B.shape
    if not 1 <= s <= n or len(x) != n:
        raise ValueError("Require 1 <= s <= n and len(x) == n.")
    width = 2 * n + s
    qc = QuantumCircuit(width, width if measure else 0, name=f"linear_tag_n{n}_s{s}")
    for q in np.flatnonzero(x):
        qc.x(n + s + int(q))
    qc.h(range(n))
    qc.barrier()
    for row, col in np.argwhere(B):
        qc.cx(int(col), n + int(row))
    qc.barrier()
    for q in range(n):
        qc.cx(q, n + s + q)
    qc.barrier()
    qc.h(range(n))
    if measure:
        qc.measure(range(width), range(width))
    return qc


def parse_memory(memory, n, s):
    # Classical indices survive routing: I[0:n], R[n:n+s], Y[n+s:2n+s].
    text = memory.replace(" ", "")
    if len(text) != 2 * n + s or any(c not in "01" for c in text):
        raise ValueError("Unexpected measurement memory.")
    b = np.array([int(c) for c in text[::-1]], dtype=np.uint8)
    return b[:n], b[n:n + s], b[n + s:]


def decode(matrices, transcripts):
    if not len(matrices) or len(matrices) != len(transcripts):
        raise ValueError("Need one transcript per keyed block.")
    blocks, values = [], []
    for B, (_, r, y) in zip(matrices, transcripts):
        B, r, y = binary(B, 2), binary(r, 1), binary(y, 1)
        if B.shape != (len(r), len(y)):
            raise ValueError("Transcript and key dimensions differ.")
        blocks.append(B)
        values.append(r ^ ((B @ y) % 2))
    return solve_gf2(np.vstack(blocks), np.concatenate(values))


def coupling_map(topology, width):
    if topology == "all":
        return None
    if topology == "line":
        return CouplingMap.from_line(width, bidirectional=True)
    if topology == "grid":
        cols = math.ceil(math.sqrt(width))
        return CouplingMap.from_grid(math.ceil(width / cols), cols, bidirectional=True)
    raise ValueError("topology must be 'all', 'line', or 'grid'.")


def compile_circuit(qc, topology="all", seed=0, optimization_level=1):
    return transpile(
        qc, basis_gates=BASIS, coupling_map=coupling_map(topology, qc.num_qubits),
        initial_layout=list(range(qc.num_qubits)), seed_transpiler=int(seed),
        optimization_level=optimization_level,
    )


def noise_model(p1=1e-3, p2=5e-3, p_meas=2e-2):
    """2026 depolarizing + symmetric readout model, aligned to the Clifford basis.

    Adapted from Leonardo Lavagna's noise_level_and_topology.py; see THIRD_PARTY.md.
    Parameters p1/p2 are Aer depolarizing parameters, not average gate infidelities.
    """
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in (p1, p2, p_meas)):
        raise ValueError("Noise parameters must be finite and in [0, 1].")
    nm = NoiseModel(basis_gates=BASIS)
    if p1:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p1, 1), ["h", "x", "s", "sdg", "z"]
        )
    if p2:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if p_meas:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
        )
    return nm


def run_memory(circuits, shots=1, seed=0, p1=0, p2=0, p_meas=0):
    if not circuits or shots < 1:
        raise ValueError("Need circuits and positive shots.")
    backend = AerSimulator(
        method="stabilizer", noise_model=noise_model(p1, p2, p_meas),
        max_parallel_threads=1,
    )
    result = backend.run(
        circuits, shots=int(shots), memory=True, seed_simulator=int(seed)
    ).result()
    if not result.success:
        raise RuntimeError(result.status)
    return [result.get_memory(j) for j in range(len(circuits))]


def recover_reused_matrix(transcripts):
    """Ciphertext-only attack for repeated executions with the SAME B and SAME x."""
    if not transcripts:
        raise ValueError("Need at least one transcript.")
    _, r0, y0 = transcripts[0]
    dy = np.array([y ^ y0 for _, _, y in transcripts[1:]], dtype=np.uint8)
    dr = np.array([r ^ r0 for _, r, _ in transcripts[1:]], dtype=np.uint8)
    dy, dr = dy.reshape(-1, len(y0)), dr.reshape(-1, len(r0))
    solution, rank, status = solve_gf2(dy, dr)
    return (None if solution is None else solution.T), rank, status


def permutation_lcu_prefix(n, x, family, phase, weights, control=0, target=1):
    """K=2 all-outcomes circuit: index, rotation, n system qubits; before final H."""
    if n < 2 or not 0 <= control < n or not 0 <= target < n or control == target:
        raise ValueError("Need n >= 2 and distinct valid control/target.")
    if family not in ("disjoint", "overlap"):
        raise ValueError("Unknown permutation family.")
    if len(weights) != 2 or not all(np.isfinite(w) and abs(w) <= 1 for w in weights):
        raise ValueError("Need two real weights in [-1, 1].")
    if not np.isfinite(phase):
        raise ValueError("Phase must be finite.")
    qc = QuantumCircuit(n + 2)
    for q in np.flatnonzero(int_bits(x, n)):
        qc.x(2 + int(q))
    qc.h(0)
    for t, w in enumerate(weights):
        if t == 0:
            qc.x(0)
        qc.cz(0, 1)
        qc.cry(2 * math.acos(w), 0, 1)  # R_t = Ry(2 acos w_t) Z.
        if t == 0:
            qc.x(0)
    if family == "disjoint":
        qc.cx(0, 2 + target)
    else:
        qc.ccx(0, 2 + control, 2 + target)
    qc.p(phase, 0)
    return qc


def permutation_lcu_probabilities(n, x, family, phase=0, dephase=0,
                                  weights=(1.0, 1.0), control=0, target=1):
    if not 0 <= dephase <= 1:
        raise ValueError("dephase must be in [0, 1].")
    # shortcut: exhaustive density matrices, use a different simulator beyond six system qubits.
    if n > 6:
        raise ValueError("This exhaustive interference benchmark supports n <= 6.")
    qc = permutation_lcu_prefix(n, x, family, phase, weights, control, target)
    rho = DensityMatrix.from_instruction(qc)
    channel = Kraus([
        math.sqrt(1 - dephase / 2) * np.eye(2),
        math.sqrt(dephase / 2) * np.diag([1, -1]),
    ])
    rho = rho.evolve(channel, qargs=[0])
    final = QuantumCircuit(n + 2)
    final.h(0)
    p = np.asarray(rho.evolve(final).probabilities(), dtype=float)
    if np.min(p) < -1e-10 or not np.isclose(p.sum(), 1):
        raise ArithmeticError("Invalid circuit distribution.")
    p = np.maximum(p, 0)
    return p / p.sum()
