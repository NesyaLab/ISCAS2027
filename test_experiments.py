"""Small deterministic checks of the scientific model: python -m unittest -v."""

import itertools
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from linear_tag import (
    bits_int, compile_circuit, decode, int_bits, linear_tag_circuit, noise_model,
    parse_memory, permutation_lcu_probabilities, rank_failure_exact, rank_gf2,
    recover_reused_matrix, run_memory, sample_transcript, solve_gf2, wilson,
)
from experiments import (
    interference_experiment, noise_experiment, noise_summary,
    resource_experiment, reuse_experiment,
)


class ModelTests(unittest.TestCase):
    def test_gf2_against_exhaustive_solutions(self):
        for entries in itertools.product((0, 1), repeat=6):
            A = np.array(entries, dtype=np.uint8).reshape(3, 2)
            for b_tuple in itertools.product((0, 1), repeat=3):
                b = np.array(b_tuple, dtype=np.uint8)
                solutions = [int_bits(x, 2) for x in range(4)
                             if np.array_equal(A @ int_bits(x, 2) % 2, b)]
                x, _, status = solve_gf2(A, b)
                if len(solutions) == 1:
                    np.testing.assert_array_equal(x, solutions[0])
                else:
                    self.assertIsNone(x)
                    self.assertEqual(status, "inconsistent" if not solutions else "rank_deficient")

    def test_rank_probability_exact(self):
        failures = sum(rank_gf2(np.array(b).reshape(3, 2)) < 2
                       for b in itertools.product((0, 1), repeat=6))
        self.assertAlmostEqual(failures / 64, rank_failure_exact(2, 3))
        self.assertEqual(rank_failure_exact(3, 2), 1)
        self.assertGreater(rank_failure_exact(32, 96), 0)

    def test_linear_tag_full_distribution(self):
        rng = np.random.default_rng(9)
        for n, s in ((1, 1), (2, 1), (3, 2)):
            B = rng.integers(0, 2, (s, n), dtype=np.uint8)
            for value in range(2**n):
                x = int_bits(value, n)
                p = Statevector.from_instruction(linear_tag_circuit(B, x, False)).probabilities()
                expected = np.zeros_like(p)
                for t in range(2**n):
                    r = bits_int(B @ int_bits(t, n) % 2)
                    for i in range(2**n):
                        expected[i + (r << n) + ((value ^ t) << (n+s))] = 2.**(-2*n)
                np.testing.assert_allclose(p, expected, atol=1e-14)

    def test_decode_sampler_and_failure_modes(self):
        rng = np.random.default_rng(4)
        keys = [np.eye(3, dtype=np.uint8), np.eye(3, dtype=np.uint8)]
        x = int_bits(5, 3)
        records = [sample_transcript(B, x, rng) for B in keys]
        np.testing.assert_array_equal(decode(keys, records)[0], x)
        corrupt = [(i.copy(), r.copy(), y.copy()) for i, r, y in records]
        corrupt[1][1][0] ^= 1
        self.assertEqual(decode(keys, corrupt)[2], "inconsistent")
        zeros = [np.zeros((1, 3), dtype=np.uint8)]
        self.assertEqual(decode(zeros, [sample_transcript(zeros[0], x, rng)])[2],
                         "rank_deficient")

    def test_routing_and_readout_noise(self):
        B, x = np.array([[1, 1], [1, 0]], dtype=np.uint8), int_bits(2, 2)
        for topology in ("all", "line", "grid"):
            qc = compile_circuit(linear_tag_circuit(B, x), topology, 3)
            ideal = run_memory([qc], shots=32, seed=8)[0]
            for mem in ideal:
                np.testing.assert_array_equal(decode([B], [parse_memory(mem, 2, 2)])[0], x)
            flipped = run_memory([qc], shots=32, seed=8, p_meas=1)[0]
            for mem in flipped:
                _, r, y = parse_memory(mem, 2, 2)
                syndrome = r ^ (B @ (x ^ y) % 2)
                np.testing.assert_array_equal(syndrome, np.ones(2, dtype=np.uint8)
                                              ^ (B @ np.ones(2, dtype=np.uint8) % 2))

    def test_gate_noise_attaches_to_compiled_basis(self):
        model = noise_model(.01, .02, .03)
        self.assertTrue({"h", "x", "cx", "measure"}.issubset(model.noise_instructions))
        B = np.eye(2, dtype=np.uint8)
        qc = compile_circuit(linear_tag_circuit(B, int_bits(0, 2)))
        memories = run_memory([qc], shots=128, seed=1, p2=1)[0]
        self.assertTrue(any(np.any(parse_memory(m, 2, 2)[1] ^ parse_memory(m, 2, 2)[2])
                            for m in memories))

    def test_reuse_attack_uses_only_records(self):
        B = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        x = int_bits(5, 3)
        ts = [np.zeros(3, dtype=np.uint8), *np.eye(3, dtype=np.uint8)]
        records = [(np.zeros(3, dtype=np.uint8), B @ t % 2, x ^ t) for t in ts]
        self.assertIsNone(recover_reused_matrix(records[:3])[0])
        recovered, rank, status = recover_reused_matrix(records)
        self.assertEqual((rank, status), (3, "unique"))
        np.testing.assert_array_equal(recovered, B)

    def test_interference_against_independent_born_formula(self):
        n, phase, weights = 2, .37, (.3, -.8)
        for family in ("overlap", "disjoint"):
            for x in range(4):
                amplitudes = np.zeros(16, dtype=complex)
                for i, b, t in itertools.product((0, 1), repeat=3):
                    y = x
                    if t and (family == "disjoint" or (x & 1)):
                        y ^= 2
                    a = weights[t] if b == 0 else np.sqrt(1 - weights[t]**2)
                    amplitudes[i + 2*b + 4*y] += (-1)**(i*t)*a*np.exp(1j*phase*t)/2
                p = permutation_lcu_probabilities(n, x, family, phase, weights=weights)
                np.testing.assert_allclose(p, abs(amplitudes)**2, atol=1e-14)

    def test_interference_controls(self):
        for x in range(4):
            coherent = permutation_lcu_probabilities(2, x, "disjoint", .2)
            incoherent = permutation_lcu_probabilities(2, x, "disjoint", .2, 1)
            np.testing.assert_allclose(coherent, incoherent, atol=1e-14)
        p = permutation_lcu_probabilities(2, 0, "overlap")
        q = permutation_lcu_probabilities(2, 0, "overlap", dephase=1)
        self.assertAlmostEqual(np.abs(p-q).sum()/2, .5)

    def test_invalid_inputs_and_confidence(self):
        with self.assertRaises(ValueError):
            solve_gf2([[2]], [0])
        with self.assertRaises(ValueError):
            noise_model(p2=-.1)
        with self.assertRaises(ValueError):
            int_bits(4, 2)
        lo, hi = wilson(0, 100)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, .03)

    def test_experiment_smoke(self):
        resources, ranks = resource_experiment([2], [1, 2], 1, 1, 8, ["all"], 7)
        self.assertEqual(len(resources), 2)
        self.assertEqual(len(ranks), 2)
        raw = noise_experiment(2, [1], 1, 4, ["line"], [0, 1],
                               ["combined"], .001, .005, .02, 7)
        summary = noise_summary(raw)
        self.assertEqual(len(summary), 2)
        ideal = raw[raw["mode"] == "ideal"]
        self.assertTrue((ideal[ideal["rank"] == 2].outcome == "correct").all())
        self.assertTrue(np.allclose(summary[["correct", "failure", "wrong"]].sum(axis=1), 1))
        interference, _ = interference_experiment(2, [0], [0, 1], [1, 1])
        self.assertEqual(len(interference), 4)
        reuse = reuse_experiment(2, 1, 3, [1, 4], 4, 9)
        self.assertEqual(len(reuse), 16)
        self.assertFalse(reuse[reuse.repetitions == 1].accepted.any())


if __name__ == "__main__":
    unittest.main()
