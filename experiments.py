"""Experiments 2--5. Parameters and interpretation live in the companion notebook."""

import math

import numpy as np
import pandas as pd

from linear_tag import (
    compile_circuit, decode, linear_tag_circuit, parse_memory,
    permutation_lcu_probabilities, rank_failure_exact, rank_gf2,
    recover_reused_matrix, run_memory, sample_transcript, wilson,
)


def resource_experiment(ns, tag_widths, delta, key_trials, rank_trials,
                        topologies, seed, optimization_level=1, confidence=0.95):
    if delta < 0 or key_trials < 1 or rank_trials < 1:
        raise ValueError("Need nonnegative delta and positive trial counts.")
    resources, ranks = [], []
    for n in ns:
        if n < 1:
            raise ValueError("n must be positive.")
        for s in tag_widths:
            if not 1 <= s <= n:
                continue
            nu = math.ceil((n + delta) / s)
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, s]))
            failures = sum(
                rank_gf2(rng.integers(0, 2, (nu * s, n), dtype=np.uint8)) < n
                for _ in range(rank_trials)
            )
            lo, hi = wilson(failures, rank_trials, confidence)
            ranks.append(dict(
                n=n, s=s, nu=nu, rows=nu*s, failures=failures, trials=rank_trials,
                failure=failures/rank_trials, ci_low=lo, ci_high=hi,
                exact=rank_failure_exact(n, nu*s), union_bound=min(1., (2.**n-1)*2.**(-nu*s)),
            ))
            for trial in range(key_trials):
                matrices = rng.integers(0, 2, (nu, s, n), dtype=np.uint8)
                x = rng.integers(0, 2, n, dtype=np.uint8)
                logical_cx = nu*n + int(matrices.sum())
                scheduled_depth = sum(
                    1 + max(int(B.sum(axis=0).max()), int(B.sum(axis=1).max()))
                    for B in matrices
                )
                for topology in topologies:
                    circuits = [
                        compile_circuit(linear_tag_circuit(B, x), topology,
                                        seed + trial, optimization_level)
                        for B in matrices
                    ]
                    resources.append(dict(
                        n=n, s=s, nu=nu, trial=trial, topology=topology,
                        logical_qubits=2*n+s,
                        physical_qubits=max(q.num_qubits for q in circuits),
                        key_bits=nu*s*n, measured_bits=nu*(2*n+s),
                        logical_cx=logical_cx, expected_cx=nu*(n+s*n/2),
                        optimal_stagewise_depth=scheduled_depth,
                        compiled_cx=sum(q.count_ops().get("cx", 0) for q in circuits),
                        compiled_2q_depth=sum(
                            q.depth(filter_function=lambda inst: len(inst.qubits) == 2)
                            for q in circuits
                        ),
                        full_rank=rank_gf2(matrices.reshape(-1, n)) == n,
                    ))
    if not resources:
        raise ValueError("No valid (n, s) configuration.")
    return pd.DataFrame(resources), pd.DataFrame(ranks)


def noise_experiment(n, tag_widths, delta, trials, topologies, scales, modes,
                     p1, p2, p_meas, seed, optimization_level=1):
    """One fresh key set per independent challenge; exactly one shot per block."""
    if n < 1 or delta < 0 or trials < 1 or any(scale < 0 for scale in scales):
        raise ValueError("Invalid dimensions, trial count or noise scales.")
    if any(mode not in ("combined", "gates", "readout") for mode in modes):
        raise ValueError("Noise modes: combined, gates, readout.")
    cases = [("ideal", 0., 0., 0., 0.)]
    for mode in modes:
        for scale in scales:
            if scale > 0:
                cases.append((
                    mode, scale, scale*p1 if mode != "readout" else 0.,
                    scale*p2 if mode != "readout" else 0.,
                    scale*p_meas if mode != "gates" else 0.,
                ))
    records = []
    for s in tag_widths:
        if not 1 <= s <= n:
            raise ValueError("Noise tag widths must satisfy 1 <= s <= n.")
        nu = math.ceil((n + delta) / s)
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, s]))
        keys = rng.integers(0, 2, (trials, nu, s, n), dtype=np.uint8)
        xs = rng.integers(0, 2, (trials, n), dtype=np.uint8)
        ranks = [rank_gf2(k.reshape(-1, n)) for k in keys]
        for topology in topologies:
            circuits = [
                compile_circuit(linear_tag_circuit(B, x), topology,
                                seed + trial, optimization_level)
                for trial, (key, x) in enumerate(zip(keys, xs)) for B in key
            ]
            for case_id, (mode, scale, e1, e2, em) in enumerate(cases):
                memories = run_memory(circuits, 1, seed + case_id, e1, e2, em)
                for trial, (key, x) in enumerate(zip(keys, xs)):
                    transcript = [
                        parse_memory(memories[trial*nu+j][0], n, s) for j in range(nu)
                    ]
                    guess, _, status = decode(key, transcript)
                    outcome = (
                        "correct" if guess is not None and np.array_equal(guess, x)
                        else "wrong" if guess is not None else "failure"
                    )
                    records.append(dict(
                        n=n, s=s, nu=nu, trial=trial, topology=topology, mode=mode,
                        scale=scale, p1=e1, p2=e2, p_meas=em, rank=ranks[trial],
                        outcome=outcome, decoder_status=status,
                        compiled_cx=sum(q.count_ops().get("cx", 0)
                                        for q in circuits[trial*nu:(trial+1)*nu]),
                    ))
    return pd.DataFrame(records)


def noise_summary(records, confidence=0.95):
    rows = []
    grouping = ["n", "s", "nu", "topology", "mode", "scale"]
    for values, group in records.groupby(grouping, sort=False):
        row = dict(zip(grouping, values))
        row["trials"] = len(group)
        for outcome in ("correct", "failure", "wrong"):
            count = int((group.outcome == outcome).sum())
            lo, hi = wilson(count, len(group), confidence)
            row.update({outcome: count/len(group), outcome+"_low": lo, outcome+"_high": hi})
        full = group[group["rank"] == group.n]
        row["full_rank_trials"] = len(full)
        row["correct_given_full_rank"] = (
            float((full.outcome == "correct").mean()) if len(full) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def interference_experiment(n, phases, dephasings, weights, control=0, target=1):
    rows, distributions = [], []
    for family in ("disjoint", "overlap"):
        for phase in phases:
            incoherent = np.array([
                permutation_lcu_probabilities(n, x, family, phase, 1., weights, control, target)
                for x in range(2**n)
            ])
            for dephase in dephasings:
                channel = np.array([
                    permutation_lcu_probabilities(
                        n, x, family, phase, dephase, weights, control, target
                    ) for x in range(2**n)
                ])
                tv = np.abs(channel - incoherent).sum(axis=1)/2
                rows.append(dict(
                    family=family, phase=phase, dephase=dephase,
                    tv_mean=float(tv.mean()), tv_max=float(tv.max()),
                    pguess=float(channel.max(axis=0).sum()/2**n),
                ))
                for x, p in enumerate(channel):
                    distributions.extend(dict(
                        family=family, phase=phase, dephase=dephase, x=x,
                        outcome=z, probability=float(prob),
                    ) for z, prob in enumerate(p))
    return pd.DataFrame(rows), pd.DataFrame(distributions)


def reuse_experiment(n, s, nu, repetitions, trials, seed):
    if not 1 <= s <= n or nu < 1 or trials < 1 or not repetitions or min(repetitions) < 1:
        raise ValueError("Invalid reuse experiment parameters.")
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        key = rng.integers(0, 2, (nu, s, n), dtype=np.uint8)
        x = rng.integers(0, 2, n, dtype=np.uint8)
        for policy in ("reused", "fresh"):
            blocks = [[
                sample_transcript(
                    B if policy == "reused" else rng.integers(0, 2, (s, n), dtype=np.uint8),
                    x, rng,
                ) for _ in range(max(repetitions))
            ] for B in key]
            for q in repetitions:
                recovered = [recover_reused_matrix(block[:q])[0] for block in blocks]
                accepted = all(B is not None for B in recovered)
                correct, returned = False, False
                if accepted:
                    guess, _, _ = decode(recovered, [block[0] for block in blocks])
                    returned = guess is not None
                    correct = returned and np.array_equal(guess, x)
                rows.append(dict(
                    n=n, s=s, nu=nu, trial=trial, policy=policy, repetitions=q,
                    accepted=accepted, returned=returned, correct=bool(correct),
                    keys_exact=(
                        accepted and all(np.array_equal(a, b) for a, b in zip(recovered, key))
                        if policy == "reused" else np.nan
                    ),
                ))
    return pd.DataFrame(rows)
