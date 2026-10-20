# ISCAS 2027 — Linear-tag LCU experiments

Start with **`iscas2027_experiments.ipynb`**. It contains explicit mathematical
models, the actual shared model implementations, configurable experiments,
theory, figures and result interpretation, all in English.

| Experiment | Research question |
|---|---|
| 2 — Resources and rank | How do tag width and topology affect qubits, gates, depth, key size and decoding? |
| 3 — Noise | How often is the entire input recovered with gate noise, readout noise and both? |
| 4 — Interference | When do phases and output collisions change the transcript and input recovery? |
| 5 — Key reuse | How many repeated executions on the same input enable a linear key-recovery attack? |

The opening demo compares the linear-tag circuit with its exact formula and
shows ideal and noisy outcome distributions.

## Receive a ZIP and run it

Requirements: **64-bit Python 3.12**, free disk space for a scientific Python
environment, and an Internet connection for the first dependency installation.
No IBM account, QPU access or access to the original 2026 repository is needed.

1. Extract the ZIP to a writable local folder.
2. On Windows, double-click **`start.cmd`**. It uses the existing environment's
   base Python when available, otherwise `py -3.12`, falling back to `python`
   if the Python launcher is unavailable.
3. The launcher creates `.venv`, installs pinned dependencies and registers a
   kernel **inside this environment**, then opens JupyterLab.
4. Select **ISCAS 2027 (.venv)**, edit the notebook's `CONFIG` cell and choose
   **Restart Kernel and Run All**.

Alternatively, from a terminal in this folder:

```text
python3.12 start.py          # macOS/Linux with a compatible Python 3.12 installation
py -3.12 start.py            # Windows
py -3.12 start.py --check    # dependency consistency and all tests
py -3.12 start.py --run      # execute/save the entire notebook without the browser
py -3.12 start.py --package  # build the shareable ZIP beside the project folder
```

On Windows, `start.cmd --check`, `start.cmd --run` and `start.cmd --package`
are equivalent. If `py` is unavailable, use an explicit Python 3.12 executable.
Do not run the launcher with a different Python minor version.
Windows is the validated platform; macOS/Linux installation and binary-wheel
availability have not been tested.

After setup, the installed environment can run offline. Installing dependencies
for a new recipient is not an offline operation. The archive does not bundle
a standalone Python interpreter.

## Share the project correctly

Run **`start.cmd --package`** to produce `iscas2027_experiments_share.zip` in
the parent directory. It includes the notebook, source, tests, launchers,
dependency files, documentation and saved results.

**A copied venv is not portable.** It can contain absolute paths and binaries
tied to a Python installation and operating system. A populated `.venv` is
provided for this local checkout, but the shareable ZIP deliberately excludes
it and recreates it on the recipient's machine. No path belonging to the original
author is required by the launcher. The full `requirements-lock.txt` pins the
resolved dependency set; `requirements.txt` lists the direct dependencies.

If the whole folder is moved manually with its environment, the launcher detects
the recorded path change, preserves the old environment in `.venv-moved-<time>`
and creates a new `.venv`. Such backups are also excluded from generated ZIPs.
The packaging command replaces the previous generated share ZIP; source files
and results are not deleted. `--run` saves new notebook outputs only after a
successful full execution.

## Files and outputs

- `linear_tag.py`: circuit, exact sampler, GF(2) solver, decoder, noise model,
  compilation, small permutation LCU families and repeated-key attack.
- `experiments.py`: experiment drivers called from the notebook.
- `test_experiments.py`, `test_start.py`: scientific and packaging checks.
- `start.py`, `start.cmd`: setup, launch, validation, execution and ZIP creation.
- `requirements-lock.txt`: pinned resolved dependencies; platform-specific
  packages carry environment markers.
- `results/<UTC timestamp>/`: raw/summary CSV, PNG/PDF figures and configuration.
- `THIRD_PARTY.md`: attribution and license for the 2026 noise-model source.

The delivered notebook retains executed outputs. Each new campaign creates a
new results directory instead of overwriting previous campaigns. `results/`
is Git-ignored to avoid automatically versioning every run, but it **is included
in the share ZIP**. The two delivered campaigns are explicitly tracked in Git.
Scientific parameters are all in `CONFIG`; the defaults are
exploratory and need more repetitions for publication-quality comparisons.

## Validation and scientific scope

Tests independently check the GF(2) solver by exhaustive enumeration, exact
rank probabilities, circuit/transcript agreement, routing, actual noise
attachment, the permutation Born formula, the reuse attack and archive contents.
The original campaign's maximum statevector/formula error was approximately
1.4e-17. All 12 tests and all 12 notebook code cells passed in the project's
isolated venv. A separate clean extraction of the share ZIP successfully
created a fresh environment from the lock file and passed all 12 tests.

Inputs and keys are uniform; singular keys are not discarded. Experiment 3
uses independent keys across blocks, one shot per block, and independent
challenges. Matched keys across conditions support comparisons, but these
conditions are never concatenated into an adversary's transcript.

Decoding distinguishes exact success, declared failure and a wrong unique
solution. Rank redundancy is not error correction. The synthetic depolarizing
and readout model is adapted from the 2026 code to the actual Clifford basis;
it is not a current device calibration.

Experiment 4 uses public parameters and exact distributions. Experiment 5
explicitly changes the threat model: repeated use of the same matrix on the
same input without noise. Its attack success is not claimed to be optimal.
No experiment establishes quantum advantage or production-cipher security.

No real-hardware campaign or publication-scale statistical campaign is included.
