
# Circuit-Level Inversion Security for All-Outcomes Linear Combinations of Unitaries

## Overview

This repository contains the theoretical validation and numerical experiments accompanying the study **Circuit-Level Inversion Security for All-Outcomes Linear Combinations of Unitaries**.

The project investigates a **keyed linear-tag quantum construction**, focusing on the relationship between authorized input recovery and adversarial inversion security.

The implementation uses **Qiskit Aer**, binary linear algebra over **GF(2)**, exact probability calculations, and Monte Carlo experiments.

## Project Structure

```text
.
├── src/
│   └── linear_tag/
│       ├── __init__.py
│       ├── gf2.py
│       ├── transcript.py
│       └── model.py
├── notebooks/
│   ├── Circuit_level_inversion_security_LCU.ipynb
│   └── img/
├── res/
└── README.md
```

The `linear_tag` package implements the construction, binary matrix operations, measurement transcripts, and authorized decoding.

The Jupyter notebook contains the numerical experiments. Generated figures are exported as **vector PDF files** to the `res/` directory.
