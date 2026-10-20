# Noise model provenance

`linear_tag.noise_model` adapts the single-/two-qubit depolarizing channels
and symmetric readout composition from the supplied 2026
`utilities/noise_level_and_topology.py` (file header: Leonardo Lavagna,
NESYA, 2025).

The original basis `u1,u2,u3,rx,ry,rz,cx` is adapted to the actual Clifford
basis `h,x,s,sdg,z,cx`. In particular, single-qubit noise must also affect
the Hadamards in the linear-tag circuit. Readout noise is uniform on all
measured qubits. Defaults match the original function: p1=0.001, p2=0.005,
p_meas=0.02. The 2026 notebooks also use p2=0.008, selectable in CONFIG.

The separate `NoiseModel.from_backend(FakeVigoV2())` in the old `config.py`
is not used. No claim of current hardware calibration is made.

The original source license is reproduced in full below.

MIT License

Copyright (c) 2024 Leonardo Lavagna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
