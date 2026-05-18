# Lean 4 companion — note

The file [`T1ForbiddenTransitions.lean`](T1ForbiddenTransitions.lean) is a
**standalone snapshot** of the Lean 4 proof of theorem T1 (forbidden modular
transitions). It depends only on `Mathlib.Tactic` and can be checked in
isolation:

```sh
lake init T1Lean math
cp T1ForbiddenTransitions.lean T1Lean/T1Lean/
cd T1Lean && lake exe cache get && lake build
```

For the **full and current Lean 4 formalisation** of M1–M5 — including the
modules `PT.Sieve.SixRough`, `PT.Sieve.T3Antidiagonal`, `PT.Stochastic.SHalf`,
`PT.Conservation.T2Alpha`, `PT.Sieve.N2UniqueFixedPoint`, `PT.Sieve.L0Uniqueness`,
`PT.Bekenstein`, `PT.ActivePrimes`, `PT.T6bAxiomsFull`, `PT.T2SpectralBridge`,
and the T1→T7→W7-1 skeleton — see the canonical PT_LEAN package:

→ **https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean**

The standalone file shipped here predates the package refactor. It is kept
for **pedagogical reproducibility**: a single self-contained `.lean` file
that anyone can verify alongside the M1 article without cloning the full
PT_LEAN project.
