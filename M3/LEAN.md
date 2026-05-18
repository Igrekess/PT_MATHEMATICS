# Lean 4 formalisation — M3

The theorems of this article are formalised in the canonical **PT_LEAN**
package (kernel-verified, depends on `Mathlib`):

→ https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean

## Modules formalising M3

| Theorem / claim | Module |
|---|---|
| **T4** Mertens convergence on active primes | [`PT/NumberTheory/T4MertensActivePrimes`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/NumberTheory/T4MertensActivePrimes.lean) |
| **T5** Mertens classical | [`PT/NumberTheory/T5Mertens`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/NumberTheory/T5Mertens.lean) |
| Spectral convergence α_k → 1/2 (Gordin decomposition) | [`Stochastic/T2T3SpectralMixing`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T2T3SpectralMixing.lean), [`T2T3KroneckerEigenvalues`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T2T3KroneckerEigenvalues.lean), [`T2T3T5KroneckerSpectrum`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T2T3T5KroneckerSpectrum.lean) |
| Stationary uniqueness, Perron eigenvector | [`T2T3StationaryUniqueness`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T2T3StationaryUniqueness.lean), [`T2T3PerronEigenvectorUniqueness`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T2T3PerronEigenvectorUniqueness.lean) |
| Cesàro limit, T3 spectral decomposition | [`T3CesaroLimit`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T3CesaroLimit.lean), [`T2T3CesaroLimit`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T2T3CesaroLimit.lean), [`T3SpectralDecomposition`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T3SpectralDecomposition.lean) |
| **T5** canonical (tight bounds) | [`T5Canonical`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T5Canonical.lean), [`T5CanonicalTight`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/T5CanonicalTight.lean) |
| **μ* = 15** unique fixed point | [`Sieve/N2UniqueFixedPoint`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N2UniqueFixedPoint.lean), [`Sieve/N2SelfCoherence`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N2SelfCoherence.lean) |
| **T7** Global uniqueness μ* | [`FixedPoint/T7MuStar`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/FixedPoint/T7MuStar.lean), [`T7GlobalUniqueness`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/FixedPoint/T7GlobalUniqueness.lean) |
| Fixed-point master | [`FixedPointMasterTheorem`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/FixedPoint/FixedPointMasterTheorem.lean), [`CrystallisationBinary`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/FixedPoint/CrystallisationBinary.lean), [`DimensionProtection`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/FixedPoint/DimensionProtection.lean) |
| **T0** Dynamical field (BA0 closure) | [`Bridge/BridgeAxioms`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Bridge/BridgeAxioms.lean) |
| Active-prime cascade chain | [`Sieve/N4PrimeCascade`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N4PrimeCascade.lean), [`N4DimensionCascade`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N4DimensionCascade.lean), [`PrimitiveRootCascade`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/PrimitiveRootCascade.lean) |

## Run locally

```sh
git clone https://github.com/Igrekess/PersistenceTheory
cd PersistenceTheory/pt_lean
lake exe cache get && lake build
```
