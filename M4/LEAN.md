# Lean 4 formalisation — M4

The theorems of this article are formalised in the canonical **PT_LEAN**
package (kernel-verified, depends on `Mathlib`):

→ https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean

## Modules formalising M4

| Theorem / claim | Module |
|---|---|
| **PM algebra** (Predictive Mathematics) | [`PT/Algebra/PMAlgebra`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Algebra/PMAlgebra.lean) |
| **EML** algebra (Extended Multiplicative Logic) | [`PT/EML/EMLAlgebra`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/EMLAlgebra.lean), [`EMLIdentities`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/EMLIdentities.lean), [`EMLDepth3`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/EMLDepth3.lean), [`EMLSheffer3Args`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/EMLSheffer3Args.lean) |
| **q-parameter** Sheffer structure | [`QSheffer`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/QSheffer.lean), [`QParameterMonotonicity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/QParameterMonotonicity.lean), [`QPlusQMinusComparison`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/EML/QPlusQMinusComparison.lean) |
| **Sieve algebra** N1–N3 | [`Sieve/N1AtomicUniqueness`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N1AtomicUniqueness.lean), [`N3aFactorisation`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N3aFactorisation.lean), [`N3aMinimalMonoid`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N3aMinimalMonoid.lean), [`N3bSieveTaxonomy`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N3bSieveTaxonomy.lean), [`N3cCanonicalOrdering`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N3cCanonicalOrdering.lean) |
| **L1** prime ↔ DOF cascade | [`Sieve/N4PrimeCascade`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N4PrimeCascade.lean), [`N4DimensionCascade`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/N4DimensionCascade.lean) |
| **Theorem H** decoherence monotonicity | [`Information/EntropyMonotonicity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/EntropyMonotonicity.lean), [`BinaryEntropyMonotonicity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/BinaryEntropyMonotonicity.lean), [`ShannonEntropyConcavity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/ShannonEntropyConcavity.lean) |
| Bekenstein bound (information limit) | [`Information/BekensteinBound`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/BekensteinBound.lean), [`BekensteinEquality`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/BekensteinEquality.lean), [`BekensteinExtensions`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/BekensteinExtensions.lean) |
| Prime gap distribution (moments to 6th) | [`Conservation/GapDistribution*`](https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean/PT/Conservation), [`PrimeGapMoments`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Conservation/PrimeGapMoments.lean), [`PtPrimeStructuralTheorem`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Conservation/PtPrimeStructuralTheorem.lean) |

## Run locally

```sh
git clone https://github.com/Igrekess/PersistenceTheory
cd PersistenceTheory/pt_lean
lake exe cache get && lake build
```
