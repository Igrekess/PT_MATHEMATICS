# Lean 4 formalisation — M1

The theorems of this article are formalised in the canonical **PT_LEAN**
package (kernel-verified, depends on `Mathlib`):

→ https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean

## Modules formalising M1

| Theorem / claim | Module |
|---|---|
| **T1** Forbidden modular transitions (6-rough, mod-3 switch) | [`PT/Sieve/T1ForbiddenTransitions`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/T1ForbiddenTransitions.lean) |
| **T1** Antidiagonal orbits | [`PT/Sieve/T1AntidiagOrbits`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/T1AntidiagOrbits.lean) |
| **T1** Orbits mod 5, mod 7 | [`PT/Sieve/T1OrbitsZMod5`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/T1OrbitsZMod5.lean), [`PT/Sieve/T1OrbitsZMod7`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/T1OrbitsZMod7.lean) |
| 6-rough basics | [`PT/Sieve/SixRough`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/SixRough.lean) |
| **T3** Antidiagonal transfer matrix | [`PT/Sieve/T3Antidiagonal`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Sieve/T3Antidiagonal.lean) |
| **s = 1/2** spectral gap | [`PT/Stochastic/SHalf`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Stochastic/SHalf.lean) |
| **T2** conservation / α | [`PT/Conservation/T2Alpha`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Conservation/T2Alpha.lean), [`T2SpectralBridge`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Conservation/T2SpectralBridge.lean) |
| **L0** maximum entropy on `Z/30Z` | [`PT/Information/L0MaxEntropy`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/L0MaxEntropy.lean), [`L0Uniqueness`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/L0Uniqueness.lean) |
| **GFT** identity log₂(m) = D_KL + H | [`PT/Information/GFTIdentity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/GFTIdentity.lean), [`GFTOnZMod30`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/GFTOnZMod30.lean) |
| Bimodality / sieve algebra (M1 §4) | [`PT/Sieve/Bimodality*`](https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean/PT/Sieve), [`AdmissibleResidues*`](https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean/PT/Sieve) |

## Standalone snapshot

[`T1ForbiddenTransitions.lean`](T1ForbiddenTransitions.lean) shipped here is a
**self-contained pedagogical companion** depending only on `Mathlib.Tactic`. It
predates the PT_LEAN package refactor and proves T1 alone, as a single file:

```sh
lake init T1Lean math
cp T1ForbiddenTransitions.lean T1Lean/T1Lean/
cd T1Lean && lake exe cache get && lake build
```

For a full audit of M1, use the canonical PT_LEAN package above.
