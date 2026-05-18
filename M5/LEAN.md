# Lean 4 formalisation — M5

The theorems of this article are formalised in the canonical **PT_LEAN**
package (kernel-verified, depends on `Mathlib`):

→ https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean

## Modules formalising M5

| Theorem / claim | Module |
|---|---|
| **BA0–BA5** Bridge axioms (all promoted to theorems) | [`PT/Bridge/BridgeAxioms`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Bridge/BridgeAxioms.lean) |
| Math ↔ physics dissolution | [`Bridge/MathPhysicsDissolution`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Bridge/MathPhysicsDissolution.lean) |
| PT cascade derivation chain | [`Bridge/PTCascadeDerivationChain`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Bridge/PTCascadeDerivationChain.lean) |
| Status graph formalisation (C1–C10) | [`Bridge/StatusGraphFormalisation`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Bridge/StatusGraphFormalisation.lean) |
| **BA5** Bargmann anomaly | [`Anomaly/BargmannBA5`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Anomaly/BargmannBA5.lean) |
| **α_EM = ∏ sin²θ_p** inverse cascade | [`Holonomy/AlphaInverseCascadeIdentity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/AlphaInverseCascadeIdentity.lean), [`AlphaInversePowerSequence`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/AlphaInversePowerSequence.lean), [`AlphaInverseTimesGammaSum`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/AlphaInverseTimesGammaSum.lean), [`InvAlphaSquaredBracket`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/InvAlphaSquaredBracket.lean) |
| **Lemma E** Coupling = spectral invariant | [`Holonomy/CouplingReconstruction`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CouplingReconstruction.lean), [`CouplingReconstructionBounds`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CouplingReconstructionBounds.lean) |
| Active-prime criterion / monotonicity (Lemma F input) | [`Holonomy/ActivePrimeCriterion`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/ActivePrimeCriterion.lean), [`ActivePrimeMonotonicity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/ActivePrimeMonotonicity.lean), [`ActivePrimeAnalyticMonotonicity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/ActivePrimeAnalyticMonotonicity.lean), [`ActivePrimeMargins`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/ActivePrimeMargins.lean) |
| **Lemma G** Hilbert space (OS reconstruction input) | [`Conservation/ConservationActivePrimes`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Conservation/ConservationActivePrimes.lean), [`ConservationID*`](https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean/PT/Conservation) |
| **PT Grand Unified Theorem** (top-level closure) | [`PTGrandUnifiedTheorem`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/PTGrandUnifiedTheorem.lean) |
| Cyclic phase (holonomy → coupling) | [`Holonomy/CyclicPhaseIdentity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CyclicPhaseIdentity.lean), [`CyclicPhaseAlgebraic`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CyclicPhaseAlgebraic.lean), [`CyclicPhaseSpectral`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CyclicPhaseSpectral.lean), [`CyclicPhaseInversion`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CyclicPhaseInversion.lean), [`CyclicPhaseTable`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/CyclicPhaseTable.lean) |

## Run locally

```sh
git clone https://github.com/Igrekess/PersistenceTheory
cd PersistenceTheory/pt_lean
lake exe cache get && lake build
```
