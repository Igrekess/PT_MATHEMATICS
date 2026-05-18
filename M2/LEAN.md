# Lean 4 formalisation — M2

The theorems of this article are formalised in the canonical **PT_LEAN**
package (kernel-verified, depends on `Mathlib`):

→ https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean

## Modules formalising M2

| Theorem / claim | Module |
|---|---|
| **G1** D_KL unique f-divergence (Shore–Johnson, autonomous form) | [`PT/Information/T6cG1Autonomous`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/T6cG1Autonomous.lean) |
| **G2** Fisher metric emergence | [`PT/Information/G2FisherEmergence`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/G2FisherEmergence.lean) |
| **G3** Čencov uniqueness of Fisher | [`PT/Information/G3FisherUniqueness`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/G3FisherUniqueness.lean), [`T6cChencov`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/T6cChencov.lean) |
| KL additivity (product, mutual info) | [`KLAdditivityProduct`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/KLAdditivityProduct.lean), [`KLAdditivityFromMI`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/KLAdditivityFromMI.lean), [`RelativeEntropyAdditivity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/RelativeEntropyAdditivity.lean) |
| Information-theoretic master framework | [`EntropyMasterFramework`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/EntropyMasterFramework.lean), [`InfoTheoreticMaster`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Information/InfoTheoreticMaster.lean) |
| **Holonomy identity** sin²θ_p = δ_p(2−δ_p) | [`PT/Holonomy/SinSqProductChain`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/SinSqProductChain.lean), [`SinSqProductBounds`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/SinSqProductBounds.lean), [`SinSqRatios`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/SinSqRatios.lean), [`InverseSinSq`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/InverseSinSq.lean), [`InverseSinSqProduct`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/InverseSinSqProduct.lean) |
| γ_p definition, monotonicity, product, sum | [`GammaProduct`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/GammaProduct.lean), [`GammaSumProduct`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/GammaSumProduct.lean), [`GammaMonotonicity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/GammaMonotonicity.lean), [`GammaRatio`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/GammaRatio.lean), [`GammaPrimorialProduct`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/GammaPrimorialProduct.lean), [`GammaTablesExtended`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/GammaTablesExtended.lean) |
| α ↔ γ relation, primorial cascade | [`AlphaGammaRelation`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/AlphaGammaRelation.lean), [`AlphaTimesGammaSum`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/AlphaTimesGammaSum.lean), [`AlphaPowerSequence`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/AlphaPowerSequence.lean) |
| Holonomy master framework | [`HolonomyMasterFramework`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Holonomy/HolonomyMasterFramework.lean) |

## Run locally

```sh
git clone https://github.com/Igrekess/PersistenceTheory
cd PersistenceTheory/pt_lean
lake exe cache get && lake build
```
