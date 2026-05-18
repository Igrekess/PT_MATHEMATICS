# Lean 4 formalisation — ζ spectral model

Canonical formalisation in PT_LEAN:

→ https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean

## Modules

| Claim | Module |
|---|---|
| **W7-1** spiral identity (forward, kernel-verified, 0 sorry) | [`PT/Analysis/W7SpiralIdentity`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Analysis/W7SpiralIdentity.lean) |
| **W7-1** spiral identity (reverse direction, partial) | [`PT/Analysis/W7SpiralIdentityReverse`](https://github.com/Igrekess/PersistenceTheory/blob/main/pt_lean/PT/Analysis/W7SpiralIdentityReverse.lean) |

The W7-1 spiral identity ($T_G^{\rm turn}_k(\sigma) = T_G^{\rm turn}_0(\sigma) \iff \sigma^2 = \pi(k+1)$)
is the formal kernel of the §6 acquired result in this article.
Numerical triple validation: k=1,2,3 confirmed at the 1% level via streamed
prime sieving.
