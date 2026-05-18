# PT MATHEMATICS (PTM)
## The Theory of Persistence — Mathematical Articles

The mathematical corpus of the Theory of Persistence (PT): a closed chain of
five articles (M1–M5) deriving the framework from a single arithmetic input
($s = 1/2$) to the reconstruction of physics, plus a Reader's Map (M0),
four thematic extensions, and one *hors série* (Ramanujan–Mihailescu).

Each article ships with companion Python scripts that verify the theorems
numerically. M1–M5 are now bilingual (EN + FR).

**Snapshot:** 2026-05-18

## To go further

→ Physics derived from this framework: [*PT-PHYSICS (PTP)*](https://github.com/Igrekess/PT_PHYSICS)
→ Chemistry derived from this framework: [*PT-CHEMISTRY (PTC)*](https://github.com/Igrekess/PT_CHEMISTRY)
→ Color theory derived from this framework: [*Simplex Color Space (SCS)*](https://github.com/Igrekess/SimplexColorSpace)
→ Lean 4 formalisation (kernel-verified proofs): [*PT_LEAN*](https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean)
→ Full theoretical framework: [Senez, Y. (2026). *The Theory of Persistence: A Complete Monograph (2026)*](https://zenodo.org/records/19655984)

---

## Core series (M0–M5)

| # | Title | EN | FR | Chapters |
|---|-------|:--:|:--:|----------|
| **M0** | [Reader's Map](M0/) — *A pedagogical entry point to the program* | ✅ 5 p | ✅ 5 p | preface |
| **M1** | [The Sieve as a Dynamical System](M1/) | ✅ 44 p | ✅ 46 p | ch. 1–4 |
| **M2** | [Information Geometry and Holonomy](M2/) | ✅ 26 p | ✅ 27 p | ch. 5–6 |
| **M3** | [Convergence and the Unique Fixed Point](M3/) | ✅ 43 p | ✅ 44 p | ch. 7–8, 25 |
| **M4** | [Extended Mathematical Structures](M4/) | ✅ 57 p | ✅ 58 p | math, PM |
| **M5** | [The Bridge from Arithmetic to Physics](M5/) | ✅ 57 p | ✅ 59 p | ch. 9–11 |

### Logical chain

```
M0 : preface (reader's map)
 |
M1 : s = 1/2            (T1, T3, T2, L0, GFT)
 |
M2 : sin²θ_p, γ_p       (G1, G2, G3, holonomy identity)
 |
M3 : μ* = 15            (T4 convergence, T5 fixed point, T0 closing BA0)
 |
M4 : structures         (sieve algebra, PM, complex mechanics, Theorem H)
 |
M5 : physics            (BA0–BA5, α_EM = 1/137, Lemmas E/F/G: all bridges closed)
```

### Key results

- **M1** — The forbidden-transition theorem (T1) forces s = 1/2. The transfer matrix T₃ = antidiag(1,1) is unique. The GFT identity log₂(m) = D_KL + H holds exactly.
- **M2** — D_KL is the unique f-divergence (G1, Shore–Johnson). The Fisher metric is the unique monotone Riemannian metric (G3, Čencov). The holonomy angle sin²θ_p = δ_p(2−δ_p) is an algebraic identity.
- **M3** — The spectral convergence α_k → 1/2 is proved via Gordin decomposition (T4). The unique fixed point μ* = 3+5+7 = 15 is proved by exhaustion (T5). The dynamical field theorem T0 promotes BA0 from postulate to theorem.
- **M4** — 32 computational tools verified (659/659 PASS). Predictive Mathematics: each prime adds exactly 1 DOF (L1). Theorem H: decoherence monotonicity = 2nd law of the sieve.
- **M5** — The six bridge axioms BA0–BA5 are all theorems. α_EM = ∏ sin²θ_p = 1/137.036 (0 parameters). Lemmas E, F, G close all remaining bridge claims: coupling (spectral invariant), metric (Hessian of ln Z_Ruelle), Hilbert space (OS reconstruction).

### Lean 4 formalisation

Every article ships a `LEAN.md` index pointing to the canonical kernel-verified
modules in the [**PT_LEAN**](https://github.com/Igrekess/PersistenceTheory/tree/main/pt_lean)
package: [M1/LEAN.md](M1/LEAN.md), [M2/LEAN.md](M2/LEAN.md),
[M3/LEAN.md](M3/LEAN.md), [M4/LEAN.md](M4/LEAN.md), [M5/LEAN.md](M5/LEAN.md).
M1 additionally ships a standalone Lean snapshot of theorem T1
([`M1/T1ForbiddenTransitions.lean`](M1/T1ForbiddenTransitions.lean), depends
only on `Mathlib.Tactic`) as a pedagogical companion.

---

## Extensions (level 2)

Thematic explorations beyond the M1–M5 closed chain.

| Folder | Title | EN | FR |
|--------|-------|:--:|:--:|
| [extensions/geometric_spectral/](extensions/geometric_spectral/) | Geometric and Spectral Readings of PT | ✅ 14 p | ✅ 14 p |
| [extensions/zeta_spectral/](extensions/zeta_spectral/) | Canonical Spectral Model of ζ Zeros | ✅ 23 p | ✅ 23 p |
| [extensions/crt_decoupling/](extensions/crt_decoupling/) | CRT Decoupling | — | ✅ |
| [extensions/holonomy_nonlocality/](extensions/holonomy_nonlocality/) | Holonomy and Nonlocality | ✅ | ✅ |

## Hors série

| Folder | Title | EN | FR |
|--------|-------|:--:|:--:|
| [hors_serie/ramanujan_mihailescu/](hors_serie/ramanujan_mihailescu/) | Catalan and Ramanujan through PT | ✅ 4 p | ✅ 4 p |

> **Note.** PT_COLOR (color perception via PT) is not part of this repository;
> it lives in [SimplexColorSpace](https://github.com/Igrekess/SimplexColorSpace)
> where the engineering implementation is also distributed.

---

## Epistemic status tags

All articles use the canonical PT corpus grid:

| Tag | Meaning |
|---|---|
| `[Thm]` | Unconditional theorem, proved |
| `[Id]` | Exact algebraic identity |
| `[Der]` | Derived result (with standard hypotheses) |
| `[Bridge]` | Bridge identification (arithmetic ↔ physics) |
| `[Cond]` | Conditional on an explicit hypothesis |
| `[Val]` | Numerical validation |
| `[Pred]` | Testable PT prediction |
| `[Open]` | Open question |
| `[Falsified]` | Tested prediction rejected (self-correction) |

The 10 PT commitments C1–C10 are currently at **10/10 [Thm]** (since the
C2/C4/C5/C7 promotions of 2026-04-26 — see the monograph corpus).

---

## Running the scripts

Each article has a `scripts/` directory with a `lib/` subfolder containing
shared utilities. To run:

```bash
cd M1/scripts
PYTHONPATH=lib python3 ch01_sieve/test_foundations_PT.py
```

Requirements: Python 3.10+, numpy, scipy (some scripts), fractions (stdlib).

## Building the PDFs

Each M-article carries its own copies of `pt-common.sty` and `pt-common-fr.sty`:

```bash
cd M1
pdflatex m1_persistence.tex && bibtex m1_persistence \
  && pdflatex m1_persistence.tex && pdflatex m1_persistence.tex
# French version:
pdflatex m1_persistence_fr.tex && bibtex m1_persistence_fr \
  && pdflatex m1_persistence_fr.tex && pdflatex m1_persistence_fr.tex
```

The M0 and extensions folders use `latexmkrc`:

```bash
cd M0
latexmk -pdf main.tex      # FR
latexmk -pdf main_en.tex   # EN
```

---

## FR — Articles Mathématiques

Le corpus mathématique de la Théorie de la Persistance : une chaîne fermée
de cinq articles (M1–M5) du paramètre arithmétique $s = 1/2$ jusqu'à la
reconstruction de la physique, accompagnée d'une carte de lecture (M0),
quatre extensions thématiques et un hors série (Ramanujan–Mihailescu).

**Chaîne logique** :

```
M0 : préface (carte de lecture)
 |
M1 : s = 1/2              (T1, T3, T2, L0, GFT)
 |
M2 : sin²θ_p, γ_p         (G1, G2, G3, identité d'holonomie)
 |
M3 : μ* = 15              (T4, T5, T0)
 |
M4 : structures           (algèbre du crible, PM, mécanique complexe, Théorème H)
 |
M5 : physique             (BA0–BA5, α_EM = 1/137, Lemmes E/F/G)
```

Volumes EN ~250 p, FR ~260 p (M1–M5 bilingues) + extensions ~75 p +
hors série ~8 p.
