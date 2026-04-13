# PT MATHEMATICS (PTM)
## The Theory of Persistence — Mathematical Articles (M1–M5)


Five self-contained articles presenting the mathematical foundations of the Theory of Persistence, from the sieve of Eratosthenes to the reconstruction of physics.

Each article includes companion scripts that verify every theorem numerically.


## To go further

-> To explore the physics derived from this framework: [*PT-PHYSICS (PTP)*](https://github.com/Igrekess/PT_PHYSICS)  
-> To explore the chemistry derived from this framework: [*PT-CHEMISTRY (PTC)*](https://github.com/Igrekess/PT_CHEMISTRY)  
-> To explore a color theory derived from Persistence Theory: [*Simplex Color Space (SCS)*](https://github.com/Igrekess/SimplexColorSpace)  
-> For the full theoretical framework: [Senez, Y. (2026). *The Theory of Persistence: A Complete Monograph (2026)*](https://zenodo.org/records/19520809)


---

## Articles

| # | Title | Pages | Scripts | Chapters |
|---|-------|-------|---------|----------|
| **M1** | [The Sieve as a Dynamical System](M1/) | 44 | 21 | ch. 1–4 |
| **M2** | [Information Geometry and Holonomy](M2/) | 26 | 7 | ch. 5–6 |
| **M3** | [Convergence and the Unique Fixed Point](M3/) | 43 | 39 | ch. 7–8, 25 |
| **M4** | [Extended Mathematical Structures](M4/) | 57 | 62 | math, PM |
| **M5** | [The Bridge from Arithmetic to Physics](M5/) | 55 | 25 | ch. 9–11 |
| | **Total** | **225** | **154** | |

## Logical chain

```
M1: s = 1/2         (T1, T3, T2, L0, GFT)
 |
M2: sin²θ_p, γ_p   (G1, G2, G3, holonomy identity)
 |
M3: μ* = 15         (T4 convergence, T5 fixed point, T0 BA0 closing)
 |
M4: structures      (sieve algebra, PM, complex mechanics, Theorem H)
 |
M5: physics         (BA0–BA5, α_EM = 1/137, Lemmas E/F/G: all bridges closed)
```

## Key results

- **M1** — The forbidden-transition theorem (T1) forces s = 1/2. The transfer matrix T₃ = antidiag(1,1) is unique. The GFT identity log₂(m) = D_KL + H holds exactly.
- **M2** — D_KL is the unique f-divergence (G1, Shore–Johnson). The Fisher metric is the unique monotone Riemannian metric (G3, Čencov). The holonomy angle sin²θ_p = δ_p(2−δ_p) is an algebraic identity.
- **M3** — The spectral convergence α_k → 1/2 is proved via Gordin decomposition (T4). The unique fixed point μ* = 3+5+7 = 15 is proved by exhaustion (T5). The dynamical field theorem T0 promotes BA0 from postulate to theorem.
- **M4** — 32 computational tools verified (659/659 PASS). Predictive Mathematics: each prime adds exactly 1 DOF (L1). Theorem H: decoherence monotonicity = 2nd law of the sieve.
- **M5** — The six bridge axioms BA0–BA5 are all theorems. α_EM = ∏ sin²θ_p = 1/137.036 (0 parameters). Lemmas E, F, G close all remaining bridge claims: coupling (spectral invariant), metric (Hessian of ln Z_Ruelle), Hilbert space (OS reconstruction).

## Running the scripts

Each article has a `scripts/` directory with a `lib/` subfolder containing shared utilities. To run:

```bash
cd M1/scripts
PYTHONPATH=lib python3 ch01_sieve/test_foundations_PT.py
```

Requirements: Python 3.10+, numpy, scipy (some scripts), fractions (stdlib).

## Building the PDFs

Each article uses `pt-common.sty` from the shared style directory:

```bash
cd M1
ln -sf ../../shared/pt-common.sty .
pdflatex m1_persistence.tex && bibtex m1_persistence && pdflatex m1_persistence.tex && pdflatex m1_persistence.tex
```

---
# PT MATHEMATICS (PTM)
## La Théorie de la Persistance — Articles Mathématiques (M1–M5)

Cinq articles autonomes présentant les fondements mathématiques de la Théorie de la Persistance, du crible d'Ératosthène à la reconstruction de la physique.

Chaque article inclut des scripts compagnons qui vérifient numériquement chaque théorème.

**Monographie :** Senez, Y. (2026). *The Theory of Persistence: A Complete Monograph.* [doi:10.5281/zenodo.18726591](https://zenodo.org/records/19520809)

---

## Articles

| # | Titre | Pages | Scripts | Chapitres |
|---|-------|-------|---------|-----------|
| **M1** | [Le crible comme système dynamique](M1/) | 44 | 21 | ch. 1–4 |
| **M2** | [Géométrie de l'information et holonomie](M2/) | 26 | 7 | ch. 5–6 |
| **M3** | [Convergence et point fixe unique](M3/) | 43 | 39 | ch. 7–8, 25 |
| **M4** | [Structures mathématiques étendues](M4/) | 57 | 62 | math, PM |
| **M5** | [Le pont de l'arithmétique à la physique](M5/) | 55 | 25 | ch. 9–11 |
| | **Total** | **225** | **154** | |

## Chaîne logique

```
M1 : s = 1/2          (T1, T3, T2, L0, GFT)
 |
M2 : sin²θ_p, γ_p    (G1, G2, G3, identité d'holonomie)
 |
M3 : μ* = 15          (T4 convergence, T5 point fixe, T0 fermeture BA0)
 |
M4 : structures        (algèbre du crible, PM, mécanique complexe, Théorème H)
 |
M5 : physique          (BA0–BA5, α_EM = 1/137, Lemmes E/F/G : tous les ponts fermés)
```

## Résultats clés

- **M1** — Le théorème des transitions interdites (T1) force s = 1/2. La matrice de transfert T₃ = antidiag(1,1) est unique. L'identité GFT log₂(m) = D_KL + H est exacte.
- **M2** — D_KL est l'unique f-divergence (G1, Shore–Johnson). La métrique de Fisher est l'unique métrique riemannienne monotone (G3, Čencov). L'angle d'holonomie sin²θ_p = δ_p(2−δ_p) est une identité algébrique.
- **M3** — La convergence spectrale α_k → 1/2 est prouvée via la décomposition de Gordin (T4). Le point fixe unique μ* = 3+5+7 = 15 est prouvé par exhaustion (T5). Le théorème T0 promeut BA0 de postulat à théorème.
- **M4** — 32 outils computationnels vérifiés (659/659 PASS). Mathématiques Prédictives : chaque premier ajoute exactement 1 degré de liberté (L1). Théorème H : monotonie de la décohérence = 2ème loi du crible.
- **M5** — Les six axiomes de pont BA0–BA5 sont tous des théorèmes. α_EM = ∏ sin²θ_p = 1/137.036 (0 paramètre). Les Lemmes E, F, G ferment tous les ponts restants : couplage (invariant spectral), métrique (hessien de ln Z_Ruelle), espace de Hilbert (reconstruction OS).

## Exécuter les scripts

```bash
cd M1/scripts
PYTHONPATH=lib python3 ch01_sieve/test_foundations_PT.py
```

Prérequis : Python 3.10+, numpy, scipy (certains scripts), fractions (stdlib).

## Compiler les PDFs

```bash
cd M1
ln -sf ../../shared/pt-common.sty .
pdflatex m1_persistence.tex && bibtex m1_persistence && pdflatex m1_persistence.tex && pdflatex m1_persistence.tex
```
