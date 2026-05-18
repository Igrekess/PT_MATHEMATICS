# SPEC — Article 2 : Lectures géométrique et spectrale de PT (FR, 20 p)

**Date :** 2026-05-15
**Auteur :** Yan Senez
**Statut :** design initial

---

## 0. Contexte : programme à deux articles

| # | Titre | Pages | Dossier |
|---|---|---:|---|
| 1 | PT : Fondations | ≤ 20 | `PT_FOUNDATIONS/` (terminé) |
| **2** | **PT : Lectures géométrique et spectrale** (ce SPEC) | **≤ 20** | `PT_GEOMETRIC_SPECTRAL/` |

L'article 2 suppose la lecture d'Article 1 mais reste lisible isolément (rappels minimaux en §1).

## 1. Objectif

Produire un article autonome de **20 pages maximum**, en français, présentant trois lectures \emph{au-delà} du noyau arithmétique de PT :

1. **Lecture algébro-géométrique** : la cascade canonique de PT admet une unique courbe algébrique de Kontsevich-Norbury, de genre `μ* − 1 = 14` (PT_GeoFlow).
2. **Lecture différentielle** : la métrique Fisher-Bianchi induite par la cascade satisfait l'équation du soliton de Ricci `Ric + Hess(f) = λ(μ) g` avec `λ(μ) = −1/μ⁴ + O(1/μ⁵)`, coefficient `−1` exact (PT_GeoFlow).
3. **Lecture spectrale** : identité K2 `1/8 = s²/2 = λ_H = Δ_N^Maslov` reliant axiome `s=1/2` à la phase de Maslov des zéros de Riemann ; identité de trace A6 vérifiée à 10⁻²⁶ (PT_New_Math_Consolidation_FR).

Le programme spectral-arithmétique élargi (RH-PT, items 39-69 de PT_New_Math, encore THM-local en Re(s)>1) est mentionné en §5 comme **programme** sans prétendre à théorème inconditionnel.

## 2. Lectorat cible

Pont physicien-mathématicien (math-ph et hep-th). On peut supposer une familiarité avec :
- géométrie différentielle riemannienne (Ricci, métriques) ;
- géométrie algébrique (courbes, genre, classes de moduli) ;
- bases d'analyse spectrale (opérateurs auto-adjoints, fonctions zêta).

Aucune connaissance préalable des théorèmes T0-T6 ou des axiomes de pont BA0-BA5 n'est requise (rappels minimaux en §1).

## 3. Positionnement épistémologique

- **Théorèmes prouvés** (PT_GeoFlow) : courbe de persistance + genre, quasi-soliton de Ricci asymptotique. Affirmatifs.
- **Identités prouvées** (PT_New_Math, K2 et A6) : `1/8 = s²/2 = λ_H = Maslov` (vérifiée structurellement, ouverte cohomologiquement K3 incomplet), identité de trace A6 (THM-local en Re(s) > 1).
- **Programme conjectural** (RH-PT) : la prolongation à la ligne critique reste ouverte. Mentionné comme programme.

## 4. Nomenclature

Suit strictement `NOMENCLATURE_MAP.md` du monographe + nomenclature de PT_GeoFlow et PT_New_Math_Consolidation_FR. Macros LaTeX :

- T-series, BA-series, L0 (cf. Article 1).
- Nouvelle macros (préambule Article 2) : `\Cper` (courbe de persistance), `\genus` (genre), `\gPT` (métrique PT), `\Ric`, `\Hess`, `\Rres` (opérateur résiduel), `\HBK` (opérateur Berry-Keating), `\lambdaH` (couplage Higgs), `\DeltaMaslov` (résidu Maslov).

**Conventions strictes** (inchangées) : `q_+/q_-` (jamais `q_stat/q_therm`), échos `{11,13}`, super-échos `{17,19,23}`, jamais `[DER]` pour les T- et BA-series.

## 5. Structure et budget (20 p)

| Section | Contenu | Pages |
|---|---|---:|
| §1. Introduction et rappels | Pointeur Article 1, motivation des trois lectures, conventions, rappels minimaux | 2 |
| §2. Lecture algébro-géométrique : courbe de persistance | Théorème principal (A∧B∧C), courbe algébrique Kontsevich-Norbury, genre = μ*−1=14 | 5 |
| §3. Lecture différentielle : quasi-soliton de Ricci | Métrique Fisher-Bianchi, équation soliton, λ(μ) = −1/μ⁴ exact | 5 |
| §4. Lecture spectrale : identités K2 et A6 | Triple identification `1/8 = s²/2 = λ_H = Maslov`, identité de trace, RvM density | 4 |
| §5. Programme spectral-arithmétique | RH-PT, opérateur de survie résiduel, conjectures | 2 |
| §6. Conclusion et perspectives | Synthèse, liens vers Article 1, applications aval | 1 |
| Annexes | Glossaire géométrique, table des résultats | 1 |
| **Total** | | **20** |

### 5.1. §1 — Introduction et rappels (2 p)

- 1.1 Pointeur Article 1, motivation : pourquoi des lectures géométrique/spectrale (~0.6 p).
- 1.2 Rappels minimaux : axiome `s = 1/2`, point fixe `μ* = 15`, premiers actifs `{3,5,7}`, échos `{11,13}`, cascade `γ_3 → γ_3γ_5 → γ_3γ_5γ_7`, holonomie `sin²θ_p = δ_p(2−δ_p)` (~1 p).
- 1.3 Plan, statut épistémique différencié (théorèmes / identités / programme) (~0.4 p).

### 5.2. §2 — Courbe de persistance et genre (5 p)

- 2.1 Préliminaires : classe Kontsevich-Norbury, récursion Eynard-Orantin (~1 p).
- 2.2 Théorème principal : trois conditions (A) holonomie polynomiale (T6+BA3), (B) seuil d'activation (BA4), (C) auto-cohérence (T5) caractérisent une **unique** courbe algébrique (~1.5 p).
- 2.3 Forme explicite de la courbe (équation, paramètres) (~1 p).
- 2.4 Calcul du genre : `genus(C_per) = μ* − 1 = 14` (~1 p).
- 2.5 Conséquence : `μ*` comme invariant topologique, non pas numérique (~0.5 p).

### 5.3. §3 — Quasi-soliton de Ricci (5 p)

- 3.1 La métrique Fisher-Bianchi `g_PT` induite par la cascade (~1 p).
- 3.2 Théorème du quasi-soliton : `Ric(g_PT) + Hess(f) = λ(μ) g_PT` (~1 p).
- 3.3 Asymptotique `λ(μ) = −1/μ⁴ + O(1/μ⁵)`, coefficient `−1` exact, indépendant du couple `(p₁, p₂)` (~1.5 p).
- 3.4 Lien avec le programme de Hamilton-Perelman (~1 p).
- 3.5 Interprétation : `λ < 0` signature expansive, le résidu non-nul comme dynamique non-triviale (~0.5 p).

### 5.4. §4 — Identités K2 et A6 (4 p)

- 4.1 Densité de Riemann-von Mangoldt et résidu `1/8` (~1 p).
- 4.2 **Identité K2** : `1/8 = s²/2 = λ_H = Δ_N^Maslov`, triple identification (~1.5 p).
- 4.3 **Identité de trace A6** : `−d/ds log R(s) = Σ_p (log p)[1/(p^s−1) + A_p p^{−s}]`, vérifiée à 10⁻²⁶ en `Re(s) > 1` (~1 p).
- 4.4 Conséquences : lien direct axiome `s=1/2` ↔ phénoménologie spectrale (~0.5 p).

### 5.5. §5 — Programme spectral-arithmétique (2 p)

- 5.1 Opérateur de survie résiduel `λ_p(s) = p^{−s}` (~0.5 p).
- 5.2 Schur-Fredholm-Bergman-Fisher (énoncés courts, statut THM-local) (~0.7 p).
- 5.3 Conjecture RH-PT : prolongement à la ligne critique (~0.8 p).

### 5.6. §6 — Conclusion (1 p)

- 6.1 Synthèse : trois lectures complémentaires d'un même objet arithmétique.
- 6.2 Articulation Article 1 ↔ Article 2.
- 6.3 Applications aval, perspectives (peer review).

### 5.7. Annexes (1 p)

- A. Glossaire géométrique-spectral.
- B. Table récapitulative des résultats (théorèmes / identités / conjectures).

## 6. Stack technique

LaTeX `article`, `latexmk`, `biblatex+biber`, git local — identique à Article 1.

## 7. Livrables

- `main.tex`, `preamble.tex` (avec macros Article 2), `references.bib`, `latexmkrc`
- `sections/` (6 fichiers : `01_intro.tex` à `06_conclusion.tex`)
- `annexes/` (2 fichiers : `A_glossaire.tex`, `B_resultats.tex`)
- `figures/` (cascade, courbe de persistance, métrique Fisher-Bianchi schématique)
- `README.md`
- `main.pdf` (≤ 20 p)

## 8. Hors-scope explicite

- Re-démonstration des théorèmes T0-T6 et BA3-BA5 d'Article 1 (rappels seulement).
- Démonstrations complètes de la courbe de persistance et du soliton Ricci → renvoyer aux papiers PT_GeoFlow.
- Démonstrations complètes du programme RH-PT → renvoyer aux notes 39-69 PT_New_Math.
- Applications aval (chimie, NMR, etc.) — pointeurs en §6 seulement.

## 9. Critères d'acceptation

- Compilation `latexmk` sans erreur, PDF ≤ 20 p.
- Naming strict NOMENCLATURE_MAP.md ; aucune occurrence de `q_stat`, `q_therm`, `ghost`, `[DER]`.
- Trois lectures clairement délimitées (§2, §3, §4) avec leurs statuts épistémiques (théorèmes prouvés vs identités prouvées vs programme).
- §2 énonce le théorème de la courbe (A∧B∧C) avec genre = 14 explicite.
- §3 énonce l'équation du soliton et l'asymptotique λ = −1/μ⁴ avec coefficient −1 exact.
- §4 énonce l'identité K2 `1/8 = s²/2 = λ_H = Δ_N^Maslov` et l'identité de trace A6.
- §5 distingue clairement la conjecture RH-PT du reste.
- Citation explicite de PT_GeoFlow et PT_New_Math_Consolidation_FR pour les sources de chaque résultat.
