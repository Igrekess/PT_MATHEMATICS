# SPEC — PT_FOUNDATIONS : Préface du programme PT

**Date :** 2026-05-15 (révisé après pivot stratégique 2-niveaux)
**Auteur :** Yan Senez
**Statut :** v1.0-draft (préface autoportante, 5 p)

---

## 0. Stratégie d'ensemble : programme à deux niveaux

Le programme PT est désormais organisé en **deux niveaux** d'articles, plus une préface.

### Préface
| Dossier | Pages | Rôle |
|---|---:|---|
| **PT_FOUNDATIONS** (ce SPEC) | ~5 | **Carte du programme** : énoncés des théorèmes et axiomes, sans preuves. Renvoie aux articles compagnons pour les démonstrations. |

### Niveau 1 — Corpus mathématique de référence
| Dossier | Pages | Contenu |
|---|---:|---|
| **PT_MATHEMATICS/M1** | 44 | Crible dynamique : T1 (transitions interdites), T3, T6, L0, GFT |
| **PT_MATHEMATICS/M2** | 26 | Géométrie de l'information : Fisher, holonomie BA3, premiers actifs |
| **PT_MATHEMATICS/M3** | 43 | Convergence T4, point fixe T5, fermeture T0 (BA0 promu) |
| **PT_MATHEMATICS/M4** | 57 | Algèbre du crible, mécanique complexe, mathématiques prédictives |
| **PT_MATHEMATICS/M5** | 55 | Pont arithmétique-physique : BA0-BA5, lemmes E/F/G, α_EM |

Total Niveau 1 : ~225 pages.

### Niveau 2 — Extensions thématiques
| Dossier | Pages | Contenu |
|---|---:|---|
| **PT_GEOMETRIC_SPECTRAL** | 14 | Courbe de persistance (genre 14), quasi-soliton Ricci, K2, A6, conjecture RH-PT |
| **PT_RH** (à créer) | ~20 | Programme spectral-arithmétique détaillé (depuis PT_New_Math_Consolidation_FR) |
| **PT_RAMANUJAN_MIHAILESCU** | 5 | Application orthogonale (Catalan via Mihailescu) — publication séparée |

## 1. Objectif de la préface

Produire une carte du programme PT en environ **5 pages**, en français, qui :
1. énonce de façon précise les sept théorèmes structurants `T0-T6` et les six axiomes de pont prouvés `BA0-BA5` ;
2. introduit le point fixe `μ* = 15`, la taxonomie des nombres premiers (actifs `{3,5,7}`, échos `{11,13}`, super-échos `{17,19,23}`, inactifs) ;
3. expose le principe de cascade arithmétique `γ_3 → γ_3γ_5 → γ_3γ_5γ_7` ;
4. déroule la chaîne `s = 1/2 → T1-T6 → μ* → BA3 → BA5 → α_bare → α_EM ≈ 1/137.036` à zéro paramètre ;
5. donne le plan de lecture des articles compagnons (M1-M5 + extensions Niveau 2).

**Aucune preuve** n'est donnée ; chaque énoncé renvoie au M-article ou extension qui le démontre.

## 2. Lectorat cible

Lecteur scientifique pressé voulant comprendre la PT en 1 heure. Pont physicien-mathématicien. Le détail technique est dans M1-M5.

## 3. Positionnement épistémique

**Affirmatif strict.** Les sept T0-T6 et les six BA0-BA5 sont tous prouvés (compteur C1-C12 = 5 [THM] + 6 [DER]/[COND] + 1 [VAL], requalification B4 du 2026-08-14). La préface s'autorise à énoncer sans preuve uniquement parce que chaque énoncé est rigoureusement démontré dans un M-article compagnon.

## 4. Nomenclature (canonique, alignée NOMENCLATURE_MAP.md)

Voir SPEC précédent (T-series, BA-series, L0, BT-series, q+/q-, échos vs super-échos). Tous les conventions strictes maintenues.

## 5. Structure (5 pages réelles)

- §1 Motivation : pourquoi un crible ? (~0.8 p)
- §2 Le crible PT et son point fixe (cadre `L^0`, T0-T6 tabulés, `μ*=15`) (~1.5 p)
- §3 Axiomes de pont et taxonomie des premiers (BA0-BA5 tabulés, 5 classes) (~1.5 p)
- §4 Principe de cascade arithmétique (~0.5 p)
- §5 Application canonique `α_EM` (chaîne complète + valeur numérique) (~0.5 p)
- §6 Plan de lecture (Niveau 1 / Niveau 2) (~0.2 p)

Tout est dans `main.tex` (236 lignes), pas de `\input{sections/...}` (les anciens fichiers sections sont conservés en archive mais ne sont plus inclus).

## 6. Stack technique

LaTeX `article`, `latexmk`, `biblatex+biber`, git local. Identique aux M-articles.

## 7. Livrables

- `main.tex` (carte du programme, 236 lignes)
- `preamble.tex` (macros canoniques)
- `references.bib` (cite M1-M5 + PT_GeoSpectral + PT_RH + Wilson/PDG/IwaniecKowalski/etc.)
- `main.pdf` (~5 p)
- `README.md` mis à jour

## 8. Hors-scope explicite

- Preuves complètes des T0-T6 ou des BA0-BA5 → M1-M5.
- Détail de la cascade et des dimensions anomales → M2, M4.
- Lectures géométrique et spectrale → PT_GEOMETRIC_SPECTRAL.
- Programme RH-PT et identités K2, A6 → PT_RH.
- Applications aval (chimie, NMR, allométrie, cosmologie) → sous-projets PT dédiés.

## 9. Critères d'acceptation

- Compilation `latexmk` sans erreur, PDF ≤ 6 p.
- Naming strict : aucun `q_stat`, `q_therm`, `ghost`, `[DER]`.
- Chaque T_i, BA_i a son article compagnon explicitement cité.
- Statut épistémique affirmatif (T0-T6 et BA0-BA5 = [THM]) maintenu.
- Plan de lecture explicite distingue Niveau 1 / Niveau 2.

## 10. Notes archives

Les fichiers `sections/*.tex` et `annexes/*.tex` rédigés dans une version antérieure (Article 1 = 20 p autoportant) sont conservés en archive dans le dossier mais ne sont plus inclus par `main.tex`. Si le besoin se présente de récupérer du matériel rédigé (par ex. énoncés détaillés des T_i ou explications de la cascade), il est récupérable depuis ces fichiers.
