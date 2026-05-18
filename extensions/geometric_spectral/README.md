# PT — Article 2 : Lectures géométrique et spectrale (FR)

Article LaTeX en français présentant **trois lectures avancées** de la Théorie de la Persistance :
1. Lecture algébro-géométrique : courbe de persistance, genre = μ\* − 1 = 14 (PT_GeoFlow).
2. Lecture différentielle : quasi-soliton de Ricci de la métrique Fisher-Bianchi (PT_GeoFlow).
3. Lecture spectrale : identités K2 et A6 (PT_New_Math_Consolidation_FR).

**Cible :** 20 pages maximum.
**Statut :** en rédaction.

## Programme à deux articles cités-couplés

| # | Titre | Pages | Dossier |
|---|---|---:|---|
| 1 | PT : Fondations | ≤ 20 | `PT_FOUNDATIONS/` (terminé) |
| **2** | **PT : Lectures géométrique et spectrale** (ce dépôt) | ≤ 20 | `PT_GEOMETRIC_SPECTRAL/` |

## Documents

- [SPEC.md](SPEC.md) — design validé
- [main.tex](main.tex) — orchestrateur LaTeX

## Build

```bash
latexmk -pdf main.tex
```

Sortie : `main.pdf` (cible ≤ 20 p).

## Auteur

Yan Senez (PT corpus), 2026.
