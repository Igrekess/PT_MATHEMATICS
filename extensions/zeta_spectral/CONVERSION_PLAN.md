# PT_ZETA — Plan de conversion markdown → LaTeX bilingue

> **Source canonique** : `/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_New_Math_Consolidation_FR/article_PT_zeta_FR.md` (1298 lignes, 2026-05-15).

## État au 2026-05-15 19:20 (post-audit naming)

| § | Statut FR | Statut EN | Fichier FR |
|---|---|---|---|
| Résumé | ✅ converti, purgé | ✅ traduit | `sections/00_resume.tex` |
| §1 Introduction | ✅ converti, purgé | ✅ traduit | `sections/01_introduction.tex` |
| §2 Cadre PT | ✅ converti, purgé | ✅ traduit | `sections/02_cadre_pt.tex` |
| §3 Modèle spectral | ✅ converti | ✅ traduit | `sections/03_modele_spectral.tex` |
| §4 Localisation Re(s)=1/2 | ✅ converti | ✅ traduit | `sections/04_localisation_critique.tex` |
| §5 Sur-détermination 1/8 | ✅ converti | ✅ traduit | `sections/05_surdetermination_un_huitieme.tex` |
| §6 Dualité Higgs ↔ ζ | ✅ converti | ✅ traduit | `sections/06_dualite_higgs_zeta.tex` |
| §7 Vérifications num. | ✅ converti | ✅ traduit | `sections/07_verifications_numeriques.tex` |
| §8 Auto-correction Dirichlet | ✅ converti | ✅ traduit | `sections/08_auto_correction_dirichlet.tex` |
| §9 Limites et ouvertures | ✅ converti | ✅ traduit | `sections/09_limites_ouvertures.tex` |
| §10 Conclusion | ✅ converti | ✅ traduit | `sections/10_conclusion.tex` |

**Build actuel** : FR 23 p, EN 23 p, 0 erreur LaTeX dans les deux versions.

**Audit naming (2026-05-15 post-migration)** :
- ✅ Aucune occurrence de `qstat`, `qtherm`, `ghost`
- ✅ Aucune occurrence de `[DER]` (3 → `[Thm partiel]`, `[Id]`, `[Id]`)
- ✅ Aucune citation `\cite{PT_Monograph}` (36 → `PT_M1`, `PT_M3`, `PT_M5`, `PT_Foundations` selon le chapitre)
- ✅ Aucune mention « note X » dans le corps
- ✅ Macros canoniques `\sper`, `\mustar`, `\qplus`, `\qminus`, `\sinp`, `\Tone`-`\Tsix`, `\BAzero`-`\BAfive` utilisées partout

## Politique de conversion

1. **Math** : `$...$` (inline), `$$...$$` → `\[...\]` ou `\begin{equation}...\end{equation}` avec label.
2. **Macros PT** :
   - `s` → `\sper`, `mu^*` → `\mustar`
   - `q_+`/`q_-` → `\qplus`/`\qminus`
   - `sin^2(theta_p)` → `\sinp{p}`
   - `H_BK` → `\HBK`, `H_PT-BK` → `\HPTBK`
   - `R(s)` → `\Rres(s)`, `D_R` → `\DRop`
   - `A_p` → `\Ap`, `kappa_p` → `\kappap`
   - `T_n` → `\Tone..\Tsix`, `BA_n` → `\BAzero..\BAfive`, `L0` → `\Lzero`
3. **Théorèmes** : `\begin{theoreme}` / `\begin{proposition}` / `\begin{lemme}` / `\begin{definition}` / `\begin{remarque}` / `\begin{conjecture}`.
4. **Citations externes** : `\citep{...}`, `\cite{...}` selon ref.bib.

## 🚨 CONTRAINTE : purge références internes

Toutes les mentions « (note X) », « notes X-Y », « démontré dans la note X »
doivent disparaître. Les démonstrations / résultats contenus dans ces notes
doivent être **inlinés** dans l'article (énoncés + esquisses 5-15 lignes).

Audit : `grep -rE "qstat|qtherm|ghost|\[DER\]|note [0-9]|cite\{PT_Monograph|PT_NewMath_Consolidation" sections/ annexes/ *.tex` → doit être vide.

## Inventaire par section : démonstrations à absorber

### §3 Modèle spectral
- **Notes référencées** : 42, 51, 52, 62, 64, 65
- **À inliner** :
  - **Note 42 (288 l)** — *Trace matching Fredholm résiduel* : dérivation `det(I − D_R(s))⁻¹ = R(s)`, forme close `κ_p(s) = 1 − (1−p^{−s}) exp(A_p p^{−s})`. → **Théorème inlined** dans §3.2 avec preuve de 10-15 lignes (passage somme → produit via Plemelj-Smithies).
  - **Note 51 (236 l)** — *Opérateur canalisé C_n et tuning β* : opérateur `H_PT` pour couplage `Σ_p A_p p^{−s}`. → Définition + énoncé inlined §3.3, sketch dérivation 5 lignes.
  - **Note 52 (227 l)** — *ε_* géométrique R5* : `ε_*^geom = 1.51` canonique, élimination du tuning. → Lemme inlined §3.3 avec valeur géométrique, preuve sketchée.
  - **Note 62 (420 l)** — *Verrou G_HP3 : formule de trace explicite* : preuve de `−d log R/ds = Σ_p log p · [1/(p^s−1) + A_p p^{−s}]` en Re(s)>1. → **Théorème principal §3.4** avec preuve complète (manipulation produit eulérien + dérivée logarithmique).
  - **Note 64 (213 l)** — *G_HP4.d falsifié, trace régularisée correcte* : auto-correction technique, montre que spectre direct fail, trace régularisée est l'objet correct. → Remarque inlined §3.5 (paragraphe).
  - **Note 65 (529 l)** — *Verrous G_HP3.b/c : régularisation Hadamard* : trois régularisations équivalentes, trace régularisée canonique, 17 zéros capturés. → Définition régularisation Hadamard inlined §3.5 + énoncé final.

### §4 Localisation Re(s)=1/2 : 4 mécanismes
- **Notes référencées** : 58, 61, 63
- **À inliner** :
  - **Note 58 (257 l)** — *Lorentzien Berry-Keating PT* : métrique Fisher lorentzienne ; BK PT sur `u = log p` ; shift Haar multiplicatif. → Définition inlined §4.2 (Haar) + lemme shift.
  - **Note 61 (496 l)** — *Verrou G_HP2 : densité Riemann-von Mangoldt* : preuve `N_PT(γ) = (γ/2π) log(γ/2πe) + 1`, différence `1/8` avec `N_RvM`. → **Proposition principale §4.1** avec preuve complète (quantification semi-classique, aire microcanonique, ordre symétrique de Weyl).
  - **Note 63 (456 l)** — *Verrou G_HP4 : auto-adjonction effective* : indices de défaut `(1,1)` ; BC antipériodique `θ = π` forcée par `T_3`. → Proposition + sketch preuve inlined §4.4 (théorie de von Neumann, déficience).

### §5 Sur-détermination cohomologique 1/8
- **Notes référencées** : 63, 68
- **À inliner** :
  - **Note 63** (déjà couverte §4.4) — la BC antipériodique θ=π forcée par T_3 est utilisée ici pour la phase de Berry `1/8 = s/4`.
  - **Note 68 (560 l)** — *Verrou K3 cohomologique* : identification `1/8 = c_1/N_corners` via projecteur spinoriel `q_+/q_-`. → **Section centrale §5.2** : énoncé cohomologique + calcul explicite de c_1 (≈15 lignes), discussion du fibré spinoriel Kähler-Fisher.

### §6 Dualité Higgs ↔ zêta
- **Notes référencées** : 69
- **À inliner** :
  - **Note 69 (455 l)** — *Verrou K4 : dualité Higgs ↔ zêta via s=1/2* : bifurcateur canonique commun, chaîne `T1 + T2 + Haar + Weyl`. → **Théorème central §6.3** avec preuve par étapes (chaîne T1→T2→Haar→Weyl, ~20 lignes). Tests d'isolation §6.4 inlined (résultats numériques en tableau).

### §7 Vérifications numériques
- **Notes référencées** : 65, 74, 77, 78
- **À inliner** :
  - **Note 65** — Hadamard régularisation (déjà §3.5).
  - **Note 74 (218 l)** — *Scan P_max* : saturation à `P_max = 3000`. → Tableau de scan + paragraphe.
  - **Note 77 (217 l)** — *V2 extended range t∈[10,500]* : 267/269 à |Δ|<1.5. → Tableau détaillé par sous-bandes de 100 unités.
  - **Note 78 (171 l)** — *V5 ε-adaptatif falsifié* : convolution `|F|` brut falsifie, R1 ε=0.2 optimum. → Tableau comparatif R1/R2.

### §8 Auto-correction Dirichlet falsifié
- **Notes référencées** : 61, 69, 70, 71
- **À inliner** :
  - **Note 61** (déjà couverte §4.1) — reformulation : le 1/8 est asymptotique, pas mesurable.
  - **Note 69** (déjà couverte §6) — mécanisme Higgs↔ζ.
  - **Note 70 (492 l)** — *M1 test L Dirichlet mpmath* : test inconclusif. → Paragraphe résumé (le test naïf échoue par bruit).
  - **Note 71 (517 l)** — *M1bis LMFDB* : test rigoureux, χ²(PT)=7.49 vs χ²(H₀)=0.023 sur 4 d.d.l. → **Sous-section §8.3 complète** avec protocole, tableau résultats par caractère, conclusion.

### §9 Limites et ouvertures
- **Notes référencées** : 42, 61, 62, 63, 64, 65, 67, 68, 69, 71
- **À inliner** : aucune nouvelle démonstration. Cette section synthétise « ce qui est acquis » (énoncés exacts des théorèmes 1-10 déjà inlinés) et « ce qui reste ouvert » (G_HP3.a prolongement analytique strict, K3 rigueur formelle, K4 cohomologique unifié). Pas d'absorption supplémentaire — juste une bibliographie d'auto-référence interne propre.

### §10 Conclusion
- **Notes référencées** : 61
- **À inliner** : aucune. La conclusion est un résumé des positions épistémiques.

## Stratégie de continuation

1. **Phase 3** : convertir §3 (modèle spectral) — gros morceau, ~5 démonstrations inlinées (notes 42, 51, 52, 62, 64, 65). Estimation ~2h.
2. **Phase 4** : convertir §4 (localisation) — preuve principale densité N_PT à inliner. ~1.5h.
3. **Phase 5** : convertir §5 (cohomologie 1/8) — calcul c_1 inlined. ~1h.
4. **Phase 6** : convertir §6 (dualité Higgs↔ζ) — chaîne T1→T2→Haar→Weyl inlined. ~1.5h.
5. **Phase 7** : convertir §7 (vérifications num) — tableaux. ~30min.
6. **Phase 8** : convertir §8 (auto-correction Dirichlet) — protocole LMFDB inlined. ~1h.
7. **Phase 9** : convertir §9 + §10 (limites + conclusion). ~30min.
8. **Phase 10** : audit naming complet, build final FR, tag `v0.1-fr`. ~15min.
9. **Phase 11** : traduction EN intégrale (sections_en/) + build. ~3-4h.
10. **Phase 12** : commit + tag `v1.0-bilingual`.

**Total restant** : ~12h estimées (raisonnable sur 2-3 sessions).

## Annexes potentielles

Si certaines preuves dépassent 25 lignes (en particulier la formule de
trace §3.4 ou la dualité §6.3), les externaliser en annexes `annexes/A_*.tex`
plutôt que de gonfler le corps. À décider section par section au moment
de la conversion.
