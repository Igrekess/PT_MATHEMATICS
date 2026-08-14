# Plan d'implémentation — Article fondateur PT (FR, 40 p)

> **Pour les workers agentiques :** REQUIRED SUB-SKILL — `superpowers:subagent-driven-development` (recommandé) ou `superpowers:executing-plans`. Les étapes utilisent la syntaxe checkbox (`- [ ]`).

**Goal :** Produire `main.pdf` (≤ 40 p) — article LaTeX en français présentant les fondations PT (axiome `s=1/2`, théorèmes T0-T6, point fixe `μ*=15`, taxonomie premiers, cascade, lien RG/QFT, dérivation `α_EM`), conforme au [SPEC.md](SPEC.md).

**Architecture :** Article LaTeX modulaire. `main.tex` orchestrateur ; chaque section dans `sections/NN_xxx.tex`, chaque annexe dans `annexes/X_xxx.tex`. Préambule adapté de `PT_MONOGRAPHY/preamble.tex`. Bibliographie partagée avec entrées propres à l'article. Rédaction par phases : setup → corps (§1-§8) → annexes (A-E) → polissage → validation.

**Tech Stack :** LaTeX (classe `article`), `latexmk` pour le build, `biblatex+biber` pour la bibliographie, git pour l'historique (init local dans `PT_ARTICLES/PT_FOUNDATIONS/`). PT-RAG (`pt_search`) pour retrouver les sources monographe lors de chaque rédaction.

**Conventions critiques (non-négociables) :** voir SPEC §6. Lint automatisé : aucune occurrence de `q_stat`, `q_therm`, `ghost`, `\qstat`, `\qtherm` dans les sources `.tex`.

---

## Structure des fichiers

```
PT_ARTICLES/PT_FOUNDATIONS/
├── SPEC.md                 (déjà écrit)
├── PLAN.md                 (ce fichier)
├── README.md               (statut, build, scope)
├── main.tex                (orchestrateur)
├── preamble.tex            (macros, packages, métadonnées)
├── references.bib          (bibliographie)
├── latexmkrc               (config build : XeLaTeX/pdfLaTeX, biber)
├── .gitignore              (PDF, aux, log, etc.)
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_cadre_formel.tex
│   ├── 03_theoremes.tex
│   ├── 04_point_fixe_premiers.tex
│   ├── 05_cascade.tex
│   ├── 06_rg_qft.tex
│   ├── 07_alpha_em.tex
│   └── 08_conclusion.tex
├── annexes/
│   ├── A_T0_T1.tex
│   ├── B_T3.tex
│   ├── C_T5.tex
│   ├── D_T6.tex
│   └── E_glossaire.tex
└── figures/
    ├── cascade.tex         (TikZ : diagramme cascade γ_3→γ_3γ_5→γ_3γ_5γ_7)
    ├── taxonomie.tex       (TikZ : actifs/échos/super-échos)
    └── tableau_themes.tex  (LaTeX tabular : T0-T6 récap)
```

**Pourquoi cette décomposition :** chaque section .tex tient en contexte ; les modifications restent locales ; la compilation totale ne dépend pas d'une réécriture monolithique. Aligné avec les conventions des articles existants (`PT_PROJECTS/PT_GeoFlow/papers/*.tex`).

---

## Phase 0 — Setup du projet

### Task 0.1 : Arborescence + git

**Files :**
- Create : `PT_ARTICLES/PT_FOUNDATIONS/{sections,annexes,figures}/`
- Create : `PT_ARTICLES/PT_FOUNDATIONS/.gitignore`
- Create : `PT_ARTICLES/PT_FOUNDATIONS/README.md`

- [ ] **Step 1 :** Créer l'arborescence

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
mkdir -p sections annexes figures
```

- [ ] **Step 2 :** Écrire `.gitignore`

```gitignore
# LaTeX auxiliaires
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
*.toc
*.lof
*.lot

# Build artifact
main.pdf

# Editor
.DS_Store
.vscode/
*.swp
```

- [ ] **Step 3 :** Écrire `README.md` minimal

```markdown
# PT Foundations — Article fondateur (FR)

Article LaTeX en français présentant les fondations théoriques de la Théorie de la Persistance : axiome `s = 1/2`, théorèmes T0-T6, point fixe μ*=15, taxonomie des premiers (actifs/échos/super-échos), principe de cascade, lien RG/QFT, dérivation `α_EM`.

**Cible :** 40 pages max.
**Statut :** en rédaction. Voir [SPEC.md](SPEC.md) et [PLAN.md](PLAN.md).

## Build

\`\`\`bash
latexmk -pdf main.tex
\`\`\`

Sortie : `main.pdf`.

## Auteur

Yan Senez (PT corpus), 2026.
```

- [ ] **Step 4 :** Initialiser git localement

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
git init
git add SPEC.md PLAN.md README.md .gitignore
git commit -m "chore: initialize PT foundations article project"
```

Expected : commit `Initial commit` créé, `git log` affiche une entrée.

---

### Task 0.2 : Préambule LaTeX

**Files :**
- Read : `PT_MONOGRAPHY/preamble.tex` (référence)
- Create : `PT_ARTICLES/PT_FOUNDATIONS/preamble.tex`

- [ ] **Step 1 :** Lire le préambule du monographe pour extraire les macros utiles

```bash
# Identifier les packages et macros nécessaires pour un article (PAS pour un livre 900 p)
```

Sources à consulter via PT-RAG :
- `pt_get_document path="/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_MONOGRAPHY/preamble.tex"`

- [ ] **Step 2 :** Écrire `preamble.tex` ciblé article

Contenu minimal requis :

```latex
% ============================================================
% Préambule — PT Foundations (article FR, 40 p)
% Adapté de PT_MONOGRAPHY/preamble.tex
% ============================================================

\documentclass[11pt,a4paper]{article}

% --- Encodage et langue ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{csquotes}

% --- Mathématiques ---
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{bm}

% --- Mise en page ---
\usepackage[margin=2.5cm]{geometry}
\usepackage{microtype}
\usepackage{enumitem}

% --- Tableaux et figures ---
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc}

% --- Bibliographie ---
\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}
\addbibresource{references.bib}

% --- Hyperliens ---
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=green!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{cleveref}

% --- Environnements de théorème ---
\theoremstyle{plain}
\newtheorem{theoreme}{Théorème}[section]
\newtheorem{proposition}[theoreme]{Proposition}
\newtheorem{lemme}[theoreme]{Lemme}
\newtheorem{corollaire}[theoreme]{Corollaire}

\theoremstyle{definition}
\newtheorem{definition}[theoreme]{Définition}
\newtheorem{convention}[theoreme]{Convention}

\theoremstyle{remark}
\newtheorem{remarque}[theoreme]{Remarque}
\newtheorem{exemple}[theoreme]{Exemple}

% --- Macros PT canoniques ---
% Secteurs (convention 2026 : q_+ / q_-, jamais q_stat/q_therm)
\newcommand{\qplus}{q_{+}}
\newcommand{\qminus}{q_{-}}

% Constantes structurelles
\newcommand{\mustar}{\mu^{*}}
\newcommand{\sper}{s}

% Dimensions anomales
\newcommand{\gammap}[1]{\gamma_{#1}}
\newcommand{\gammathree}{\gamma_{3}}
\newcommand{\gammafive}{\gamma_{5}}
\newcommand{\gammaseven}{\gamma_{7}}

% Phases cycliques
\newcommand{\thetap}[1]{\theta_{#1}}
\newcommand{\sinp}[1]{\sin^{2}\theta_{#1}}

% Status tags (compteur C1-C12 : 5 [THM] + 6 [DER]/[COND] + 1 [VAL], requalification B4 du 2026-08-14)
\newcommand{\THM}{\textsc{[Thm]}}

% --- Théorèmes nommés ---
\newcommand{\Tzero}{\textbf{T0}}
\newcommand{\Tone}{\textbf{T1}}
\newcommand{\Ttwo}{\textbf{T2}}
\newcommand{\Tthree}{\textbf{T3}}
\newcommand{\Tfour}{\textbf{T4}}
\newcommand{\Tfive}{\textbf{T5}}
\newcommand{\Tsix}{\textbf{T6}}

% --- Métadonnées article ---
\title{La Théorie de la Persistance : axiome, théorèmes et cascade des nombres premiers actifs}
\author{Yan Senez}
\date{\today}
```

- [ ] **Step 3 :** Vérifier qu'aucune macro `\qstat`/`\qtherm` n'a été copiée par inadvertance

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
grep -nE "qstat|qtherm" preamble.tex
```

Expected : aucune sortie (grep silencieux).

- [ ] **Step 4 :** Commit

```bash
git add preamble.tex
git commit -m "feat: add LaTeX preamble with PT canonical macros (q+/q-, no q_stat/q_therm)"
```

---

### Task 0.3 : Squelette `main.tex`

**Files :**
- Create : `PT_ARTICLES/PT_FOUNDATIONS/main.tex`

- [ ] **Step 1 :** Écrire `main.tex`

```latex
\input{preamble}

\begin{document}

\maketitle

\begin{abstract}
\noindent
La Théorie de la Persistance dérive les constantes du Modèle Standard depuis un unique axiome arithmétique~$\sper=1/2$. Cet article présente, de façon autonome et concise, les fondations de ce cadre : la catégorie sous-jacente, les six théorèmes structurants $\Tzero$--$\Tsix$, l'apparition du point fixe~$\mustar=15$, la taxonomie complète des nombres premiers (actifs $\{3,5,7\}$, échos $\{11,13\}$, super-échos $\{17,19,23\}$), et le principe de cascade arithmétique qui en résulte. La cohérence avec le flot de renormalisation de la théorie quantique des champs est explicitée, et l'on déroule, jusqu'à la valeur numérique, la dérivation de la constante de structure fine~$\alpha_{\mathrm{EM}}$ à zéro paramètre ajusté.
\end{abstract}

\tableofcontents

\input{sections/01_introduction}
\input{sections/02_cadre_formel}
\input{sections/03_theoremes}
\input{sections/04_point_fixe_premiers}
\input{sections/05_cascade}
\input{sections/06_rg_qft}
\input{sections/07_alpha_em}
\input{sections/08_conclusion}

\appendix
\input{annexes/A_T0_T1}
\input{annexes/B_T3}
\input{annexes/C_T5}
\input{annexes/D_T6}
\input{annexes/E_glossaire}

\printbibliography

\end{document}
```

- [ ] **Step 2 :** Créer les stubs des 8 sections et 5 annexes (fichiers vides + commentaire)

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
for f in 01_introduction 02_cadre_formel 03_theoremes 04_point_fixe_premiers 05_cascade 06_rg_qft 07_alpha_em 08_conclusion; do
  printf "%% TODO: rédaction\n\\section{Placeholder %s}\n%% \\input par main.tex\n" "$f" > "sections/${f}.tex"
done
for f in A_T0_T1 B_T3 C_T5 D_T6 E_glossaire; do
  printf "%% TODO: rédaction annexe\n\\section{Placeholder annexe %s}\n" "$f" > "annexes/${f}.tex"
done
```

- [ ] **Step 3 :** Créer `references.bib` minimal (sera enrichi à Task 0.5)

```bibtex
% Bibliographie PT Foundations
% Entrées propres à l'article + références externes

@book{PT_Monograph,
  author = {Senez, Yan},
  title  = {La Théorie de la Persistance --- Monographe},
  year   = {2026},
  note   = {Manuscrit, version 2026-05}
}
```

- [ ] **Step 4 :** Créer `latexmkrc`

```perl
# latexmkrc — config build PT Foundations
$pdf_mode = 1;       # pdfLaTeX
$bibtex_use = 2;     # biber (via biblatex)
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$out_dir = '.';
$clean_ext = 'synctex.gz run.xml bcf';
```

- [ ] **Step 5 :** Commit

```bash
git add main.tex references.bib latexmkrc sections/ annexes/
git commit -m "feat: scaffold article structure (main.tex + section stubs + bib + latexmkrc)"
```

---

### Task 0.4 : Première compilation

**Files :** aucun nouveau, validation des Tasks 0.2-0.3.

- [ ] **Step 1 :** Compiler

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
latexmk -pdf main.tex
```

Expected : exit 0, `main.pdf` créé, ~3-5 pages (titre + abstract + TOC + placeholders).

- [ ] **Step 2 :** Vérifier le compte de pages du squelette

```bash
pdfinfo main.pdf | grep "^Pages:"
```

Expected : `Pages: 3` à `Pages: 6` (avec placeholders).

- [ ] **Step 3 :** Si erreurs LaTeX, lire `main.log` ; corriger preamble.tex ou main.tex ; recompiler.

- [ ] **Step 4 :** Commit du PDF de référence (snapshot du build sain)

```bash
git add main.pdf || true   # PDF dans .gitignore, mais on veut savoir si build OK
git commit --allow-empty -m "build: first successful compilation (skeleton, ~5p)"
```

---

### Task 0.5 : Enrichir `references.bib`

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/references.bib`

- [ ] **Step 1 :** Identifier les sources externes à citer (théorie des cribles, RG, MS, Wilson coefficients, géométrie de Fisher)

Sources externes attendues (à confirmer selon le contenu rédigé) :
- Wilson, K.G. *The renormalization group and critical phenomena.* Rev. Mod. Phys. 55 (1983).
- Particle Data Group, *Review of Particle Physics* (édition récente).
- Iwaniec & Kowalski, *Analytic Number Theory* (cribles).
- Buchbinder, Odintsov, Shapiro, *Effective Action in Quantum Gravity* (RG).
- Référence locale `PT_Monograph` (déjà ajoutée).

- [ ] **Step 2 :** Ajouter ~10-15 entrées BibTeX dans `references.bib`

Exemple d'entrée à intégrer :

```bibtex
@article{Wilson1983,
  author  = {Wilson, K. G.},
  title   = {The renormalization group and critical phenomena},
  journal = {Rev. Mod. Phys.},
  volume  = {55},
  pages   = {583--600},
  year    = {1983},
  doi     = {10.1103/RevModPhys.55.583}
}

@book{IwaniecKowalski2004,
  author    = {Iwaniec, Henryk and Kowalski, Emmanuel},
  title     = {Analytic Number Theory},
  publisher = {American Mathematical Society},
  series    = {Colloquium Publications},
  volume    = {53},
  year      = {2004}
}

@article{PDG2024,
  author  = {{Particle Data Group}},
  title   = {Review of Particle Physics},
  journal = {Phys. Rev. D},
  year    = {2024},
  note    = {Édition consultée : 2024}
}
```

- [ ] **Step 3 :** Vérifier que `biber` accepte le fichier

```bash
biber --validate-datamodel --validate-control references.bib 2>&1 | head -20
```

Expected : aucune erreur de parsing.

- [ ] **Step 4 :** Commit

```bash
git add references.bib
git commit -m "feat: seed references.bib with external sources (RG, sieves, PDG, monograph)"
```

---

## Phase 1 — Rédaction du corps (§1 à §8)

**Méthode commune à chaque section :**
1. Consulter le monographe via PT-RAG pour les sources.
2. Rédiger la section dans son fichier dédié.
3. Compiler.
4. Audit naming (`grep` pour interdictions).
5. Vérifier budget pages local.
6. Commit.

---

### Task 1.1 : §1 Introduction (3 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/01_introduction.tex`

- [ ] **Step 1 :** Sources monographe à consulter

```
pt_search query="problème des paramètres libres Modèle Standard 19 nombres ajustés"
  filters={"language": "fr", "project": ["monography/chapters_fr", "monography/frontmatter_fr"]}
pt_search query="axiome s=1/2 motivation auto-cohérence crible"
  filters={"language": "fr"}
pt_search query="aperçu cascade γ_3 γ_5 γ_7 résumé pédagogique"
  filters={"language": "fr"}
```

Lire en particulier `chapters_fr/ch01_sieve.tex`, `frontmatter_fr/preface.tex` (s'il existe), `essays/principes-pt/*.mdx` (résumés narratifs).

- [ ] **Step 2 :** Rédiger les 4 sous-sections

Structure conforme au SPEC §5 :
- 1.1 Paramètres libres du MS (~0.7 p)
- 1.2 L'axiome `s = 1/2` (~0.7 p)
- 1.3 Aperçu cascade + teaser `α_EM` (~1 p)
- 1.4 Plan + conventions + statut C1-C12 = 5 [THM] + 6 [DER]/[COND] + 1 [VAL] (~0.6 p)

Forme du fichier :

```latex
\section{Introduction}
\label{sec:intro}

\subsection{Le problème des paramètres libres}
\label{sec:intro:libres}
% ...

\subsection{L'axiome de persistance \texorpdfstring{$\sper = 1/2$}{s = 1/2}}
\label{sec:intro:axiome}
% ...

\subsection{Aperçu : cascade arithmétique et constante de structure fine}
\label{sec:intro:apercu}
% ...

\subsection{Plan, conventions, statut}
\label{sec:intro:plan}
% ...
```

- [ ] **Step 3 :** Compilation

```bash
latexmk -pdf main.tex
```

Expected : exit 0, pas de nouveau warning critique.

- [ ] **Step 4 :** Naming audit

```bash
grep -nE "qstat|qtherm|q_stat|q_therm|ghost" sections/01_introduction.tex && echo "FAIL" || echo "OK"
```

Expected : `OK`.

- [ ] **Step 5 :** Budget pages local

```bash
# Compter les pages de la section dans le PDF
# Approximation : différence entre Pages après et avant ajout
pdfinfo main.pdf | grep "^Pages:"
```

Cible : §1 occupe ~3 p. Si > 4 p, élaguer ; si < 2 p, dire au worker de densifier.

- [ ] **Step 6 :** Commit

```bash
git add sections/01_introduction.tex main.pdf
git commit -m "feat(intro): write §1 introduction (libres du MS, axiome s=1/2, teaser α_EM, plan)"
```

---

### Task 1.2 : §2 Cadre formel (4 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/02_cadre_formel.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="catégorie L0 Set pointé ZFC fermeture BA0"
  filters={"language": "fr", "project": ["monography/chapters_fr", "monography/ch09_bridge"]}
pt_search query="crible PT modulo 3 5 7 phases cycliques θ_p"
  filters={"language": "fr"}
pt_search query="secteurs q_+ q_- définition canonique"
  filters={"language": "fr"}
pt_get_document path="/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_MONOGRAPHY/NOMENCLATURE_BILINGUE.md"
```

- [ ] **Step 2 :** Rédiger les 4 sous-sections

- 2.1 Catégorie `L0` (~1.2 p)
- 2.2 Crible PT mod {3,5,7} + phases `θ_p` (~1.2 p)
- 2.3 Secteurs `\qplus` / `\qminus` (~0.8 p) — utiliser explicitement les macros, jamais le texte brut `q_stat`
- 2.4 Table de correspondance bilingue physique↔math (~0.8 p, format `tabular`)

- [ ] **Step 3 :** Compilation + audit naming + budget pages

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|q_stat|q_therm|ghost" sections/02_cadre_formel.tex && echo "FAIL" || echo "OK"
pdfinfo main.pdf | grep "^Pages:"
```

Expected : compile OK, naming OK, §2 ajoute ~4 p au total.

- [ ] **Step 4 :** Commit

```bash
git add sections/02_cadre_formel.tex main.pdf
git commit -m "feat(cadre): write §2 formal framework (L0, sieve mod {3,5,7}, q±, bilingual table)"
```

---

### Task 1.3 : §3 Les six théorèmes T0-T6 (9 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/03_theoremes.tex`

C'est la section la plus longue. Décomposer la rédaction théorème par théorème ; un commit par théorème pour granularité fine.

- [ ] **Step 1 :** Sources monographe pour T0-T6

```
pt_search query="théorème T0 BA0 closing s=1/2 forcé fermeture"
  filters={"language": "fr"}
pt_search query="théorème T1 transitions interdites spectre"
  filters={"language": "fr"}
pt_search query="théorème T2 unicité isomorphisme persistance"
  filters={"language": "fr"}
pt_search query="théorème T3 antidiagonale matrice transition mod 3"
  filters={"language": "fr"}
pt_search query="théorème T4 conservation charge cyclique cascade"
  filters={"language": "fr"}
pt_search query="théorème T5 géométrie Fisher T³ trois cosinus"
  filters={"language": "fr"}
pt_search query="théorème T6 stabilité activation premiers seuil"
  filters={"language": "fr"}
```

Lire en particulier les chapitres `monography/chapters_fr/ch01_sieve.tex` à `ch08_fixed_point.tex` ainsi que les pages du site `website/theorems/T0` à `T6` pour le format pédagogique.

- [ ] **Step 2 :** Structure du fichier (squelette)

```latex
\section{L'axiome \texorpdfstring{$\sper = 1/2$}{s=1/2} et les six théorèmes}
\label{sec:theoremes}

\subsection{\Tzero{} --- Fermeture du crible (\textit{Closing})}
\label{sec:theoremes:T0}
% Énoncé + idée de preuve + conséquence + renvoi annexe A
%
% Format théorème encadré :
\begin{theoreme}[\Tzero, fermeture]
\label{thm:T0}
% Énoncé précis
\end{theoreme}
% ~1.5 p

\subsection{\Tone{} --- Transitions interdites}
\label{sec:theoremes:T1}
% ~1.5 p ; renvoi annexe A

\subsection{\Ttwo{} --- Unicité de la persistance}
\label{sec:theoremes:T2}
% ~1 p ; PREUVE COMPLÈTE dans le corps (courte)

\subsection{\Tthree{} --- Antidiagonale}
\label{sec:theoremes:T3}
% ~1.5 p ; renvoi annexe B

\subsection{\Tfour{} --- Conservation cyclique}
\label{sec:theoremes:T4}
% ~1 p ; PREUVE COMPLÈTE dans le corps

\subsection{\Tfive{} --- Géométrie de Fisher sur $T^{3}$}
\label{sec:theoremes:T5}
% ~1.5 p ; renvoi annexe C

\subsection{\Tsix{} --- Stabilité et seuillage des premiers}
\label{sec:theoremes:T6}
% ~1 p ; renvoi annexe D
```

- [ ] **Step 3 :** Rédiger T0 et T1 (renvoi annexe A)

Rédaction conforme au format encadré (énoncé + 1-2 paragraphes d'idée de preuve + référence `\cref{app:T0T1}` à l'annexe A).

Compile + audit + commit :

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" sections/03_theoremes.tex && echo "FAIL" || echo "OK"
git add sections/03_theoremes.tex main.pdf
git commit -m "feat(theoremes): write T0 (Closing) and T1 (transitions interdites)"
```

- [ ] **Step 4 :** Rédiger T2 (preuve directe dans le corps)

T2 doit avoir une preuve courte et complète sur place, pas de renvoi annexe.

```bash
latexmk -pdf main.tex
git add sections/03_theoremes.tex main.pdf
git commit -m "feat(theoremes): write T2 (unicité) with direct proof"
```

- [ ] **Step 5 :** Rédiger T3 (renvoi annexe B)

```bash
latexmk -pdf main.tex
git add sections/03_theoremes.tex main.pdf
git commit -m "feat(theoremes): write T3 (antidiagonale)"
```

- [ ] **Step 6 :** Rédiger T4 (preuve directe dans le corps)

```bash
latexmk -pdf main.tex
git add sections/03_theoremes.tex main.pdf
git commit -m "feat(theoremes): write T4 (conservation cyclique) with direct proof"
```

- [ ] **Step 7 :** Rédiger T5 (renvoi annexe C) — section sensible, mentionner les trois cosinus de Fisher dérivés en C

```bash
latexmk -pdf main.tex
git add sections/03_theoremes.tex main.pdf
git commit -m "feat(theoremes): write T5 (géométrie de Fisher sur T³)"
```

- [ ] **Step 8 :** Rédiger T6 (renvoi annexe D)

```bash
latexmk -pdf main.tex
git add sections/03_theoremes.tex main.pdf
git commit -m "feat(theoremes): write T6 (stabilité et seuillage des premiers)"
```

- [ ] **Step 9 :** Budget pages final de §3

```bash
pdfinfo main.pdf | grep "^Pages:"
```

Cible : section §3 totalise ~9 p. Si > 10 p, élaguer dans le plus long (T0 ou T5). Si < 8 p, densifier ou ajouter remarques utiles.

---

### Task 1.4 : §4 Point fixe μ* et taxonomie premiers (6 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/04_point_fixe_premiers.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="point fixe μ*=15 3+5+7 auto-cohérence preuve"
  filters={"language": "fr"}
pt_search query="dimensions anomales γ_3 γ_5 γ_7 fractions exactes"
  filters={"language": "fr", "project": ["monography/ch06_holonomy"]}
pt_search query="premiers actifs {3,5,7} exhaustivité T6"
  filters={"language": "fr"}
pt_search query="premiers échos {11,13} Wilson C9 C10 β_echo IR universel"
  filters={"language": "fr"}
pt_search query="super-échos {17,19,23} scale-dependent hadronique"
  filters={"language": "fr"}
pt_search query="premiers inactifs F_inactive p≥29 marge"
  filters={"language": "fr"}
```

- [ ] **Step 2 :** Rédiger les 6 sous-sections (cf. SPEC §5.4)

Particulièrement vigilant sur :
- 4.4 : utiliser le terme « échos » exclusivement (jamais « ghost »).
- 4.4 : préciser que `{11,13}` sont **strictement** échos, jamais super-échos (cf. feedback mémoire interne).
- 4.5 : super-échos `{17,19,23}` sont disjoints, n'incluent pas 11/13.

Inclure un tableau récapitulatif (figures/taxonomie.tex à créer) :

```latex
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lcll@{}}
\toprule
Classe & Premiers & Rôle & Référence \\
\midrule
Canal de parité & $\{2\}$ & Information/anti-information & \cref{sec:cadre:l0} \\
Actifs & $\{3,5,7\}$ & Cascade, $\mustar = 3{+}5{+}7$ & \cref{sec:theoremes:T6} \\
Échos & $\{11,13\}$ & IR universel ($\beta_{\mathrm{echo}}$) & \cref{sec:point_fixe:echos} \\
Super-échos & $\{17,19,23\}$ & Scale-dependent (hadronique) & \cref{sec:point_fixe:super} \\
Inactifs & $\{p \geq 29\}$ & Contribution collective $F_{\mathrm{inactive}}\approx 0$ & \cref{sec:point_fixe:inactifs} \\
\bottomrule
\end{tabular}
\caption{Taxonomie des nombres premiers en PT.}
\label{tab:taxonomie}
\end{table}
```

- [ ] **Step 3 :** Compilation + audit + budget

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" sections/04_point_fixe_premiers.tex && echo "FAIL" || echo "OK"
# Vérifier que "écho" est utilisé (pas "ghost") :
grep -c "écho" sections/04_point_fixe_premiers.tex
pdfinfo main.pdf | grep "^Pages:"
```

Expected : naming OK, `écho` apparaît plusieurs fois (>5), section ajoute ~6 p.

- [ ] **Step 4 :** Commit

```bash
git add sections/04_point_fixe_premiers.tex main.pdf
git commit -m "feat(point-fixe): write §4 (μ*=15, γ_p, taxonomy: actifs/échos/super-échos/inactifs)"
```

---

### Task 1.5 : §5 Le principe de cascade (5 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/05_cascade.tex`
- Create : `PT_ARTICLES/PT_FOUNDATIONS/figures/cascade.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="principe cascade arithmétique définition formelle raffinements modulaires"
  filters={"language": "fr"}
pt_search query="cascade canonique γ_3 γ_3γ_5 γ_3γ_5γ_7 amplitude cumulative"
  filters={"language": "fr"}
pt_search query="cumulants conditionnement points de stabilité commutativité diagramme"
  filters={"language": "fr"}
pt_search query="μ* attracteur cascade T6"
  filters={"language": "fr"}
```

- [ ] **Step 2 :** Créer figure TikZ `figures/cascade.tex`

Diagramme commutatif simplifié : étages `(∅) → (3) → (3,5) → (3,5,7)` avec γ_p en flèches.

```latex
% figures/cascade.tex
\begin{tikzpicture}[node distance=2.5cm, >={Stealth[scale=1.2]}]
  \node (etage0) {$(\emptyset)$};
  \node (etage1) [right=of etage0] {$(3)$};
  \node (etage2) [right=of etage1] {$(3,5)$};
  \node (etage3) [right=of etage2] {$(3,5,7) \to \mustar$};

  \draw[->] (etage0) -- node[above] {$\gammathree$} (etage1);
  \draw[->] (etage1) -- node[above] {$\gammafive$} (etage2);
  \draw[->] (etage2) -- node[above] {$\gammaseven$} (etage3);
\end{tikzpicture}
```

- [ ] **Step 3 :** Rédiger les 4 sous-sections (cf. SPEC §5.5)

- [ ] **Step 4 :** Compilation + audit + budget

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" sections/05_cascade.tex && echo "FAIL" || echo "OK"
pdfinfo main.pdf | grep "^Pages:"
```

- [ ] **Step 5 :** Commit

```bash
git add sections/05_cascade.tex figures/cascade.tex main.pdf
git commit -m "feat(cascade): write §5 (principe de cascade, fig. TikZ, propriétés universelles)"
```

---

### Task 1.6 : §6 Cohérence avec RG/QFT (3 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/06_rg_qft.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="flot renormalisation RG cascade discrétisée échelle μ*"
  filters={"language": "fr"}
pt_search query="β_echo Wilson coefficient identité dimension anomale γ_p"
  filters={"language": "fr"}
pt_search query="phases θ_p angles mélange correspondance"
  filters={"language": "fr"}
pt_search query="PT théorie continue discret émerge points stabilité"
  filters={"language": "fr"}
```

Sources externes à citer : Wilson 1983 (RG), revue PDG.

- [ ] **Step 2 :** Rédiger les 3 sous-sections (cf. SPEC §5.6)

**Point critique** (cf. feedback interne) : §6.3 doit énoncer clairement que **PT n'est PAS une théorie du discret**. Le discret émerge comme points de stabilité d'une dynamique continue. Ne pas opposer PT au continu via substitution simpliste.

- [ ] **Step 3 :** Compilation + audit + budget

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" sections/06_rg_qft.tex && echo "FAIL" || echo "OK"
# Vérifier la présence du caveat anti-discret :
grep -i "continu" sections/06_rg_qft.tex
pdfinfo main.pdf | grep "^Pages:"
```

Expected : naming OK, mention explicite du caveat continu/discret, ~3 p.

- [ ] **Step 4 :** Commit

```bash
git add sections/06_rg_qft.tex main.pdf
git commit -m "feat(rg): write §6 (RG/QFT correspondences, caveat: PT not a theory of the discrete)"
```

---

### Task 1.7 : §7 Application canonique α_EM (4 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/07_alpha_em.tex`

- [ ] **Step 1 :** Sources monographe — section critique car la dérivation doit être complète

```
pt_search query="α_EM dérivation holonomie sin²θ_p T³ zéro paramètre"
  filters={"language": "fr"}
pt_search query="constante structure fine 1/137 formule explicite cascade"
  filters={"language": "fr"}
pt_search query="ch10 fine_structure α_EM"
  filters={"language": "fr", "project": ["monography/ch10_fine_structure"]}
pt_search query="holonomy T³ sin² θ_p ch06_holonomy"
  filters={"language": "fr", "project": ["monography/ch06_holonomy"]}
```

Lire intégralement `chapters_fr/ch10_fine_structure.tex` et `chapters_fr/ch06_holonomy.tex` pour la dérivation exacte.

- [ ] **Step 2 :** Rédiger les 3 sous-sections (cf. SPEC §5.7)

Structure :
- 7.1 Holonomie sur T³ et `sin²θ_p` (~1 p)
- 7.2 Dérivation complète `α_EM = …` avec chaîne explicite `s=1/2 → T1 → T3 → T5 → μ* → holonomie → α_EM` (~2 p)
- 7.3 Comparaison numérique avec valeur mesurée, résidu (~1 p)

**Critère d'acceptation strict :** la dérivation doit aboutir à une valeur numérique. Pas de renvoi extérieur (cf. SPEC §10).

- [ ] **Step 3 :** Compilation + audit + budget

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" sections/07_alpha_em.tex && echo "FAIL" || echo "OK"
# Vérifier que α_EM apparaît avec une valeur numérique :
grep -E "1/137|0\.00729|7\.29.*10" sections/07_alpha_em.tex
pdfinfo main.pdf | grep "^Pages:"
```

Expected : naming OK, valeur numérique présente, ~4 p.

- [ ] **Step 4 :** Commit

```bash
git add sections/07_alpha_em.tex main.pdf
git commit -m "feat(alpha-em): write §7 (canonical application: derivation of α_EM ≈ 1/137, zero param)"
```

---

### Task 1.8 : §8 Conclusion et perspectives (2 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/sections/08_conclusion.tex`

- [ ] **Step 1 :** Sources monographe (pointeurs aval)

```
pt_search query="applications PT chimie NMR allométrie cosmologie BSM"
  filters={"language": "fr"}
pt_search query="ch26 conclusion synthèse 10/10 [THM]"
  filters={"language": "fr"}
```

- [ ] **Step 2 :** Rédiger les 4 sous-sections (cf. SPEC §5.8)

- 8.1 Récap (~0.5 p)
- 8.2 Statut C1-C12 = 5 [THM] + 6 [DER]/[COND] + 1 [VAL] (~0.3 p)
- 8.3 Perspectives en une ligne chacune : Yukawa, chimie (`PUBLIC/PT_CHEMISTRY`), NMR, allométrie, cosmologie, BSM (~0.8 p)
- 8.4 Peer review externe (~0.4 p)

- [ ] **Step 3 :** Compilation + audit + budget

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" sections/08_conclusion.tex && echo "FAIL" || echo "OK"
pdfinfo main.pdf | grep "^Pages:"
```

- [ ] **Step 4 :** Commit

```bash
git add sections/08_conclusion.tex main.pdf
git commit -m "feat(conclusion): write §8 (recap, 10/10 [THM], perspectives, open question: peer review)"
```

---

## Phase 2 — Rédaction des annexes

### Task 2.1 : Annexe A — Démos T0 et T1 (1 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/annexes/A_T0_T1.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="preuve complète T0 BA0 closing démonstration fermeture"
  filters={"language": "fr"}
pt_search query="preuve complète T1 transitions interdites démonstration"
  filters={"language": "fr"}
pt_search query="ch25 BA0_closing preuve s=1/2"
  filters={"language": "fr", "project": ["monography/ch25_BA0_closing"]}
```

- [ ] **Step 2 :** Rédiger l'annexe (compact, rigoureux)

```latex
\section{Démonstrations de \Tzero{} et \Tone}
\label{app:T0T1}

\subsection*{Preuve de \Tzero}
% Preuve complète et rigoureuse en ~0.5 p

\subsection*{Preuve de \Tone}
% ~0.5 p
```

- [ ] **Step 3 :** Compilation + audit + budget

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" annexes/A_T0_T1.tex && echo "FAIL" || echo "OK"
pdfinfo main.pdf | grep "^Pages:"
```

- [ ] **Step 4 :** Commit

```bash
git add annexes/A_T0_T1.tex main.pdf
git commit -m "feat(annexes): write Annex A (proofs of T0, T1)"
```

---

### Task 2.2 : Annexe B — Démo T3 (0.5 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/annexes/B_T3.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="preuve complète T3 antidiagonale calcul direct mod 3"
  filters={"language": "fr"}
```

- [ ] **Step 2 :** Rédiger (compact)

```latex
\section{Démonstration de \Tthree}
\label{app:T3}
% Calcul direct, ~0.5 p
```

- [ ] **Step 3 :** Compilation + audit

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" annexes/B_T3.tex && echo "FAIL" || echo "OK"
```

- [ ] **Step 4 :** Commit

```bash
git add annexes/B_T3.tex main.pdf
git commit -m "feat(annexes): write Annex B (proof of T3, antidiagonal)"
```

---

### Task 2.3 : Annexe C — Démo T5 (1 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/annexes/C_T5.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="preuve T5 Fisher géométrie T³ trois cosinus dérivation unifiée"
  filters={"language": "fr"}
pt_search query="ch15 SM_observables cosinus Fisher α_35"
  filters={"language": "fr", "project": ["monography/ch15_sm_observables"]}
```

- [ ] **Step 2 :** Rédiger l'annexe (rigoureux, ~1 p)

```latex
\section{Démonstration de \Tfive}
\label{app:T5}
% Dérivation des trois cosinus de Fisher, ~1 p
```

- [ ] **Step 3 :** Compilation + audit

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" annexes/C_T5.tex && echo "FAIL" || echo "OK"
```

- [ ] **Step 4 :** Commit

```bash
git add annexes/C_T5.tex main.pdf
git commit -m "feat(annexes): write Annex C (proof of T5, Fisher geometry, three cosines)"
```

---

### Task 2.4 : Annexe D — Démo T6 (0.5 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/annexes/D_T6.tex`

- [ ] **Step 1 :** Sources monographe

```
pt_search query="preuve T6 stabilité seuil activation premiers fondement taxonomie"
  filters={"language": "fr"}
```

- [ ] **Step 2 :** Rédiger (compact)

```latex
\section{Démonstration de \Tsix}
\label{app:T6}
% Critère de stabilité, ~0.5 p
```

- [ ] **Step 3 :** Compilation + audit

```bash
latexmk -pdf main.tex
grep -nE "qstat|qtherm|ghost" annexes/D_T6.tex && echo "FAIL" || echo "OK"
```

- [ ] **Step 4 :** Commit

```bash
git add annexes/D_T6.tex main.pdf
git commit -m "feat(annexes): write Annex D (proof of T6, stability and prime activation)"
```

---

### Task 2.5 : Annexe E — Glossaire bilingue (1 p)

**Files :**
- Modify : `PT_ARTICLES/PT_FOUNDATIONS/annexes/E_glossaire.tex`

- [ ] **Step 1 :** Sources

```
pt_get_document path="/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_MONOGRAPHY/NOMENCLATURE_BILINGUE.md"
pt_get_document path="/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_MONOGRAPHY/frontmatter_fr/glossary.tex"
```

- [ ] **Step 2 :** Rédiger la table de correspondance bilingue

```latex
\section{Glossaire bilingue physique\,\textendash\,mathématique}
\label{app:glossaire}

\begin{table}[htbp]
\centering
\small
\begin{tabular}{@{}p{4cm}p{4cm}p{6cm}@{}}
\toprule
Symbole & Math & Physique \\
\midrule
$\sper = 1/2$ & paramètre de persistance & exposant critique du crible \\
$\mustar = 15$ & point fixe arithmétique & échelle d'équilibre de la cascade \\
$\gammap{p}$ & dimension anomale & coefficient de couplage à l'étage $p$ \\
$\thetap{p}$ & phase cyclique mod $p$ & angle de mélange \\
$\sinp{p}$ & $\sin^2 \thetap{p}$ & probabilité d'holonomie \\
$\qplus, \qminus$ & secteurs $+/-$ de $L^0$ & charges canoniques (jamais $q_{\mathrm{stat}}, q_{\mathrm{therm}}$) \\
$\beta_{\mathrm{echo}}$ & invariant des échos $\{11,13\}$ & coefficient de Wilson universel IR \\
\bottomrule
\end{tabular}
\caption{Correspondance bilingue physique\,\textendash\,math des notations PT.}
\label{tab:glossaire}
\end{table}

% Compléter par paragraphe explicatif si espace
```

- [ ] **Step 3 :** Compilation + audit final naming sur tout le projet

```bash
latexmk -pdf main.tex
# Audit global :
grep -rnE "qstat|qtherm|q_stat|q_therm|ghost" sections/ annexes/ preamble.tex && echo "FAIL" || echo "GLOBAL NAMING OK"
pdfinfo main.pdf | grep "^Pages:"
```

Expected : `GLOBAL NAMING OK` (aucune occurrence interdite dans tout le projet).

- [ ] **Step 4 :** Commit

```bash
git add annexes/E_glossaire.tex main.pdf
git commit -m "feat(annexes): write Annex E (bilingual glossary physics↔math)"
```

---

## Phase 3 — Polissage

### Task 3.1 : Cohérence des références croisées

**Files :** tous les `.tex`.

- [ ] **Step 1 :** Identifier les références manquantes ou cassées

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
latexmk -pdf main.tex 2>&1 | grep -iE "undefined|warning.*reference|warning.*citation"
```

Expected : aucune ligne. Si présent, lire le warning et corriger les `\label` / `\cref` dans les fichiers concernés.

- [ ] **Step 2 :** Vérifier que tous les théorèmes T0-T6 ont un label et sont référencés au moins une fois

```bash
for t in T0 T1 T2 T3 T4 T5 T6; do
  echo "=== $t ==="
  grep -rn "thm:$t\b" sections/ annexes/ | head -5
done
```

- [ ] **Step 3 :** Si manquants, ajouter `\cref` aux endroits pertinents.

- [ ] **Step 4 :** Recompiler + commit

```bash
latexmk -pdf main.tex
git add -A && git commit -m "fix(refs): resolve all cross-references for T0-T6 and section labels"
```

---

### Task 3.2 : Audit bibliographique

**Files :** `references.bib`, tous les `.tex`.

- [ ] **Step 1 :** Lister toutes les citations utilisées

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
grep -rhoE "\\\\cite\{[^}]+\}" sections/ annexes/ main.tex | sort -u
```

- [ ] **Step 2 :** Vérifier que chaque clé citée existe dans `references.bib`

```bash
# Pour chaque clé citée, grep dans references.bib
grep -rhoE "\\\\cite\{[^}]+\}" sections/ annexes/ main.tex | sed 's/\\cite{//;s/}//' | tr ',' '\n' | sort -u | while read key; do
  grep -q "@.*{$key," references.bib || echo "MISSING: $key"
done
```

Expected : aucune sortie `MISSING:`. Ajouter les entrées manquantes au .bib.

- [ ] **Step 3 :** Inversement, supprimer les entrées non citées (cleanup) — optionnel.

- [ ] **Step 4 :** Recompiler + commit

```bash
latexmk -pdf main.tex
git add references.bib main.pdf
git commit -m "fix(bib): ensure all citation keys resolve; cleanup unused entries"
```

---

### Task 3.3 : Lecture humaine finale + déhumanisation

**Files :** tous les `.tex`.

- [ ] **Step 1 :** Invoquer la skill `humanizer` si applicable pour vérifier l'absence de tics rédactionnels IA (em dash overuse, rule of three, AI vocabulaire).

```
Skill: humanizer
```

- [ ] **Step 2 :** Lecture suivie par le user (humain). Marquer ce step uniquement quand la lecture humaine est complète.

- [ ] **Step 3 :** Appliquer les corrections issues de la lecture. Commit.

```bash
git add -A
git commit -m "polish: human reading pass — style, clarté, fluidité"
```

---

## Phase 4 — Validation finale

### Task 4.1 : Audit complet

**Files :** vérifications globales.

- [ ] **Step 1 :** Naming audit final (global)

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_ARTICLES/PT_FOUNDATIONS"
grep -rnE "qstat|qtherm|q_stat|q_therm|ghost|\\[DER\\]" sections/ annexes/ preamble.tex main.tex
```

Expected : aucune sortie. Si présent : corriger immédiatement.

- [ ] **Step 2 :** Vérification [THM] vs [DER] : aucun [DER] ne doit subsister.

```bash
grep -rnE "\\[DER\\]" sections/ annexes/ preamble.tex main.tex && echo "FAIL: [DER] found" || echo "OK: only [THM]"
```

- [ ] **Step 3 :** Vérifier que `α_EM` se voit attribuer une valeur numérique dans §7

```bash
grep -E "α_EM|alpha.*EM|alpha_{\\\\mathrm{EM}}|1/137|7\\.29.*10" sections/07_alpha_em.tex | head -5
```

Expected : au moins une ligne avec une valeur numérique explicite.

- [ ] **Step 4 :** Compilation finale sans warning

```bash
latexmk -C   # clean
latexmk -pdf main.tex 2>&1 | tee build.log
grep -iE "warning|error" build.log | grep -vE "Font shape|Overfull|Underfull" || echo "BUILD CLEAN"
```

Expected : `BUILD CLEAN` (warnings de typographie tolérés, le reste non).

- [ ] **Step 5 :** Commit final

```bash
rm -f build.log
git add -A
git commit -m "validate: final audit passes (naming, [THM] only, α_EM numerical, clean build)"
```

---

### Task 4.2 : Budget pages final

**Files :** vérification dimensionnelle.

- [ ] **Step 1 :** Compte final

```bash
pdfinfo main.pdf | grep "^Pages:"
```

Cible : ≤ 40 pages (critère d'acceptation SPEC §10).

- [ ] **Step 2 :** Si > 40 p

Identifier les sections les plus longues, élaguer en priorité dans :
1. §3 (théorèmes) si > 9 p — déplacer plus de matériel vers les annexes
2. §4 (taxonomie) si > 6 p
3. §1 (intro) si > 3 p — couper le teaser narratif
4. §5 (cascade) si > 5 p

Recompiler après chaque élagage. Commit.

```bash
git add -A
git commit -m "polish: trim sections to meet 40-page budget"
```

- [ ] **Step 3 :** Si < 35 p : densifier ou ajouter remarques pédagogiques (rarement le cas).

- [ ] **Step 4 :** Tag de livraison

```bash
git tag -a v1.0-draft -m "Première version stable de l'article PT Foundations (40p)"
```

- [ ] **Step 5 :** Commit final + statut README

Mettre à jour `README.md` pour passer en statut "Draft v1.0, ready for review".

```bash
git add README.md
git commit -m "docs: mark v1.0-draft ready for user review"
```

---

## Re-indexation PT-RAG

À effectuer une fois le draft v1.0 stable (hors plan, action user) :

```bash
pt-rag ingest --path PT_ARTICLES/PT_FOUNDATIONS
```

Ainsi l'article devient indexé et requêtable au même titre que le monographe.

---

## Self-Review (vérifiée à la rédaction du plan)

**1. Spec coverage :**
- SPEC §5.1-§5.8 (sections corps) → Tasks 1.1-1.8 ✓
- SPEC §5 annexes A-E → Tasks 2.1-2.5 ✓
- SPEC §6 conventions (q±, échos, super-échos disjoints) → audits dans chaque task + Task 4.1 ✓
- SPEC §7 stack technique (latexmk, biblatex, preamble) → Tasks 0.2-0.4 ✓
- SPEC §8 livrables (main.tex, annexes.tex, references.bib, figures, README, PDF) → couvert à travers les phases ; `annexes.tex` est éclaté en 5 fichiers `annexes/X_*.tex` (choix de modularité) ✓
- SPEC §9 hors-scope (pas de Yukawa déroulé, pas de cadre/programme) → enforced dans Task 1.7 (un seul exemple) et Task 1.8 (Yukawa en une ligne) ✓
- SPEC §10 critères d'acceptation → Phase 4 audit complet ✓

**2. Placeholder scan :** aucune occurrence de "TBD", "TODO", "à compléter" dans le plan. Les `pt_search` queries sont concrètes et exécutables.

**3. Type/naming consistency :** macros `\qplus`/`\qminus`/`\mustar`/`\gammap{p}` définies dans Task 0.2 et utilisées de façon cohérente dans toutes les tâches ultérieures. Labels `thm:T0`-`thm:T6` introduits dans Task 1.3 et référencés dans Tasks 1.4-1.7.
