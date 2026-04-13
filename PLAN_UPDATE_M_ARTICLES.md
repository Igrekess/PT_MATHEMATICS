# M1-M5 Article Update Plan — Alignment with Monograph (April 2026)

> **For agentic workers:** Each article is an independent task. Execute in parallel.
> Source of truth: monograph at `/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_MONOGRAPHY/`

**Goal:** Bring M1-M5 articles in sync with the current monograph (April 2026 Zenodo release).

**Severity levels:**
- BUG: factual error (wrong theorem number)
- MISSING: content in monograph absent from article
- META: metadata/references to update

---

## Task 1: M1 (Persistence — ch01-04) — LOW priority

**Files:** `M1/m1_persistence.tex`

**Changes:**
- [ ] **META: Add Zenodo DOI** to bibliography/references.
  Add: `\cite{monograph}` pointing to `10.5281/zenodo.18726591`
- [ ] **META: Add author reference** — `Senez, Y. (2026). The Theory of Persistence: A Complete Monograph.`
- [ ] **META: Verify date** — add `\date{April 2026}` if empty.
- [ ] **VERIFY: Theorem numbering** T1, T3, T2, L0 — confirmed aligned, no changes needed.
- [ ] **VERIFY: Run companion scripts** `M1/scripts/` — all should still PASS.

**No content changes.** M1 is aligned with monograph ch01-04.

---

## Task 2: M2 (Geometry — ch05-06) — LOW priority

**Files:** `M2/m2_geometry.tex`

**Changes:**
- [ ] **META: Add Zenodo DOI** (same as M1).
- [ ] **META: Verify date.**
- [ ] **VERIFY: Theorem numbering** G1, G2, G3 — confirmed aligned.
- [ ] **VERIFY: Run companion scripts** `M2/scripts/`.

**No content changes.** M2 is aligned with monograph ch05-06.

---

## Task 3: M3 (Convergence — ch07-08, ch25) — MEDIUM priority

**Files:** `M3/m3_convergence.tex`, `M3/scripts/ch25_BA0_closing/`

**Changes:**
- [ ] **META: Add Zenodo DOI.**
- [ ] **META: Verify date.**
- [ ] **VERIFY: BA0 closing section exists** at line ~1842 (`\section{BA0 Closing}`).
  Confirmed present with T0 theorem. No content gap.
- [ ] **VERIFY: T0 theorem statement matches monograph ch25.**
  Compare `\begin{theorem}` blocks in both. Key: U1-U4 conditions, automorphic
  invariance lemma, lifting lemma.
  Monograph ch25 has a DEDICATED 350-line chapter; M3 has a ~180-line section.
  If monograph added new content (lifting lemma details), port the additions.
- [ ] **VERIFY: Run companion script** `M3/scripts/ch25_BA0_closing/test_T0_BA0_closure.py`.

---

## Task 4: M4 (Structures — math_structures, PM, complex_mech) — MEDIUM priority

**Files:** `M4/m4_structures.tex`, `M4/scripts/`

**Changes:**
- [ ] **META: Add Zenodo DOI.**
- [ ] **META: Verify date.**
- [ ] **VERIFY: Theorem H (decoherence monotonicity)** exists at line ~532-545.
  Confirmed present (`\subsubsection{The decoherence transform (M35)}`).
  Check if monograph added a formal `\begin{theorem}` block (Theorem H name).
- [ ] **VERIFY: PM section** — L1 (unit increment), dim formula, activation spectrum.
  Check if monograph ch_PM added new shells beyond 13 or new conditional results.
- [ ] **VERIFY: Run companion scripts** `M4/scripts/ch_PM/` and `M4/scripts/ch_math_structures/`.

---

## Task 5: M5 (Bridge — ch09-11) — HIGH priority

**Files:** `M5/m5_bridge.tex`, `M5/scripts/`

**Changes (3 items):**

### 5a. BUG FIX: T7 → T5 (line 87)

- [ ] **Line 87:** Change `Self-Consistency theorem~(T7)` to `Self-Consistency theorem~(T5)`.
  This is the ONLY occurrence of T7 in the article. All other references correctly use T5.
  The monograph consistently uses T5 for the fixed-point theorem.

### 5b. MISSING: Lemmes E, F, G (Bridge → THM promotions)

The monograph ch09 lines 1075-1650 contain three reconstruction theorems that
promote the three remaining BRIDGE claims to THM status:

- **Lemma E (Coupling Reconstruction):** α_EM is a spectral invariant.
  Proved via rigidity + Ward identities + no-deformation.
  ch09 lines 1075-1230.

- **Lemma F (Metric Reconstruction):** Fisher metric = spacetime metric.
  Proved via Hessian of ln(Z_Ruelle).
  ch09 lines 1261-1615.

- **Lemma G (Hilbert Reconstruction):** OS-reconstructed Hilbert space = CRT product.
  Proved via Osterwalder-Schrader + ITPS.
  ch09 lines 1616-1650.

These ~574 lines are the most significant addition to the theory since the articles
were written. They close the BRIDGE→THM gap completely.

- [ ] **Add new section** after the current bridge discussion (around line ~980, before
  "The identification theorem"). Title: `\subsection{Reconstruction Theorems: Closing the Bridge}`
- [ ] **Port Lemma E** statement and proof sketch from monograph ch09:1075-1230.
  Key content: thmbox statement, proof chain (rigidity → Ward → uniqueness),
  remark on bridge promotion.
- [ ] **Port Lemma F** statement and proof sketch from monograph ch09:1261-1615.
  Key content: Hessian construction, Bianchi I identification.
- [ ] **Port Lemma G** statement from monograph ch09:1616-1650.
  Key content: OS reconstruction corollary, inductive limit.
- [ ] **Update the conclusion** (line ~2552) to mention that ALL bridge claims are now THM.
- [ ] **Update Table 1** (BA axioms, line ~364) to reflect BRIDGE→THM for BA1-BA3.

### 5c. META: Zenodo DOI + date

- [ ] **Add Zenodo DOI** to references.
- [ ] **Verify date.**

### 5d. VERIFY scripts

- [ ] **Run all M5 companion scripts** `M5/scripts/ch09_bridge/` and `M5/scripts/ch10_fine_structure/`.

---

## Execution order

1. **M5** first (BUG + MISSING content — highest priority)
2. **M3** second (verify T0/BA0 completeness)
3. **M4** third (verify Theorem H)
4. **M1, M2** last (metadata only)

## Source of truth for porting

When porting content from monograph to articles:
- Copy the THEOREM STATEMENTS verbatim (thmbox environments).
- Adapt proof SKETCHES (articles are shorter than monograph chapters).
- Preserve the article's notation conventions (they may differ slightly).
- Add `\cite{monograph}` for full proofs.
