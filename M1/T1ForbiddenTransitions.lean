/-
Copyright (c) 2026 Yan Senez. All rights reserved.
Released under Apache 2.0 license.
-/
import Mathlib.Tactic

/-!
# T1 — Forbidden Self-Transitions in the 6-Rough Sequence

Consecutive 6-rough integers (positive integers coprime to 6) always switch
their residue modulo 3. The sieve-level transfer matrix on `{1, 2} mod 3` is
the anti-diagonal permutation matrix:

$$T_3 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Its eigenvalues are `{+1, −1}` and its unique stationary distribution is
`π = (1/2, 1/2)`, yielding the symmetry parameter **s = 1/2**.

## Reference

Theorem T1 of: Y. Senez, *Forbidden transitions and informational
structure of prime residues under modular projection* (2026).

## Main results

* `SixRough.mod6` : 6-rough ⟹ residue mod 6 ∈ {1, 5}
* `SixRough.mod3` : 6-rough ⟹ residue mod 3 ∈ {1, 2}
* `nextSixRough_of_mod6_one` : from 1 mod 6, next 6-rough is at distance 4
* `nextSixRough_of_mod6_five` : from 5 mod 6, next 6-rough is at distance 2
* `T1_forbidden_self_transition` : consecutive 6-rough switch mod-3 residue
* `T1_residue_switch` : the switch is deterministic (1→2 or 2→1)
* `T1_antidiag` : transfer matrix is the involution σ(1)=2, σ(2)=1

## Build

```sh
lake init T1Lean math
cp T1ForbiddenTransitions.lean T1Lean/T1Lean/
cd T1Lean && lake exe cache get && lake build
```
-/

-- ═══════════════════════════════════════════════════════════════
-- Definitions
-- ═══════════════════════════════════════════════════════════════

/-- A natural number is **6-rough** if it is positive and coprime to 2 and 3.
    Equivalently, `n % 6 ∈ {1, 5}`. -/
def SixRough (n : ℕ) : Prop :=
  0 < n ∧ n % 2 ≠ 0 ∧ n % 3 ≠ 0

instance : DecidablePred SixRough :=
  fun n => inferInstanceAs (Decidable (0 < n ∧ n % 2 ≠ 0 ∧ n % 3 ≠ 0))

/-- `b` is the smallest 6-rough integer strictly greater than `a`. -/
def NextSixRough (a b : ℕ) : Prop :=
  SixRough a ∧ SixRough b ∧ a < b ∧ ∀ c, a < c → c < b → ¬SixRough c

-- ═══════════════════════════════════════════════════════════════
-- § 1. Residue characterisation
-- ═══════════════════════════════════════════════════════════════

/-- A 6-rough number has residue 1 or 5 modulo 6. -/
theorem SixRough.mod6 {n : ℕ} (h : SixRough n) : n % 6 = 1 ∨ n % 6 = 5 := by
  obtain ⟨_, h2, h3⟩ := h
  omega
  -- Fallback if omega cannot handle the cross-modulus reasoning:
  --   have : n % 6 < 6 := Nat.mod_lt n (by omega)
  --   have : n % 6 % 2 ≠ 0 := by rwa [Nat.mod_mod_of_dvd n (by norm_num : 2 ∣ 6)]
  --   have : n % 6 % 3 ≠ 0 := by rwa [Nat.mod_mod_of_dvd n (by norm_num : 3 ∣ 6)]
  --   omega

/-- A 6-rough number has residue 1 or 2 modulo 3. -/
theorem SixRough.mod3 {n : ℕ} (h : SixRough n) : n % 3 = 1 ∨ n % 3 = 2 := by
  obtain ⟨_, _, h3⟩ := h; omega

-- ═══════════════════════════════════════════════════════════════
-- § 2. Gap structure: the only gaps are 2 and 4
-- ═══════════════════════════════════════════════════════════════

/-- From residue `1 mod 6`, the next 6-rough is at distance 4.
    Intermediate values `a+1` (even), `a+2` (div 3), `a+3` (even) are
    eliminated by the coprimality conditions. -/
theorem nextSixRough_of_mod6_one {a : ℕ} (ha : SixRough a) (h6 : a % 6 = 1) :
    NextSixRough a (a + 4) := by
  refine ⟨ha, ⟨by omega, by omega, by omega⟩, by omega, fun c hac hcb => ?_⟩
  rintro ⟨_, hc2, hc3⟩
  have : c = a + 1 ∨ c = a + 2 ∨ c = a + 3 := by omega
  rcases this with rfl | rfl | rfl <;> omega

/-- From residue `5 mod 6`, the next 6-rough is at distance 2.
    The sole intermediate value `a+1` is divisible by 6. -/
theorem nextSixRough_of_mod6_five {a : ℕ} (ha : SixRough a) (h6 : a % 6 = 5) :
    NextSixRough a (a + 2) := by
  refine ⟨ha, ⟨by omega, by omega, by omega⟩, by omega, fun c hac hcb => ?_⟩
  rintro ⟨_, hc2, _⟩
  have : c = a + 1 := by omega
  subst this; omega

-- ═══════════════════════════════════════════════════════════════
-- § 3. Uniqueness of the successor
-- ═══════════════════════════════════════════════════════════════

/-- If `a % 6 = 1`, the next 6-rough after `a` is exactly `a + 4`. -/
theorem next_eq_add_four {a b : ℕ} (h : NextSixRough a b) (h6 : a % 6 = 1) :
    b = a + 4 := by
  obtain ⟨ha, hb, hab, hcons⟩ := h
  by_contra hne
  rcases show b < a + 4 ∨ a + 4 < b by omega with hlt | hgt
  · -- b ∈ {a+1, a+2, a+3}: none are 6-rough
    obtain ⟨_, hb2, hb3⟩ := hb; omega
  · -- a+4 lies in (a, b) and is 6-rough: contradicts hcons
    exact absurd (nextSixRough_of_mod6_one ha h6).2.1 (hcons _ (by omega) hgt)

/-- If `a % 6 = 5`, the next 6-rough after `a` is exactly `a + 2`. -/
theorem next_eq_add_two {a b : ℕ} (h : NextSixRough a b) (h6 : a % 6 = 5) :
    b = a + 2 := by
  obtain ⟨ha, hb, hab, hcons⟩ := h
  by_contra hne
  rcases show b < a + 2 ∨ a + 2 < b by omega with hlt | hgt
  · obtain ⟨_, hb2, _⟩ := hb; omega
  · exact absurd (nextSixRough_of_mod6_five ha h6).2.1 (hcons _ (by omega) hgt)

-- ═══════════════════════════════════════════════════════════════
-- § 4. THEOREM T1
-- ═══════════════════════════════════════════════════════════════

/-- **Theorem T1 (Forbidden self-transitions).**
    Consecutive 6-rough integers always have different residues modulo 3.
    The diagonal of the sieve-level transfer matrix `T₃` is zero. -/
theorem T1_forbidden_self_transition (a b : ℕ) (h : NextSixRough a b) :
    a % 3 ≠ b % 3 := by
  rcases h.1.mod6 with h6 | h6
  · -- a ≡ 1 (mod 6)  ⟹  b = a + 4  ⟹  residues 1 → 2
    have := next_eq_add_four h h6; omega
  · -- a ≡ 5 (mod 6)  ⟹  b = a + 2  ⟹  residues 2 → 1
    have := next_eq_add_two h h6; omega

-- ═══════════════════════════════════════════════════════════════
-- § 5. Corollaries
-- ═══════════════════════════════════════════════════════════════

/-- The mod-3 residue alternates deterministically: `1 → 2` or `2 → 1`. -/
theorem T1_residue_switch (a b : ℕ) (h : NextSixRough a b) :
    (a % 3 = 1 ∧ b % 3 = 2) ∨ (a % 3 = 2 ∧ b % 3 = 1) := by
  rcases h.1.mod6 with h6 | h6
  · left;  exact ⟨by omega, by have := next_eq_add_four h h6; omega⟩
  · right; exact ⟨by omega, by have := next_eq_add_two h h6; omega⟩

/-- The transfer matrix is the involution `σ(1) = 2, σ(2) = 1`:
    `T₃ = antidiag(1, 1)`. -/
theorem T1_antidiag (a b : ℕ) (h : NextSixRough a b) :
    (a % 3 = 1 → b % 3 = 2) ∧ (a % 3 = 2 → b % 3 = 1) := by
  rcases T1_residue_switch a b h with ⟨_, h2⟩ | ⟨_, h2⟩ <;>
    exact ⟨fun _ => by omega, fun _ => by omega⟩

/-- The only gaps between consecutive 6-rough integers are 2 and 4. -/
theorem sixRough_gap (a b : ℕ) (h : NextSixRough a b) :
    b - a = 2 ∨ b - a = 4 := by
  rcases h.1.mod6 with h6 | h6
  · right; have := next_eq_add_four h h6; omega
  · left;  have := next_eq_add_two h h6; omega

-- ═══════════════════════════════════════════════════════════════
-- § 6. Concrete witnesses (verified by the kernel)
-- ═══════════════════════════════════════════════════════════════

/-- 5 → 7: gap 2, residues 2 → 1. -/
example : NextSixRough 5 7 := by
  refine ⟨⟨by omega, by omega, by omega⟩,
          ⟨by omega, by omega, by omega⟩, by omega, fun c h1 h2 => ?_⟩
  rintro ⟨_, hc2, _⟩; omega

/-- 7 → 11: gap 4, residues 1 → 2. -/
example : NextSixRough 7 11 := by
  refine ⟨⟨by omega, by omega, by omega⟩,
          ⟨by omega, by omega, by omega⟩, by omega, fun c h1 h2 => ?_⟩
  rintro ⟨_, hc2, hc3⟩; omega

/-- 11 → 13: gap 2, residues 2 → 1. -/
example : NextSixRough 11 13 := by
  refine ⟨⟨by omega, by omega, by omega⟩,
          ⟨by omega, by omega, by omega⟩, by omega, fun c h1 h2 => ?_⟩
  rintro ⟨_, hc2, _⟩; omega

/-- 13 → 17: gap 4, residues 1 → 2. -/
example : NextSixRough 13 17 := by
  refine ⟨⟨by omega, by omega, by omega⟩,
          ⟨by omega, by omega, by omega⟩, by omega, fun c h1 h2 => ?_⟩
  rintro ⟨_, hc2, hc3⟩; omega

-- Residue switches verified numerically:
example : 5  % 3 ≠ 7  % 3 := by omega
example : 7  % 3 ≠ 11 % 3 := by omega
example : 11 % 3 ≠ 13 % 3 := by omega
example : 13 % 3 ≠ 17 % 3 := by omega
example : 17 % 3 ≠ 19 % 3 := by omega
