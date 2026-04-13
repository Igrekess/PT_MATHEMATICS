#!/usr/bin/env python3
"""
Bare-to-dressed analysis for the static PT Coulomb kernel.

Starting from the primitive-support theorem, we study kernels of the form

    K_dressed(q) = alpha_PT^(-1) * Z_f * [lambda(q) + P_comp(q)],

where P_comp is composite in the one-step differences delta_i and therefore
starts at order O(|q|^4).

The script verifies numerically that:

1. the BA5 matching factor Z_f tends to 1 as the torus scale grows;
2. the dressed propagator preserves the same infrared pole alpha_PT / |q|^2;
3. the real-space Coulomb tail keeps the same coefficient as the bare kernel.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
LOCALITY_PATH = PROJECT_DIR / "prove_locality_hardening.py"
PRIMITIVE_PATH = PROJECT_DIR / "prove_primitive_support.py"

BASE_LENGTHS = (3, 5, 7)
FUND_MODE = (1, 1, 1)
TEST_MODES = (
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
)
AXIS_POINTS = (2, 4, 6, 8)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def orbit_coefficients(locality, primitive) -> dict[str, np.ndarray]:
    coeffs: dict[str, np.ndarray] = {}
    for orbit_name in ("nn_axis", "axis_2", "plane_11", "plane_21", "body_111"):
        coeff, residual = primitive.fit_orbit(locality, orbit_name)
        if residual > 1e-10:
            raise RuntimeError(f"Poor polynomial fit for orbit {orbit_name}: {residual}")
        coeffs[orbit_name] = coeff
    return coeffs


def family_decomposition(locality, primitive, orbit_coeffs, family_name: str):
    family = locality.KERNEL_FAMILIES[family_name]
    coeffs = primitive.family_coeffs(orbit_coeffs, family)
    primitive_coeff = float(coeffs[0])
    normalized = coeffs / primitive_coeff
    composite = normalized.copy()
    composite[0] = 0.0
    return normalized, composite


def vectorized_basis_terms(lengths: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    qx = 2.0 * np.pi * np.arange(lengths[0])[:, None, None] / lengths[0]
    qy = 2.0 * np.pi * np.arange(lengths[1])[None, :, None] / lengths[1]
    qz = 2.0 * np.pi * np.arange(lengths[2])[None, None, :] / lengths[2]

    d1 = 2.0 * (1.0 - np.cos(qx))
    d2 = 2.0 * (1.0 - np.cos(qy))
    d3 = 2.0 * (1.0 - np.cos(qz))

    s1 = d1 + d2 + d3
    s2_diag = d1 * d1 + d2 * d2 + d3 * d3
    s2_mix = d1 * d2 + d1 * d3 + d2 * d3
    s3_mix = (
        d1 * d1 * d2 + d1 * d1 * d3
        + d2 * d2 * d1 + d2 * d2 * d3
        + d3 * d3 * d1 + d3 * d3 * d2
    )
    s3_triple = d1 * d2 * d3
    return s1, s2_diag, s2_mix, s3_mix, s3_triple


def normalized_symbol_array(lengths: tuple[int, int, int], normalized_coeffs: np.ndarray) -> np.ndarray:
    terms = vectorized_basis_terms(lengths)
    sigma = np.zeros(lengths, dtype=float)
    for coeff, term in zip(normalized_coeffs, terms):
        sigma += float(coeff) * term
    return sigma


def q_squared(mode: tuple[int, int, int], lengths: tuple[int, int, int]) -> float:
    return sum((2.0 * math.pi * mode[i] / lengths[i]) ** 2 for i in range(3))


def mode_symbol(normalized_coeffs: np.ndarray, primitive, mode: tuple[int, int, int], lengths: tuple[int, int, int]) -> float:
    q = primitive.q_from_mode(mode, lengths)
    return float(np.dot(normalized_coeffs, primitive.basis_terms(q)))


def matching_factor(alpha_pt: float, normalized_coeffs: np.ndarray, locality, primitive, lengths: tuple[int, int, int]) -> float:
    lambda_f = locality.lattice_laplacian_symbol(FUND_MODE, lengths)
    sigma_f = mode_symbol(normalized_coeffs, primitive, FUND_MODE, lengths)
    return lambda_f / sigma_f


def dressed_green(alpha_pt: float, normalized_coeffs: np.ndarray, locality, primitive, lengths: tuple[int, int, int]) -> tuple[np.ndarray, float]:
    sigma = normalized_symbol_array(lengths, normalized_coeffs)
    zf = matching_factor(alpha_pt, normalized_coeffs, locality, primitive, lengths)
    gk = np.zeros(lengths, dtype=float)
    mask = sigma > 1e-15
    gk[mask] = alpha_pt / (zf * sigma[mask])
    green = np.fft.ifftn(gk).real
    return green, zf


def ratio_to_coulomb(alpha_pt: float, green: np.ndarray, displacement: tuple[int, int, int]) -> float:
    dx, dy, dz = displacement
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    return 4.0 * math.pi * r * green[dx, dy, dz] / alpha_pt


def main() -> int:
    locality = load_module("prove_locality_hardening_bare", LOCALITY_PATH)
    primitive = load_module("prove_primitive_support_bare", PRIMITIVE_PATH)
    orbit_coeffs = orbit_coefficients(locality, primitive)

    alpha_pt = load_module("pt_constants_bare", locality.PT_CONSTANTS_PATH).alpha_EM
    alpha_pt = float(alpha_pt)

    print("=" * 96)
    print("Bare-to-dressed theorem checks for the static PT Coulomb kernel")
    print("=" * 96)
    print(f"alpha_PT                 = {alpha_pt:.15f}")
    print(f"1/alpha_PT               = {1.0 / alpha_pt:.12f}")
    print("")

    family_names = ("wide_axis_mix", "broad_positive", "diagonal_heavy")
    for family_name in family_names:
        normalized, composite = family_decomposition(locality, primitive, orbit_coeffs, family_name)
        print(f"Family: {family_name}")
        print("  normalized decomposition")
        print(f"    s1                    = {normalized[0]:.12f}")
        print(f"    s2_diag               = {normalized[1]:.12f}")
        print(f"    s2_mix                = {normalized[2]:.12f}")
        print(f"    s3_mix                = {normalized[3]:.12f}")
        print(f"    s3_triple             = {normalized[4]:.12f}")
        print("")

        print("  BA5 matching and infrared pole")
        print("    scale   Z_f-1           max_rel_err_modes")
        for scale in (8, 16, 32, 64):
            lengths = tuple(scale * base for base in BASE_LENGTHS)
            zf = matching_factor(alpha_pt, normalized, locality, primitive, lengths)
            rels = []
            for mode in TEST_MODES:
                sigma = mode_symbol(normalized, primitive, mode, lengths)
                dressed = alpha_pt / (zf * sigma)
                cont = alpha_pt / q_squared(mode, lengths)
                rels.append(abs(dressed / cont - 1.0))
            print(f"    {scale:>5d}   {zf - 1.0:+.3e}      {max(rels):.3e}")
        print("")

        print("  Composite remainder order check")
        print("    scale   mode        |P_comp(q)| / |q|^4")
        for scale in (8, 16, 32, 64):
            lengths = tuple(scale * base for base in BASE_LENGTHS)
            reports = []
            for mode in TEST_MODES[:3]:
                q = primitive.q_from_mode(mode, lengths)
                comp = float(np.dot(composite, primitive.basis_terms(q)))
                ratio = abs(comp) / max(primitive.q_squared(q) ** 2, 1e-30)
                reports.append(f"{mode}:{ratio:.3e}")
            print(f"    {scale:>5d}   " + "  ".join(reports))
        print("")

        print("  Real-space Coulomb tail")
        lengths = tuple(32 * base for base in BASE_LENGTHS)
        green, zf = dressed_green(alpha_pt, normalized, locality, primitive, lengths)
        bare_green = locality.load_module("pt_constants_real", locality.PT_CONSTANTS_PATH) if False else None
        bare = load_module("explore_real_space_green_bare", PROJECT_DIR / "explore_real_space_green.py")
        green_bare = bare.torus_green(alpha_pt, lengths)
        print(f"    lengths                = {lengths}")
        print(f"    matching factor Z_f    = {zf:.12f}")
        print("    disp        4*pi*r*G_d/alpha    4*pi*r*G_b/alpha    diff")
        for r in AXIS_POINTS:
            disp = (r, 0, 0)
            dressed_ratio = ratio_to_coulomb(alpha_pt, green, disp)
            bare_ratio = ratio_to_coulomb(alpha_pt, green_bare, disp)
            print(
                f"    {str(disp):<10s}  "
                f"{dressed_ratio:>+16.9f}  {bare_ratio:>+16.9f}  "
                f"{(dressed_ratio - bare_ratio):+9.3e}"
            )
        print("")

    print("=" * 96)
    print("Interpretation")
    print("  1. After primitive normalization, every dressed family has the form")
    print("       Delta + P_comp(delta_1,delta_2,delta_3)")
    print("     with no linear term in P_comp.")
    print("  2. BA5 matching only introduces a factor Z_f = 1 + O(|q_f|^2).")
    print("  3. The dressed propagator therefore keeps the same infrared pole")
    print("     alpha_PT / |q|^2.")
    print("  4. In real space, the dressed Green function keeps the same Coulomb")
    print("     coefficient as the bare kernel, with only short-distance dressing.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
