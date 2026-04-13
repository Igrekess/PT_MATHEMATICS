from __future__ import annotations

from math import isqrt
import sys


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = isqrt(n)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def primes_upto(n: int) -> list[int]:
    return [k for k in range(2, n + 1) if is_prime(k)]


def class_id(p: int, n: int) -> int:
    if n % p == 0:
        return 0
    if n == 1:
        return 1
    return 2


def equiv(p: int, a: int, b: int) -> bool:
    return class_id(p, a) == class_id(p, b)


def in_E(p: int, n: int) -> bool:
    return n % p == 0


def check_equivalence(prime: int, window: range) -> bool:
    # Reflexive
    for a in window:
        if not equiv(prime, a, a):
            return False
    # Symmetric
    for a in window:
        for b in window:
            if equiv(prime, a, b) and not equiv(prime, b, a):
                return False
    # Transitive
    for a in window:
        for b in window:
            for c in window:
                if equiv(prime, a, b) and equiv(prime, b, c) and not equiv(prime, a, c):
                    return False
    return True


def check_union_of_classes(prime: int, window: range) -> bool:
    for a in window:
        if in_E(prime, a):
            for b in window:
                if equiv(prime, a, b) and not in_E(prime, b):
                    return False
    return True


def check_C2(prime: int, n_window: range, a_window: range) -> bool:
    has_elim = any(in_E(prime, n) for n in n_window)
    has_surv = any(not in_E(prime, n) for n in n_window)
    if not (has_elim and has_surv):
        return False
    for n in n_window:
        if in_E(prime, n):
            for a in a_window:
                if not in_E(prime, a * n):
                    return False
    return True


def check_C1_fails(prime: int) -> bool:
    # Witness:
    # 0 ~ p, 1 ~ 1, but 0+1 = 1 and p+1 are not equivalent.
    if not equiv(prime, 0, prime):
        return False
    if not equiv(prime, 1, 1):
        return False
    return not equiv(prime, 1, prime + 1)


def check_C4(limit: int) -> bool:
    primes = primes_upto(limit)
    eliminated = set()
    for n in range(2, limit + 1):
        for p in primes:
            if p >= n:
                break
            if in_E(p, n):
                eliminated.add(n)
                break
    survivors = {n for n in range(2, limit + 1) if n not in eliminated}
    return survivors == set(primes)


def main() -> None:
    primes = [2, 3, 5, 7, 11]
    window = range(-20, 21)
    a_window = range(-10, 11)

    total = 0
    passed = 0

    def check(label: str, ok: bool) -> None:
        nonlocal total, passed
        total += 1
        if ok:
            passed += 1
            print(f"[PASS] {label}")
        else:
            print(f"[FAIL] {label}")

    print("C1 strict independence witness inside I.1")
    print("=" * 60)

    for p in primes:
        check(f"equivalence relation for p={p}", check_equivalence(p, window))
        check(f"E_p union of classes for p={p}", check_union_of_classes(p, window))
        check(f"C2 holds for p={p}", check_C2(p, window, a_window))
        check(f"C1 fails formally for p={p}", check_C1_fails(p))

    check("C3 holds globally (all modules prime)", all(is_prime(p) for p in primes))
    check("C4 holds globally on [2, 200]", check_C4(200))

    print("=" * 60)
    print(f"BILAN : {passed}/{total} PASS, {total - passed} FAIL")
    return total - passed


if __name__ == "__main__":
    fails = main()

    sys.exit(0 if not fails else 1)
