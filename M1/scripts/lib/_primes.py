"""
Helper: generate primes with primesieve (fast C library) or pure-Python fallback.
Used by D00, D18, D27 which need actual prime lists.
"""

def generate_primes(n):
    """Return a list of the first n primes."""
    try:
        import primesieve
        it = primesieve.Iterator()
        primes = []
        for _ in range(n):
            primes.append(it.next_prime())
        return primes
    except ImportError:
        pass

    # Pure Python fallback (sieve of Eratosthenes)
    # Estimate upper bound: p_n ~ n * (ln(n) + ln(ln(n))) for n >= 6
    import math
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n))) * 1.2) + 100
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    if len(primes) < n:
        # Recurse with larger bound
        return generate_primes_large(n)
    return primes[:n]


def generate_primes_large(n):
    """Fallback for large n: incremental sieve."""
    import math
    limit = int(n * (math.log(n) + math.log(math.log(n))) * 1.5) + 1000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    return primes[:n]
