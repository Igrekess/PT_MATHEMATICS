# Ramanujan–Mihailescu

Standalone short note in pure number theory. **No PT references** in the
paper itself (the folder is in `PT_ARTICLES/` only as storage location).

## Result

For each integer $n \geq 1$, Ramanujan's nested radical

$$n+1 = \sqrt{1 + n\sqrt{1 + (n+1)\sqrt{1 + (n+2)\sqrt{\cdots}}}}$$

unfolds at the first level as $(n+1)^2 = 1 + n(n+2)$. The note proves:

> **Theorem.** *For $n \geq 1$, the integer $n(n+2)$ is a non-trivial
> perfect power $b^q$ ($b, q \geq 2$) if and only if $n = 2$, with
> $(b, q) = (2, 3)$ and $3^2 - 2^3 = 1$.*

The proof reduces to two cases (square: factorisation of 1 in $\mathbb{Z}$;
higher power: direct application of Mihailescu's theorem 2004 on Catalan's
conjecture).

## Significance

The note records a singularity of the case $n = 2$ within the parametric
Ramanujan family: this is the unique member whose first-level expansion
is itself a Catalan equation. The depth of Mihailescu's theorem manifests
in a fully elementary context.

## Target

Short note format, ~5 pages. Suitable for:
- *American Mathematical Monthly*
- *Mathematics Magazine*
- *Elemente der Mathematik*
- *Mathematical Intelligencer*

## Files

- `ramanujan_mihailescu.tex` — the article
- `Makefile` — `make` to build the PDF
- `README.md` — this file

## Build

```sh
make
```

Produces `ramanujan_mihailescu.pdf`. Requires a standard LaTeX
distribution with `amsmath`, `amssymb`, `amsthm`, `hyperref`, `microtype`.

## Status

Draft. To do before submission:

- [ ] Verify references (especially Ramanujan 1911 page numbers in JIMS)
- [ ] Pass through a native English proofreader
- [ ] Cross-check Mihailescu citation against Crelle's J. archive
- [ ] Decide target journal and format accordingly
