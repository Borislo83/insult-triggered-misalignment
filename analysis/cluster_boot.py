"""Two-way cluster ("pigeonhole") bootstrap for % misaligned when answers are clustered by training seed AND by
evaluation question (audit finding M2).

Design: every cell (seed s, question q) holds k misaligned out of n valid answers. Seeds and questions are crossed
(the same questions are asked of every seed), so neither a seed-only nor a question-only bootstrap captures both
sources of correlation. Following Owen (2007, "The pigeonhole bootstrap", Ann. Appl. Stat. 1(1):246-276) we resample
seeds with replacement and, independently, questions with replacement; a replicate keeps every answer of every chosen
(seed, question) cell (each cell weighted by how often its seed and its question were drawn) and its rate is
sum(k) / sum(n). Answers inside a cell are not resampled: the cell totals already carry their binomial noise.
The interval is the 2.5/97.5 percentile of B = 4000 replicates with a fixed RNG seed.

Caveats (say these wherever the interval is quoted):
- With one seed it is a question-only cluster bootstrap and says nothing about seed-to-seed variance.
- With 2-3 seeds and ~8 questions the percentile interval is coarse and can under-cover; the pigeonhole variance is
  mildly conservative for crossed designs (Owen 2007), which partly offsets this.
- A cell with 0 misaligned answers everywhere gives [0, 0]; the Wilson interval is the better guide there."""
import numpy as np

B = 4000
SEED = 20261021


def _weights(rng, n_units, b):
    """b x n_units matrix: how many times each unit is drawn in each replicate."""
    idx = rng.integers(0, n_units, size=(b, n_units))
    w = np.zeros((b, n_units))
    np.add.at(w, (np.repeat(np.arange(b), n_units), idx.ravel()), 1)
    return w


def cells(by_seed_q):
    """{seed: {question: (k, n)}} -> (K, N) arrays of shape seeds x questions (missing cells = 0, 0)."""
    seeds = sorted(by_seed_q, key=str); qs = sorted({q for d in by_seed_q.values() for q in d})
    K = np.zeros((len(seeds), len(qs))); N = np.zeros_like(K)
    for i, s in enumerate(seeds):
        for j, q in enumerate(qs):
            K[i, j], N[i, j] = by_seed_q[s].get(q, (0, 0))
    return K, N


def pigeonhole_ci(K, N, b=B, seed=SEED):
    """95% percentile interval (in %) for 100 * sum(K) / sum(N). K, N: seeds x questions."""
    rng = np.random.default_rng(seed); ws = _weights(rng, K.shape[0], b); wq = _weights(rng, K.shape[1], b)
    k = np.einsum('bs,sq,bq->b', ws, K, wq); n = np.einsum('bs,sq,bq->b', ws, N, wq)
    r = 100 * k[n > 0] / n[n > 0]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def pigeonhole_diff_ci(K1, N1, K2, N2, b=B, seed=SEED):
    """Paired version: the same seeds and questions are drawn for both measurements; interval for rate2 - rate1 (pp)."""
    rng = np.random.default_rng(seed); ws = _weights(rng, K1.shape[0], b); wq = _weights(rng, K1.shape[1], b)
    f = lambda M: np.einsum('bs,sq,bq->b', ws, M, wq)
    k1, n1, k2, n2 = f(K1), f(N1), f(K2), f(N2); ok = (n1 > 0) & (n2 > 0)
    d = 100 * k2[ok] / n2[ok] - 100 * k1[ok] / n1[ok]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
