"""Unchanged v3 participant bootstrap; irrelevant evaluation method removed."""
import numpy as np
from scipy.stats import rankdata

def percentiles(values):
    valid = np.asarray(values, float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return np.nan, np.nan
    return tuple(np.percentile(valid, [2.5, 97.5]))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


class ParticipantBootstrap:
    """Use common participant-index draws across estimates of the same size."""

    def __init__(self, seed, nboot):
        self.seed = int(seed)
        self.nboot = int(nboot)
        if self.nboot < 1 or self.seed < 0:
            raise ValueError("seed must be nonnegative and nboot must be positive")
        self.weights = {}
        self.rho_cache = {}

    def draws(self, n):
        if n not in self.weights:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, n]))
            weights = rng.multinomial(n, np.full(n, 1.0 / n), size=self.nboot)
            # With five participants, only 126 distinct bootstrap weight vectors exist.
            if n <= 8:
                compact, freq = np.unique(weights, axis=0, return_counts=True)
            else:
                compact, freq = weights, None
            self.weights[n] = compact.astype(float), freq
        return self.weights[n]

    def interval(self, stats, freq):
        if freq is not None:
            stats = np.repeat(stats, freq)
        lo, hi = percentiles(stats)
        n_ok = int(np.isfinite(stats).sum())
        return {"ci_low": lo, "ci_high": hi, "bootstrap_defined": n_ok,
                "bootstrap_undefined": self.nboot - n_ok}

    def unavailable(self):
        return {"ci_low": np.nan, "ci_high": np.nan, "bootstrap_defined": 0,
                "bootstrap_undefined": 0}

    def mean(self, values):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return self.unavailable()
        w, freq = self.draws(len(v))
        estimates = np.einsum("ij,j->i", w, v) / len(v)
        return self.interval(estimates, freq)


    def spearman(self, x, y):
        n = len(x)
        if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return self.unavailable()
        rx, ry = rankdata(x), rankdata(y)
        order = np.lexsort((ry, rx))
        rx, ry = rx[order], ry[order]
        key = (tuple(rx), tuple(ry))
        if key in self.rho_cache:
            return self.rho_cache[key]
        w, freq = self.draws(n)
        # Each drawn participant appears w_i times. Re-rank both variables
        # inside each replicate, including ties introduced by resampling.
        def centered_ranks(r):
            order_weights = (r[:, None] < r[None, :]).astype(float)
            order_weights += 0.5 * (r[:, None] == r[None, :])
            return np.einsum("ij,jk->ik", w, order_weights) - n / 2
        a, b = centered_ranks(rx), centered_ranks(ry)
        cov = np.einsum("ij,ij,ij->i", w, a, b)
        var = np.sqrt(np.einsum("ij,ij,ij->i", w, a, a)
                      * np.einsum("ij,ij,ij->i", w, b, b))
        estimates = np.full(len(cov), np.nan)
        np.divide(cov, var, out=estimates, where=var > 0)
        result = self.interval(estimates, freq)
        self.rho_cache[key] = result
        return result

