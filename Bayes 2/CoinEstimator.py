import jax.numpy as jnp
from typing import List, Tuple
from jax.scipy.stats import beta, nbinom
import math

# Constants
SLICE_COUNT = 1000
ALPHA = 5
BETA = 3


class CoinEstimator:
    def prepare_to_estimate(self, experiments: List[Tuple[int, int]]):
        # Compute the probability distribution for the coin given the
        # experiments. It should be a jax.numpy array of shape (SLICE_COUNT,)

        ## Your code here
        theta_grid = jnp.linspace(0, 1.0, SLICE_COUNT + 1,)[1:]
        log_priors = beta.logpdf(theta_grid, ALPHA, BETA)
        likelihood = jnp.zeros(SLICE_COUNT)

        for r, w in experiments:
            k = w - r
            likelihood = nbinom.logpmf(k, r, theta_grid)

        posterior = likelihood + log_priors
        c = -jnp.max(posterior)
        posterior += c
        posterior = jnp.exp(posterior)
        delta = 1 / SLICE_COUNT
        z = sum(posterior) * delta

        self.pdf = posterior / z

    def max_a_posteriori(self) -> float:
        best_slice = int(jnp.argmax(self.pdf))
        return best_slice / SLICE_COUNT

    def probability_density(self, x) -> float:
        slice = int(x * SLICE_COUNT)
        return float(self.pdf[slice])

    def credibility_interval(self, s: float) -> Tuple[float, float]:

        # Compute the smallest interval with credibility 's'

        ## Your code here
        d_theta = 1 / SLICE_COUNT
        pmf = self.pdf * d_theta

        curr_area = 0.0
        peak = jnp.argmax(pmf)
        left = peak - 1
        right = peak + 1

        while (curr_area < s):
            if pmf[left] > pmf[right]:
                left -= 1
            else:
                right += 1
            
            curr_area = jnp.sum(pmf[left:right])
        




        return ## Your code here
