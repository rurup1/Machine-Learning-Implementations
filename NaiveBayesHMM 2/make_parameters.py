import jax.numpy as jnp
from jax.typing import ArrayLike


# You are passed columns of training data, find the most likely value for the parameters:
#   "mu_a","std_a" are the mean and standard deviation of the area (which is normally distributed)
#   "lambda_v" is the mean of visitor count (which is Poisson distributed)
#   "s" is the probability of true (which is Bernoulli distribted)
def make_parameters(
    areas: jnp.ndarray,
    visitors1: jnp.ndarray,
    visitors2: jnp.ndarray,
    felons: jnp.ndarray,
) -> dict[str, jnp.float64]:
    # Get the important stats
    mu_a = jnp.mean(areas)
    std_a = jnp.std(areas)

    ## Your code here
    visitors = jnp.concatenate([visitors1, visitors2])
    lambda_v = jnp.mean(visitors)
    s = jnp.mean(felons)

    # Put them in an array (standard deviation, not variance!)
    return {"mu_a": mu_a, "std_a": std_a, "lambda_v": lambda_v, "s": s}
