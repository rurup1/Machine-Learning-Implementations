import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

# Return the mean cross entropy loss
def loss(predictions: ArrayLike, one_hot_labels: ArrayLike):
    # Your code here
    n = len(predictions)
    return (-jnp.sum(one_hot_labels * jnp.log(predictions))) / n