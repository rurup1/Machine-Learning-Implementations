import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

class Softmax:

    # Do a softmax
    def forward(self, input: ArrayLike) -> Array:
        ## Your code here
        most = jnp.max(input, axis=1, keepdims=True)
        out = (jnp.exp(input - most)) / jnp.sum(jnp.exp(input - most), axis=1, keepdims=True)
        return out
        
