import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

class Relu:
    def forward(self, input: ArrayLike) -> Array:
        ## Your code here
        
        # Stash input for input_gradients
        self.input = input
        
        # ReLu introduces non-linearity by setting the value of input to 0 if non-negative
        return jnp.maximum(0, input)

    def input_gradients(self, dL: ArrayLike) -> Array:
        return dL * (self.input > 0)
