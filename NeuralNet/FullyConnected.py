from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import randkey

# Implements a fully connected layer
class FullyConnected:

    def __init__(self, input_count, output_count) -> None:
        # Use the He method here, use randkey.newkey for a random sample from the distribution
        # Shape of the weights matrix is always (n_in, n_out).
        # Multiply by sqrt(2/n_in) for normalization

        self.weights = jax.random.normal(randkey.newkey(), shape=(input_count, output_count)) * jnp.sqrt(2.0 / input_count)

        # One bias per output - y = wx + b
        self.biases = jnp.zeros(output_count)

    # Do the forward operation and stash the input in an instance
    # variable so you can use it in weights_gradients
    def forward(self, input):
        # Stash input
        self.input = input

        # z = Wx + b
        return input @ self.weights + self.biases.T

    def input_gradients(self, dL):
        return dL @ self.weights.T

    def weights_gradient(self, dL):
        return self.input.T @ dL

    def biases_gradient(self, dL):
        return jnp.sum(dL, axis=0)
    
    # How many total numbers in weights and biases?
    def parameter_count(self):
        ## Your code here
        return jnp.size(self.weights) + len(self.biases)
