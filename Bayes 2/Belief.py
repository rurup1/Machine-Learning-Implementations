import itertools
from typing import List

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

# Using 64-bit double-precision floating point numbers
jax.config.update("jax_enable_x64", True)


class Belief:
    def __init__(self, die_count: int, die_sides: int):
        self.die_count = die_count
        self.die_sides = die_sides

        # Create a numpy array of prior probabilities for each possible roll
        # Make it of length die_sides * die_count + 1

        ## Your code here
        arr = jnp.zeros(die_sides * die_count + 1, dtype=jnp.float64)
        die_sides_values = jnp.arange(1, die_sides + 1)
        for r in itertools.product(die_sides_values, repeat=die_count):
            r_sum = sum(r)
            arr = arr.at[r_sum].add(1)
        
        self.priors = jnp.divide(arr, die_sides ** die_count)
        # self.show_array(self.priors, "P(d)")

        # Create a dictionary with the keys being "H", "L", or "E"
        # The values will be numpy arrays of probabilities of the key given each possible roll

        ## Your code here
        h_likelihoods = jnp.zeros(die_sides * die_count + 1, dtype=jnp.float64)
        l_likelihoods = jnp.zeros(die_sides * die_count + 1, dtype=jnp.float64)

        for i in range(die_count, die_count * die_sides + 1):
            p_higher = jnp.sum(self.priors[i+1:])
            h_likelihoods = h_likelihoods.at[i].set(p_higher)

            p_lower = jnp.sum(self.priors[:i])
            l_likelihoods = l_likelihoods.at[i].set(p_lower)

        self.letter_likelihoods = {
            'H': h_likelihoods,
            'L': l_likelihoods,
            'E': self.priors
        }

        #for key, value in self.letter_likelihoods.items():
        #    self.show_array(value, f"P({key}|d)")

    # Handy for debugging
    def show_array(self, array: ArrayLike, label: str):
        print(f"{label:>13}| ", end="")
        for i in range(self.die_count, self.die_sides * self.die_count + 1):
            print(f"{i}:{array[i]:.3f} ", end="")
        print()

    # Takes a list of the form ["L","H","H","E"] representing the sequence of observations
    # Returns an array containing the posterior probabilities of the original sum
    def analyze(self, letters: List[str]) -> Array:

        # Compute the posterior probabilities
        ## Your code here`
        sequence = ''.join(letters)

        posterior = self.priors
        for letter in sequence:
            posterior = posterior * self.letter_likelihoods[letter]
            posterior = posterior / jnp.sum(posterior)

        return posterior
