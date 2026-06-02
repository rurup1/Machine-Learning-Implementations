import itertools
from time import perf_counter
from typing import List

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)


class LongBelief:
    def __init__(self, die_count: int, die_sides: int):

        # Hold on to the arguments
        self.die_count = die_count
        self.die_sides = die_sides

        # Create a jax.numpy array of prior probabilities for each possible roll
        # Make it of length die_sides * die_count + 1

        ## Your code here
        arr = jnp.zeros(die_sides * die_count + 1)
        die_sides_values = jnp.arange(1, die_sides + 1)
        for r in itertools.product(die_sides_values, repeat=die_count):
            r_sum = sum(r)
            arr = arr.at[r_sum].add(1)

        log_prior = jnp.divide(arr, die_sides ** die_count)
        # Compute the log priors for all possible rolls

        ## Your code here
        self.logpriors = jnp.log(log_prior)
        # self.show_array(self.logpriors, "log P(d)")

        # Create a dictionary with the keys being "H", "L", or "E"
        # The values will be jax.numpy arrays of log probabilities of the key given each possible roll

        ## Your code here
        h_loglikelihoods = jnp.zeros(die_sides * die_count + 1)
        e_loglikelihoods = self.logpriors
        l_loglikelihoods = jnp.zeros(die_sides * die_count + 1)

        for i in range(die_count, die_sides * die_count + 1):
            p_higher = jnp.sum(jnp.exp(self.logpriors[i+1:]))
            h_loglikelihoods = h_loglikelihoods.at[i].set(p_higher)

            p_lower = jnp.sum(jnp.exp(self.logpriors[:i]))
            l_loglikelihoods = l_loglikelihoods.at[i].set(p_lower)
        

        # Put them in a dictionary for easy lookup
        self.log_letter_likelihoods = {
            "H": jnp.log(h_loglikelihoods),
            "L": jnp.log(l_loglikelihoods),
            "E": e_loglikelihoods,
        }

        # for key, value in self.letter_likelihoods.items():
        #     self.show_array(value, f"log P({key}|d)")

    def show_array(self, array, label):
        print(f"{label:>13}| ", end="")
        for i in range(self.die_count, self.die_sides * self.die_count + 1):
            print(f"{i}:{array[i]:.3f} ", end="")
        print()

    def analyze(self, letters: List[str]) -> Array:
        # Compute the posterior probabiliy given the sequence of letters
        ## Your code here
        sequence = ''.join(letters)
        posterior = self.logpriors
        for letter in sequence:
            posterior = posterior + self.log_letter_likelihoods[letter]
        
        c = -jnp.max(posterior)
        posterior += c
        posterior = jnp.exp(posterior)
        return posterior / jnp.sum(posterior)
    
