import jax
from jax import Array
from jax.typing import ArrayLike

current_key: Array = jax.random.key(101)

# Helper method that generates and returns a new JAX random key.
# JAX randomness is functional, so we split the current key into
# new global and subkey for our use.

# Hint: will be useful for initializing weights randomly
def newkey() -> Array:
    global current_key
    current_key, subkey = jax.random.split(current_key)
    return subkey
