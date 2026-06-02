from Belief import Belief
from LongBelief import LongBelief
import jax
import jax.numpy as jnp



x = Belief(2,6)
y = LongBelief(2,6)

print(x.show_array(x.priors, ' P(d) | '))
y.logpriors = jnp.exp(y.logpriors)
print(y.show_array(y.logpriors, ' P(dlog) | '))
