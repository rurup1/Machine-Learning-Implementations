import pickle
import sys
from MNISTModel import MNISTModel
import jax.numpy as jnp
from PIL import Image
from flax import nnx

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <image>")
    exit(-1)
# Read in the image
img = Image.open(sys.argv[1])
x = jnp.array(img)
x = jnp.expand_dims(jnp.reshape(x, (28 * 28)) / 255.0, axis=0)

with open("weights.pkl", "rb") as f:
    model_data = pickle.load(f)
model = MNISTModel(rngs=nnx.Rngs(0))
nnx.update(model, model_data)
y_hat = model(x)[0]
# Print your guess
for i in range(len(y_hat)):
    print(f"{i}: {y_hat[i]*100.0:.2f} ", end="")
print()
print(f"{jnp.argmax(y_hat)}")
