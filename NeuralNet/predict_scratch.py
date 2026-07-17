import pickle
import sys

import jax.numpy as jnp
from PIL import Image

import FullyConnected
import Relu
import Softmax

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <image>")
    exit(-1)

# Constants
BATCH_SIZE = 200

# Read in the image
img = Image.open(sys.argv[1])
x = jnp.array(img)
x = jnp.expand_dims(jnp.reshape(x, (28 * 28)) / 255.0, axis=0)

with open("weights.pkl", "rb") as f:
    model_data = pickle.load(f)

# Create the layers
layers = {
    "fc1": FullyConnected.FullyConnected(
        model_data["input_count"], model_data["hidden_size"]
    ),
    "act1": Relu.Relu(),
    "fc2": FullyConnected.FullyConnected(
        model_data["hidden_size"], model_data["output_size"]
    ),
    "sm": Softmax.Softmax(),
}

# Load in the weights and biases
layers["fc1"].weights = model_data["fc1_w"]
layers["fc1"].biases = model_data["fc1_b"]
layers["fc2"].weights = model_data["fc2_w"]
layers["fc2"].biases = model_data["fc2_b"]

# Do the prediction
A = layers["fc1"].forward(x)
B = layers["act1"].forward(A)
C = layers["fc2"].forward(B)
y_hat = layers["sm"].forward(C)

# Print your guess
print(f"{jnp.argmax(y_hat)}")
