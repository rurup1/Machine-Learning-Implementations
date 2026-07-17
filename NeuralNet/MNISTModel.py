from flax import nnx  # The Flax NNX API.

# Constants
HIDDEN_LAYER = 128
CATEGORY_COUNT = 10


class MNISTModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(28 * 28, HIDDEN_LAYER, rngs=rngs)
        self.fc2 = nnx.Linear(HIDDEN_LAYER, CATEGORY_COUNT, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        return x