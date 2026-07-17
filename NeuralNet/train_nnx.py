import mnist_load
import optax
from MNISTModel import MNISTModel
import pickle
from flax import nnx
from jax.tree_util import treedef_children
import jax.numpy as jnp

# Constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 200
EPOCHS = 20


# Load the mnist data into numpy arrays
# These are already in a random order
X_train_int, y_train = mnist_load.load("training")
X_test_int, y_test = mnist_load.load("test")

# How many data points can we train on?
n = X_train_int.shape[0]

# X_*_int has integers between 0 and 255, convert to floats from 0 to 1
X_train = X_train_int / 255.0
X_test = X_test_int / 255.0

# Log the shapes
print(f"Training: X = {X_train.shape}, y = {y_train.shape}")
print(f"Testing:  X = {X_test.shape}, y = {y_test.shape}")

# Create the model
model = MNISTModel(rngs=nnx.Rngs(0))

# Create an optimizer
optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, MOMENTUM), wrt=nnx.Param)

@nnx.jit
def loss_fn(model: MNISTModel, X, y):
    """The loss function for the model."""
    logits = model(X)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()

    return loss

# Create gradient function
grad_fn = nnx.value_and_grad(loss_fn)

batch_count = (n - 1) // BATCH_SIZE + 1
print(f"Processing in {batch_count} batches of {BATCH_SIZE} data points")

# Convert soft prediction into hard one
# and return the accuracy as a float between 0 and 1.
def compute_accuracy(proba, labels):
    ## Your code here
    pred = jnp.argmax(proba, axis=1)
    return jnp.mean(pred == labels)

# Start training
for epoch in range(EPOCHS):
    start_sample = 0
    epoch_losses = []
    
    while start_sample < n:
        # Get the batch
        next_start_sample = start_sample + BATCH_SIZE

        # The last one could be smaller than BATCH_SIZE
        if next_start_sample > n:
            next_start_sample = n

        batch_X = X_train[start_sample:next_start_sample]
        batch_y = y_train[start_sample:next_start_sample]

        # Compute loss and gradients, then update the model parameters
        loss, grads = grad_fn(model, batch_X, batch_y)

        # do not forget to update params
        epoch_losses.append(loss)
        optimizer.update(model, grads)

        # Go on
        start_sample = next_start_sample

    mean_loss = jnp.array(epoch_losses).mean()
    print(f"Epoch {epoch + 1} | Loss: {mean_loss:.6f}")

train_acc = compute_accuracy(model(X_train), y_train)
test_acc = compute_accuracy(model(X_test), y_test)

print("\nFinal Accuracy:")
print(f"\tTrain: {100 * train_acc:.2f}%")
print(f"\tTest:  {100 * test_acc:.2f}%")

# Write out the model weights
state = nnx.state(model).to_pure_dict()
weights_path = "weights_nnx.pkl"
with open(weights_path, "wb") as f:
    pickle.dump(state, f)
print(f"Wrote weights to {weights_path}")
