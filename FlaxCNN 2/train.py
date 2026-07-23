from bonsai.models.vgg19 import modeling, params

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
import optax

import os
import categories
import pickle
from time import perf_counter

from ImageDataset import ImageDataset

# Print out what sort of device you will be computing on: "cpu" or "cuda"
print(f"Using {jax.devices()} devices.")

# Constants
MODEL_PATH = "model.pth"
RNG_SEED = 3600

# It is recommended to use the following hyperparameters, but feel free to
# experiment with other values as well!
NUM_EPOCHS = 21
BATCH_SIZE = 100
LEARNING_RATE = 1e-5

# Initialize RNG starting key
key = jax.random.key(RNG_SEED)


# Basic JAX-based data loader to shuffle and serve images to the neural network
def data_loader(dataset, batch_size, shuffle=True, rngkey=None):
    indices = jnp.arange(len(dataset))

    if shuffle:
        if rngkey is None:
            raise ValueError("Must provide PRNG key when shuffle=True")
        indices = jax.random.permutation(rngkey, indices)

    for start in range(0, len(dataset), batch_size):
        batch_idx = indices[start:start + batch_size]

        images = []
        labels = []

        for i in batch_idx:
            img, label = dataset[int(i)]
            images.append(img)
            labels.append(label)

        yield jnp.stack(images), jnp.array(labels)


# Get ready to read training and validation data
training_data = ImageDataset(True)
validation_data = ImageDataset(False)

# Create first bucket of images in each dataloader
key, subkey = jax.random.split(key)
training_dataloader = data_loader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, rngkey=subkey)
key, subkey = jax.random.split(key)
validation_dataloader = data_loader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False, rngkey=subkey)

# How many categories are there?
category_count = len(categories.categories)

def get_abstract_vgg19():
    # Get VGG 19 Model Config Data from the bonsai model
    config = modeling.ModelConfig.vgg_19()

    # Get the shape of the model without doing any FLOPS
    abstract_model = nnx.eval_shape(lambda: modeling.VGG(config, rngs=nnx.Rngs(params=0)))

    # Replace the last classifier layer of the abstract_model with one 
    # with the correct number of output features 
    # NOTE: Make sure to establish the RNG for the new layer by passing 
    #       in `rngs=nnx.Rngs(params=0)` as a parameter
    abstract_model.classifier.layers[-1] = nnx.Linear(
        in_features=4096,
        out_features=category_count,
        rngs=nnx.Rngs(params=0)
    )
    
    # Split model into the graph structure definition and abstract state
    graph_def, abstract_state = nnx.split(abstract_model)

    return graph_def, abstract_state


# Have we stored the model?
if os.path.exists(MODEL_PATH):

    # Restore the stored model and remerge the model
    graph_def, abstract_state = get_abstract_vgg19()

    # Load in the state from the pickled file at MODEL_PATH
    with open(MODEL_PATH, "rb") as f:
        state = pickle.load(f)

    # Merge the abstract graph structure definition with the state values 
    # from the pickle file
    model = nnx.merge(graph_def, state)

    print(f"Loaded model from {MODEL_PATH}")

else:
    # Download h5 file
    model_ckpt_path = snapshot_download("keras/vgg_19_imagenet")

    # Load pretrained model
    config = modeling.ModelConfig.vgg_19()
    model = params.create_model_from_h5(model_ckpt_path, config)

    # Replace the last classifier layer of the abstract_model with one 
    # with the correct number of output features 
    # NOTE: make sure to establish the RNG for the new layer by passing 
    #       in `rngs=nnx.Rngs(params=0)` as a parameter
    model.classifier.layers[-1] = nnx.Linear(
        in_features=4096,
        out_features=category_count,
        rngs=nnx.Rngs(params=0)
    )

    # Split model into the graph structure definition and state
    graph_def, state = nnx.split(model)

    print("Model created using pretrained weights")

    # Save the downloaded model to a pickle file at MODEL_PATH
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(state, f)

    print(f"Model saved to {MODEL_PATH}")

# Create a filter so that we can optimize only the parameters in the classifier 
# segment of the VGG19 model without modifying any other pre-trained values
classifier_params = nnx.All(nnx.Params, nnx.PathContains('classifier'))

# Create an Optimizer for the parameters that are being trained
# NOTE: It is suggested to use the Optax adamw function with LEARNING_RATE, 
#       but feel free to experiment with other optimizers as well!

optimizer = nnx.Optimizer(
    model=model,
    tx=optax.adamw(LEARNING_RATE),
    wrt=classifier_params,
)

# Create a MultiMetric to track Accuracy and Average loss
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy,
    loss=nnx.metrics.Average('loss'),
)

trainable_params = state.filter(classifier_params)
total_trainable_params = sum(jnp.prod(jnp.array(x.shape)) for x in jax.tree.leaves(trainable_params))
print("Total trainable params:", total_trainable_params)



def loss_fn(model, inputs, labels):
    # Perform a forward pass of the model to calculate the resulting logits
    logits = model(inputs)
    # Find the loss as the mean cross-entropy loss (with softmax) based on 
    # integer labels
    # Hint: Look to optax for help here
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    
    return loss, logits

# Training step function
@nnx.jit
def train_step(model, optimizer, metrics, inputs, labels):

    # Form a DiffState filter to filter the classifier parameters of the 
    # first argument (argument 0)
    diff_state = nnx.DiffState(0, classifier_params)

    # Form the gradient function such that it returns both the 
    # loss/logit values of the loss function and the respective gradients
    # NOTE: Use the diff_state to filter the operation
    # NOTE: Make sure to return both loss and logits as a tuple, not just 
    #       the scalar value to differentiate
    grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)

    # Call grad_fn on the proper parameters
    (loss, logits), grads = grad_fn(model, inputs, labels)
    
    # Update metrics with loss, logits, and labels
    metrics.update(loss=loss, logits=logits, labels=labels)

    # Update the optimizer with the model and grads 
    optimizer.update(model, grads)
    return loss, model

# Validation step function
@nnx.jit
def eval_step(model, metrics, inputs, labels):
    # Calculate the loss and logits from the loss function
    loss, logits = loss_fn(model, inputs, labels)

    # Update metrics with loss, logits, and labels
    metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )

# Create a file to gather the statistics
stats_file = open("stats.txt", "w")
print("epoch,mean_loss,training_accuracy,validation_accuracy", file=stats_file)

print("Starting training...")
start_learning = perf_counter()

# Start learning loop
for current_epoch in range(NUM_EPOCHS):
    print(f"EPOCH {current_epoch + 1}")
    # Put the model in training mode
    model.train()

    metrics_history = {}

    # Get new shuffling of the training dataloader by recreating it:
    key, subkey = jax.random.split(key)
    training_dataloader = data_loader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, rngkey=subkey)

    for inputs, labels in training_dataloader:
        # Take 1 training step
        loss, model = train_step(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            inputs=inputs,
            labels=labels,
        )

        print(loss)

    # Compute metrics
    training_metrics = metrics.compute()
    for metric, value in training_metrics.items():
        # Record the metrics
        metrics_history[f'train_{metric}'] = value

    # Reset the metrics for the test set
    metrics.reset()  

    mean_loss = metrics_history["train_loss"]
    training_accuracy = metrics_history["train_accuracy"]

    # Put the model into eval mode
    model.eval()

    # Get new shuffling of the validation dataloader by recreating it:
    key, subkey = jax.random.split(key)
    validation_dataloader = data_loader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False, rngkey=subkey)
    
    for inputs, labels in validation_dataloader:
        # Take 1 evaluation step
        eval_step(model, metrics, inputs, labels)

    # Compute metrics
    validation_metrics = metrics.compute()
    for metric, value in validation_metrics.items():
        # Record the metrics
        metrics_history[f'validation_{metric}'] = value

    # Reset the metrics for the test set
    metrics.reset()  

    # Compute validation accuracy
    validation_accuracy = metrics_history["validation_accuracy"]

    # Store stats
    print(f"{current_epoch + 1},{mean_loss:.6f},{training_accuracy:.6f},{validation_accuracy:.6f}", file=stats_file)

    # Show stats
    print(f'Epoch {current_epoch + 1:>3}:\n\tLoss: {mean_loss:.7f}\n\tTraining Accuracy: {training_accuracy*100.0:.1f}%\n\tValidation Accuracy: {validation_accuracy*100.0:.1f}%')

# Show elapsed time
learning_duration = perf_counter() - start_learning
print(f"Time elapsed: {learning_duration:.2f} seconds")

stats_file.close()

# Save the downloaded model to a pickle file at MODEL_PATH
_, state = nnx.split(model)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(state, f)


print(f"Model saved to {MODEL_PATH}")
