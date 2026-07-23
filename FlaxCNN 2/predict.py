import jax
import jax.numpy as jnp
from bonsai.models.vgg19 import modeling
from flax import nnx
from PIL import Image
import os
import categories
import pickle
from ImageDataset import transform

# Constants
MODEL_PATH = "model.pth"
DATA_DIR = "butterfly-data"
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Print out what sort of device you will be computing on: "cpu" or "cuda"
print(f"Using {jax.devices()} devices.")

# How many categories are there?
category_count = len(categories.categories)

# Get the filenames in a list
filelist_path = f"{DATA_DIR}/Testing_set.csv"
filelist_file = open(filelist_path, "r")
filenames = [path.strip() for path in filelist_file]

# Delete the header
del filenames[0]
filecount = len(filenames)

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
        rngs=nnx.Rngs(params=0),
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
    print(f"Model not found at {MODEL_PATH}")
    exit()

# Create the file we are writing the predictions to
outfile = open("predictions.csv", "w")
print("filename,prediction", file=outfile)

# Put model into eval mode
model.eval()

# Instead of batches, process the images one at a time
for filename in filenames:
    path = f"{DATA_DIR}/test/{filename}"
    print("Testing:", path)

    # Use pillow to load the image
    img = Image.open(path).convert("RGB")

    # Use the transform function in ImageDataset to resize, crop, and normalize
    img = transform(img, NORMALIZATION_MEAN, NORMALIZATION_STD)

    # Add a new axis at position 0 to the image to make it a batch of 1
    inputs = jnp.expand_dims(img, axis=0)

    # Do model inference
    outputs = model(inputs)

    # Get the hard prediction
    pred = jnp.argmax(outputs, axis=-1)

    # Get the category label corresponding to the prediction
    label = categories.categories[int(pred[0])]

    # Store the filenane and prediction in a file
    print(f"{filename},{label}", file=outfile)

outfile.close()
