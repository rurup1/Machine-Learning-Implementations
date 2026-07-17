import gzip
import os
from urllib import request

import jax.numpy as jnp

filenames = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(filename):
    # base_url = "http://yann.lecun.com/exdb/mnist/"
    base_url = "https://github.com/fgnt/mnist/raw/master/"
    url = base_url + filename
    print(f"Downloading {url} ...")
    request.urlretrieve(base_url + filename, filename)
    print("Download complete.")


# Takes "training" or "test" as an argument
# return (images, labels) as result
def load(name):
    key = name + "_images"
    image_path = filenames[key]
    if not os.path.exists(image_path):
        download_mnist(image_path)
    with gzip.open(image_path, "rb") as f:
        images = jnp.frombuffer(f.read(), jnp.uint8, offset=16).reshape(-1, 28 * 28)

    key = name + "_labels"
    label_path = filenames[key]
    if not os.path.exists(label_path):
        download_mnist(label_path)
    with gzip.open(label_path, "rb") as f:
        labels = jnp.frombuffer(f.read(), jnp.uint8, offset=8)
    return (images, labels)
