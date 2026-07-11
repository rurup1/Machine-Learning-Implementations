"""
utils.py
Common utilities for the HAR KNN assignments with tie-breaking.
"""

import os
import jax 
import jax.numpy as jnp
import functools
import numpy as np
from typing import Tuple, List


def load_har_dataset(base_path: str = "UCI HAR Dataset") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load the HAR dataset from the UCI archive."""
    train_x = os.path.join(base_path, "train", "X_train.txt")
    train_y = os.path.join(base_path, "train", "y_train.txt")
    test_x = os.path.join(base_path, "test", "X_test.txt")
    test_y = os.path.join(base_path, "test", "y_test.txt")

    if not (os.path.exists(train_x) and os.path.exists(train_y) and os.path.exists(test_x) and os.path.exists(test_y)):
        raise FileNotFoundError(
            f"Dataset files not found under {base_path}. Ensure UCI HAR Dataset is in current directory."
        )

    X_train = jnp.asarray(np.loadtxt(train_x), dtype=jnp.float32)
    y_train = jnp.asarray(np.loadtxt(train_y, dtype=np.int32).ravel(), dtype=jnp.int32)
    X_test = jnp.asarray(np.loadtxt(test_x), dtype=jnp.float32)
    y_test = jnp.asarray(np.loadtxt(test_y, dtype=np.int32).ravel(), dtype=jnp.int32)
    return X_train, y_train, X_test, y_test


def euclidean_distances(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute squared Euclidean distances between rows of a and rows of b."""
    #Your code goes here
    
    # Need broadcasting here. We know a.shape -> (i, f) -> (2947, 561)
    # b.shape -> (j, f) -> (7352, 561). 
    # We know the result tensor is of shape (i,j), and the sum() marginalizes over the features. Therefore, the pre-sum tensor is of shape (i,j,f).
    # We can achieve this by broadcasting: a[:, None, :] - b[None, :, :]. This is a naive solution because it is not memory efficient.

    # We can achieve the memory solution because (x-y)^2 = |x|^2 + |y|^2 - 2xy. Therefore:
    a_sum = jnp.sum(a ** 2, axis=1) # a.shape -> (i, )
    b_sum = jnp.sum(b ** 2, axis=1) # b.shape -> (j, )

    # We still want an end of (i,j). So we can use the same broadcasting techinque:
    dist = a_sum[:, None] + b_sum[None, :] - (2 * (a @ b.T))

    # Note that we transpose here so that (2947, 561) @ (7352, 561) becomes (2947, 561) @ (561, 7352). Then, finally, we return the squared Euclidean distances.
    # We need to clamp the values because of floating point errors of squaring small numbers.
    return jnp.maximum(0.0, dist)



def majority_vote(neighbor_labels: jnp.ndarray, neighbor_distances: jnp.ndarray) -> int:
    """
    neighbor_labels: array of labels of k nearest neighbors
    neighbor_distances: array of distances corresponding to neighbor_labels
    Returns the majority label. In case of tie, pick the label whose closest
    neighbor among the tied labels is nearest.
    """
    #Your code goes here
    # What is the shape of neighbor_labels, neighbor_distances?
    # Does neighbor_labels.shape == neighbor_distances.shape? YES
    # So from the euclidean_distances function, we know that the output is of shape (i,j). Therefore, we are mapping each test point to every other training point.
    # For this function, we might assume that the input shape is (j, ), but it is important to remember that we are grabbing the majority vote of the k nearest
    # neighbors. So, the input shape is really (k,). The output shape is just one label, the majority vote.

    # First, we need to grab the counts for each class. In the HAR dataset, there are 6 classes.
    # We can do this with jnp.bincount()
    counts = jnp.bincount(neighbor_labels, length=7)

    # We now have the classes, now we should find the minimum distances per class
    min_dist = jax.ops.segment_min(neighbor_distances, neighbor_labels, num_segments=7)

    # Now we need to mask counts such we only get the majority counts. This is a boolean mask. The most frequent class becomes True at that
    # index. Counts that are tied all become true.
    counts = counts == counts.max()

    # Finally, we need to return the majority label. If tied, we need to consider the smallest distance as well. We push values that are not valid
    # to infinity so that argmin() works smoothly
    res = jnp.argmin(jnp.where(counts, min_dist, jnp.inf))

    return res

def confusion_matrix_multiclass(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the confusion matrix
    Rows = true classes, Columns = predicted classes.
    """
    #Your code goes here
    cm = jnp.zeros((6,6), dtype=jnp.int32)
    cm = cm.at[y_true -1 , y_pred - 1].add(1)
    return cm

def display_confusion_matrix_and_accuracy(k: int, y_true: jnp.ndarray, y_pred: jnp.ndarray):
    """
    Display a readable confusion matrix table.
    """
    acc = accuracy(y_true, y_pred)
    print(f"k = {k}  accuracy={acc*100:5.2f}%")
    cm = confusion_matrix_multiclass(y_true, y_pred)
    classes = jnp.unique(jnp.concatenate((y_true, y_pred)))

    print("Confusion Matrix:")
    print("True\\Pred", end="\t")
    for c in classes:
        print(f"{c}", end="\t")
    print()

    for i, c in enumerate(classes):
        print(f"{c}", end="\t\t")
        for j in range(len(classes)):
            print(f"{cm[i, j]}", end="\t")
        print()

def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    #Your code goes here
    return jnp.mean(y_true == y_pred)


@functools.partial(jax.jit, static_argnames=("k",))
def predict_knn_jit(X_test: jnp.ndarray, X_train: jnp.ndarray, y_train: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    JIT-compiled KNN prediction helper.

    Students: implement this function using JAX operations only.
    Requirements (enforced by the assignment writeup):
      - call euclidean_distances(X_test, X_train)
      - select k nearest neighbors using jnp.argpartition or jnp.argsort
      - gather neighbor labels and distances using indexing / jnp.take_along_axis
      - vote using jax.vmap(majority_vote)(...)
    Returns:
      y_pred: shape (n_test,), dtype int32
    """
    # Your code goes here
    # 1. Call euclidean_distances (X_test, X_train)
    dists = euclidean_distances(X_test, X_train)

    # 2. Now, we need to select the k-nearest neighbors (distances)
    neighbors_idx = jnp.argpartition(dists, kth=k-1, axis=1)[:, :k]

    # 3. Now, we gather the k neighbor distances and labels corresponding to these indices
    neighbor_distances = jnp.take_along_axis(dists, neighbors_idx, axis=1)
    neighbor_labels = y_train[neighbors_idx]

    # 4. Now, we take the majority vote using jax.vmap. Need to transform the function first, then call it.
    y_pred =  jax.vmap(majority_vote)(neighbor_labels, neighbor_distances)
    return y_pred


class ScratchKNN:
    """KNN classifier from scratch with nearest-neighbor tie-breaking."""

    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test: jnp.ndarray) -> jnp.ndarray:
        """
        Predict labels for test samples.
        Uses full sorting to get k nearest neighbors.
        """
        y_pred = predict_knn_jit(X_test, self.X_train, self.y_train, self.k)
        return jnp.asarray(y_pred, dtype=jnp.int32)
