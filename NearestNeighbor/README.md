# KNN + Gradient Boosted Decision Trees

## I implemented the K-Nearest Neighbor Algorithm from scratch, and used popular libraries to compare and benchmark speed vs. accuracy tradeoffs.

---

## Table of Contents

- [Overview](#overview)
- [The Dataset](#the-dataset)
- [Background: How KNN works](#how-knn-works)
- [Repository Structure](#repository-structure)
- [Core Engine: utils.py](#core-engine-utilspy)
- [Part 1: KNN from Scratch](#part-1---knn-from-scratch)
- [Part 2: scikit-learn + cross validation](#part-2---scikit-learn--cross-validation)
- [Part 3: Scaling with FAISS](#part-3---scaling-with-faiss)
- [Part 4: XGBoost Baseline](#part-4---xgboost-baseline)
- [How to Run](#how-to-run)
- [What I Learned](#what-i-learned)

---

## Overview

Everything is based off the UCI HAR dataset, which has 561 sensor features per sample, and classifies into six human activities (walking, standing, sitting, etc.) There are four ways I built this algorithm: fully vectorized, JIT-compiled version with JAX (Part 1); the same thing with scikit-learn, using n-fold cross validation to select k (Part 2); then I implement an approximate nearest neighbor version using FAISS that trades some accuracy for significant speedup (Part 3); Finally, I used a gradient-boosted tree model with XGBoost as a decision tree-based baseline (Part 4). By doing this, I realized that exact KNN is simple and accurate, but is not scalable. Each tool/library I add on improves performance but gives up something else.

---

## The Dataset

The UCI HAR Dataset is collected from volunteers who carried smartphones in their pockets while performing six different activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying.

There are 561 pre-processed measurements from smartphone accelerometers and gyroscopes. The labels are the six activity classes, and the split is the 561 pre-processed measurements.

To run this project, we need to download the UCI HAR Dataset from the link: [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Make sure you place this dataset in the root NearestNeighbor folder.

### Shapes and Labels

| **Array** | **Shape** | **Meaning** |
|---|---|---|
| X_train | (7352, 561) | 7352 samples of 561 features |
| y_train | (7352, ) | 7352 labels ranging from 1-6 |
| X_test | (2947, 561) | 2947 samples of 561 features |
| y_test | (2947, ) | 2947 labels ranging from 1-6 |

---

## How KNN Works

### Lazy Learning

We call KNN a *lazy learning* algorithm because it does most of its work during the prediction, not during training. fit() doesn't learn a mathematical model, it just stores the training data, and when predict() is called, it searches through that stored data and finds the nearest neighbors. The pipeline becomes: Training Data - fit() - stores all the points - predict - computes distances to every stored point.

The simplest implementation of fit() is just:

```python
self.X_train = X_train
self.y_train = y_train
```

There are no weights, optimization, gradient descent, or loss minimization, just memorizing the dataset.

### Distance Metric

The most common distance metric that is used in KNN is Euclidean distance. This measures the straight-line distance between two points. The formula is:

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

KNN only cares about which points are *closest*, not their exact distances. Therefore, we can ignore the square root, because the square root function is strictly increasing. Since computing a square root is far more expensive than addition and multiplication, avoiding millions of square root operations can noticeably improve performance.

### Voting and Tie-Breaking

Once the algorithm has decided the ***k*** nearest neighbors, it must decide what prediction to make. Because this is a classification algorithm, we do this with a majority vote.

You can think of this like an election. Each of the k nearest neighbors gets one vote, and the class with the most votes wins.

There are cases where two or more classes can receive the same vote, so the algorithm must have a tie-breaking rule as well. There are multiple ways to do this:

1. Choose the class of the closest neighbor (nearest point among the tied classes wins)
2. Use distance-weighted voting (assign more weight to closer neighbors)

For this implementation, we choose the class with the closest neighbor.

### Choosing k

For KNN, ***k*** is a hyperparameter. It controls how many neighbors KNN targets before making a prediction. This directly affects how flexible or smooth the decision boundary is, which creates a bias/variance trade-off.

Consider k = 1, which just copies the label of the closest label. This makes the model very flexible and it can capture the tiniest local patterns in the data. This includes random noise as well. However, small changes to the training data can entirely change the prediction. Therefore, there is high variance and low bias with k = 1, which often leads to overfitting.

Now consider k = 100. This takes the majority vote on the 100 nearest neighbors. Noisy samples have very little influence because they are outvoted by surrounding examples. The model becomes less flexible and can miss some important patterns. However, the predictions are more stable because they depend on many data points. Therefore, with k = 100, we have high bias, and low variance, which often leads to underfitting.

So, as k increases, the predictions get smoother, but are less effective at capturing local patterns. There is no universally *best* k, it all depends on the dataset. We will use n-fold cross validation to determine which k performs best on the unseen data.

---

## Repository Structure

```
NearestNeighbor/
├── utils.py               # Common utilities for KNN
├── part1_knn_scratch.py   # Runs the ScratchKNN class from utils.py
├── part2_sklearn.py       # scikit-learn + n-fold cross validation
├── part3_faiss.py         # FAISS L2 Search + majority vote
├── part4_xgboost.py       # XGBoost multiclass classifier
└── accuracy_vs_k.png      # n-fold cross validation results
```

---

## Core Engine: utils.py

### load_har_dataset

This function reads X_train.txt, X_test.txt, y_train.txt, y_test.txt from the HAR dataset. The X arrays are loaded as float32 (continuous features) and the y arrays are loaded as int32, as the labels are discrete.

The shape of the X arrays are: (n, 561). We ravel to flatten the array into a (n, ) vector, which is what the rest of the code expects.

### euclidean_distances

This function computes the squared Euclidean distances between rows of a and rows of b. To do this, we use broadcasting.

**Input / Output shapes:**

a: (2947, 561) b: (7352, 561) -> result: (2947, 7352)

**How it works:**

Say i = 2947, j = 7352, f = 561.

We know that the output tensor must be of shape (i, j), but have input tensors of (i, f), (j, f). Therefore, we must sum() to marginalize over f. So, the pre-sum tensor must be of shape (i, j, f). We can achieve this by broadcasting:

```python
pre_sum_tensor = a[:, None, :] - b[None, :, :]
```

However, this is not a memory efficient solution, as it calculates an intermediate 3-D tensor of shape (i, j, f), which is around 48GB. In fact, this crashed my computer. We don't want this middle tensor, so we can use this identity:

$$
\lVert x - y \rVert^2 = \lVert x \rVert^2 + \lVert y \rVert^2 - 2xy
$$

With this, the f dimension gets contracted inside the matmul a @ b.T, so its output is already (i, j). The memory is dominated by (i, j), which is around 87 MB plus the two small norm vectors. Basically, the feature axis gets summed away as a part of the matmul rather than blowing up into a 3D tensor.

While the naive subtraction is numerically more stable, the identity form can produce tiny negative values from floating-point cancellation when two points are nearly identical. So, we clamp to zero.

### majority_vote

This function returns the majority label based on the labels and distances of the k nearest labels and distances. In the case of a tie, it picks the label whose closest neighbor among the tied labels is nearest.

The shape of the input labels and distances is (k, ), because we are already given the k-nearest labels and distances. The output is just an integer, the majority vote. First, we need to find how many times each class appears in neighbor_labels. In the HAR dataset, there are 6 classes. We can use jnp.bincount(neighbor_labels, length=7). We set the length to 7 because the labels are 1-6, so we can label by index and ignore index 0.

Now that we have the counts for each class, now we should find the minimum distances per class. We use this in case of a tie. The jax.ops.segment_min() is useful here. Once this is done, we need to perform a boolean mask to filter for the majority counts. If there is a single majority, there will only be one index in the array with value True, else multiple will be set to True. This is easily done with counts = counts == counts.max()

Finally, we need to return the majority label. If tied, we must consider the smallest distance as well. We push values that are not valid to infinity, so we can properly utilize argmin(). This is all. The code becomes:

```python
def majority_vote(neighbor_labels, neighbor_distances) -> int:
     counts = jnp.bincount(neighbor_labels, length=7)
     min_dist = jax.ops.segment_min(neighbor_distances, neighbor_labels, num_segments=7)
     counts = counts == counts.max()
     res = jnp.argmin(jnp.where(counts, min_dist, jnp.inf))
     return res
```

### predict_knn_jit

This function is the JIT-compiled KNN prediction helper. It calls euclidean_distances, selects the k nearest neighbors, gathers the neighbor labels and distances using indexing, and votes. This returns y_pred.

We use partial to bake the static_argnames setting into jax.jit, so it can be used as a decorator. jax.jit compiles the function per input-signature, and marking k static lets JAX use k's real value to pin the shapes that it depends on. This comes at the cost of a recompile whenever k changes.

We start off by calling euclidean_distances(X_test, X_train) to gather the distances. We now need to select the k-nearest neighbors from these distances. We could use jnp.argsort() or jnp.argpartition(). argpartition is a better choice here. argsort is slower because it sorts the distances, but if we look at our majority vote function, we do not need the distances to be sorted! Therefore, we can use argpartition to grab the k-nearest indices

Then, we need to gather the k neighbor distances and labels that correspond to these indices. We can use jnp.take_along_axis() to grab the k neighbor distances, and fancy indexing with y_train[idx] to grab the labels.

Once we have this, we need to take the majority vote. We do this with jax.vmap(). This transforms the majority_vote function by vectorizing it across all test points (eliminating the need for loops).

### confusion_matrix_multiclass

This function computes the confusion matrix based on the true classes and the predicted classes.

We use a scatter add approach with cm.at[row, col].add(1)

### accuracy

This function returns the accuracy given an array of true predictions and an array of predictions. We can take the mean of y_true == y_pred to give us the correct fraction, as True = 1, False = 0

---

## Part 1 - KNN from Scratch

This script will load the HAR dataset, run the Scratch KNN class from utils.py for k = 4, 6, and then display the confusion matrix and accuracy. The flow is:

```python
classifier = ScratchKNN(k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

For k = 4, accuracy = 89.72%. For k = 6, accuracy = 90.57%

---

## Part 2 - scikit-learn + cross-validation

### KNN_using_sklearn

First, we train a scikit-learn KNN classifier with k neighbors, and evaluate on (X_test, y_test). This follows the same steps as part 1.

### cross_validate_knn

Then, we need to run n-fold cross validation to *choose* the most optimal k.

n-fold cross validation is a technique used for estimating how well our model will perform on the new, test data. Instead of splitting the data into a training set and a testing set, we split it into n equal-sized folds. Consider n = 5:

| Iteration | Training Folds | Test Fold |
|---|---|---|
| 1 | 2,3,4,5 | 1 |
| 2 | 1,3,4,5 | 2 |
| 3 | 1,2,4,5 | 3 |
| 4 | 1,2,3,5 | 4 |
| 5 | 1,2,3,4 | 5 |

Every sample is in the training set exactly n - 1 times, and in the testing set exactly once. We can average all the folds and determine the best k.

We do this because a single train/test can be unlucky.

I then tested this for k=1 to k=20, and chose the k that yielded the highest accuracy. This ended up being k = 8, for accuracy 90.74%

---

## Part 3 - Scaling with FAISS

### Exact vs. Approximate Nearest Neighbor

If FitWell plans to deploy activity recognition on millions of daily logs from users, exact KNN will be too slow and memory-intensive. To handle this at scale, we need Approximate Nearest Neighbors. (ANN). This is a version of KNN that uses an IVF-PQ (Inverted File with Product Quantization)

ANN is typically better because if we had millions of logs, it avoids computing and tracking all of the vectors. Instead, it uses clever data structures to quickly narrow the search. So, instead of searching millions, we might search 50,000.

If the true nearest neighbors were ABCDE, ANN might return ABDEF. It missed C, but found F instead. This tradeoff comes with significant speed improvements, which is why it is used for search, recommendations, and RAG.

### The FAISS Index Stack

FAISS (Facebook AI Similarity Search) is the library that we will use for fast similarity search. It is an open-source library created by Meta for efficient similarity search and vector clustering.

The IVF-PQ is one of FAISS's most scalable index types. It combines an inverted file index with product quantization. This is what happens under the hood, suppose we have 10 million embeddings:

1. Train the IVF clusters

   Before adding them to an index, FAISS runs k-means clustering on a representative sample of vectors. Say this transforms our embeddings into 4096 centroids, where a centroid is a center of a cluster. Centroids can act as buckets for organizing our vectors.

2. Assign each vector to a centroid

   Every vector is assigned to its nearest centroid. Instead of one huge list, now we have many smaller lists

3. Compress each vector (PQ)

   FAISS uses product quantization to split the vector into smaller chunks, learns a small codebook for each chunk, and stores only the closest codeword for each chunk. So, instead of storing thousands of floating point values, FAISS stores a compact sequence of small integers. This significantly reduces memory usage.

4. A query arrives

   Suppose we query the index, then it first computes the query embedding

5. Find the nearest clusters

   FAISS then compares the query only against the IVF centroids. The number of clusters searched is controlled by the nprobe parameter.

6. Search within those clusters

   Instead of searching all 10 million vectors, FAISS only searches the vectors in the selected clusters. This is where the speedup lives.

7. Compare compressed vectors

   The vectors in the clusters are stored as PQ codes. FAISS estimates distances using compressed representations, which avoids needing to fully decompress a vector.

8. Return the top-k results

   FAISS finally ranks the candidates and returns the nearest neighbor.

Our ANN implementation is slightly more complex. We use an HNSW index as the coarse quantizer inside of an IVFPQ index. At any IVF index, a quantizer's job is to answer which IVF clusters should this query search? HNSW (Hierarchical Navigable Small World) improves recall while keeping indexing efficient.

So, once HNSW returns the closest centroids, IVFPQ takes over. The rest of the pipeline remains.

### Timing Comparison

I ran a comparison on the timing of my scratch KNN and FAISS implementation. As expected, FAISS yielded a speedup with minimal loss in accuracy.

---

## Part 4 - XGBoost Baseline

KNN is a strong baseline, but it can be expensive at inference because it has to compare each test point to many training points. Therefore, we will train a gradient boosted decision tree using XGBoost and evaluate it on the HAR dataset.

Instead of remembering every example, like KNN, a GBDT builds many small decision trees. Tree 1 might learn some pattern, and Tree 2 focuses on correcting Tree 1's mistakes, and so on. These many trees end up combining to approximate complicated boundaries.

XGBoost (Extreme Gradient Boosting) is an optimized implementation of a GBDT that is supposed to be faster, more accurate, and less prone to overfitting. It also made boosting practical at large scale because of parallelized tree construction, efficient memory usage, sparse feature handling, regularization, tree pruning, and optimized cache use.

The main strength of XGBoost is that it discourages overly complex trees due to regularization (penalizes too many leaves, very large leaf weights). This is why it helps reduce overfitting.

Some important hyperparameters are: n_estimators, learning_rate, num_class, max_depth, reg_lambda (L2 regularization strength), eval_metric (used internally during training to track performance (mlogloss)), tree_method="hist".

My final XGBoost test accuracy was 93.79%.

---

## How to Run

### 1. Install dependencies

```bash
pip install jax numpy scikit-learn matplotlib faiss-cpu xgboost
```

### 2. Download the dataset

The dataset is not included in this repo. Download it from [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), unzip it, and place the `UCI HAR Dataset/` folder in the root of `NearestNeighbor/`:

```
NearestNeighbor/
├── UCI HAR Dataset/
│   ├── train/
│   └── test/
├── utils.py
└── ...
```

### 3. Run the parts

Each part is a standalone script. Run them from inside the `NearestNeighbor/` folder:

```bash
python part1_knn_scratch.py   # scratch KNN for k = 4, 6 + confusion matrix
python part2_sklearn.py       # sklearn KNN + n-fold CV, saves accuracy_vs_k.png
python part3_faiss.py         # FAISS ANN vs scratch KNN timing comparison
python part4_xgboost.py       # XGBoost baseline
```

---

## What I Learned

**Shape-first thinking.** The biggest takeaway from this project was learning to reason about array shapes before writing any code. For every function in utils.py, I started by writing down the input and output shapes, then figured out which operations get from one to the other. Once I knew the output of euclidean_distances had to be (2947, 7352) from inputs of (2947, 561) and (7352, 561), the structure of the solution was already half-decided.

**Vectorization has a memory cost.** Broadcasting is powerful, but the naive broadcasted subtraction created a 48GB intermediate tensor that crashed my computer. The fix was the expanded norm identity, which lets the feature axis get contracted inside a matmul instead of materializing a 3D tensor. Lesson: think about the shape of every intermediate, not just the output.

**JAX's mental model.** jit compiles per input-signature, so k has to be marked static because array shapes depend on it. vmap turns a function that handles one test point into one that handles all of them, with no loops. And small algorithmic choices matter: argpartition beats argsort because the majority vote never needed sorted distances.

**Hyperparameters need cross-validation.** A single train/test split can be lucky or unlucky. n-fold cross validation gave a much more trustworthy estimate for picking k, and it only uses the training set, so the test set stays untouched until the final evaluation.

**Exact vs. approximate is a real engineering tradeoff.** Exact KNN is simple and accurate but compares every query to every training point. FAISS's IVF-PQ index gave a significant speedup by searching only a few clusters of compressed vectors, at the cost of occasionally missing a true neighbor. That tradeoff is why ANN powers search, recommendations, and RAG at scale.

**KNN is a baseline, not an endpoint.** XGBoost beat every KNN variant (93.79% vs ~90.7%) while being cheaper at inference, because it learns a model instead of memorizing the dataset. Starting from a simple, transparent algorithm made it much clearer what the fancier tools are actually buying you.
