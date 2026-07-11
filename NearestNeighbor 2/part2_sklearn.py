"""
Part 2: KNN using scikit-learn + Cross-Validation (Model Selection)
Run: python part2_sklearn.py

This script will:
  Part 2A: Train/test sweep for k=1..10 and save accuracy_vs_k.png
  Part 2B: 6-fold cross-validation on TRAIN set for k=1..20 to pick best k
  Part 2C: Train on full TRAIN set with best k and evaluate on TEST set
"""

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from utils import load_har_dataset, display_confusion_matrix_and_accuracy, accuracy


def KNN_using_sklearn(X_train, y_train, X_test, y_test, k):
    """
    Train a scikit-learn KNN classifier with k neighbors and evaluate on (X_test, y_test).
    Returns:
        y_pred: predicted labels for X_test
        acc: accuracy(y_test, y_pred)
    """
    
    #Your code goes here
    classifier = KNeighborsClassifier(k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy(y_test, y_pred)

    return y_pred, acc


def cross_validate_knn(X, y, k, n_splits=6, random_state=42):
    """
    Manual k-fold cross-validation on the provided dataset (X, y) using scikit-learn KNN.
    Returns mean accuracy across folds.

    NOTE: Cross-validation is performed on the TRAINING SET ONLY.
    """
    key = jax.random.PRNGKey(random_state)
    indices = jnp.arange(len(X))
    indices = jax.random.permutation(key, indices)

    #Your code goes here
    # We can use jnp.array_split to evenly split the indices into n_split folds
    folds = jnp.array_split(indices, n_splits)
    classifier = KNeighborsClassifier(n_neighbors=k)
    res = 0
    for i in range(n_splits):
        test_index = folds[i]
        train_index = jnp.concatenate([folds[j] for j in range(n_splits) if j != i])

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        res += accuracy(y_test, y_pred)

    return res / n_splits

            


def main():
    print("Part 2: KNN using scikit-learn + Cross-Validation")

    # Load HAR dataset
    X_train, y_train, X_test, y_test = load_har_dataset()

    # Part 2A: test sweep k=1..10
    print("\nPart 2A: Accuracy + confusion matrix on TEST set for k=1..10")
    ks = list(range(1, 11))
    accs = []

    for k in ks:
        y_pred, acc = KNN_using_sklearn(X_train, y_train, X_test, y_test, k)
        accs.append(acc)
        display_confusion_matrix_and_accuracy(k, y_test, y_pred)

    best_k_test = ks[int(np.argmax(np.array(accs)))]
    print(f"\nBest k on TEST sweep (k=1..10): {best_k_test} with accuracy {max(accs)*100:.2f}%")

    # Plot and save accuracy vs k (from test sweep)
    plt.figure(figsize=(8, 4))
    plt.plot(ks, [a * 100 for a in accs], marker='o')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN (sklearn) Accuracy vs k')
    plt.grid(True)
    plt.savefig('accuracy_vs_k.png', bbox_inches='tight')
    print("Saved plot to accuracy_vs_k.png")

    # Part 2B: CV on train set k=1..20
    print("\nPart 2B: 6-fold cross-validation on TRAIN set for k=1..20")
    cv_results = {}

    for k in range(1, 21):
        mean_acc = cross_validate_knn(X_train, y_train, k=k, n_splits=6, random_state=42)
        cv_results[k] = mean_acc
        print(f"k={k:2d}, Mean CV Accuracy={mean_acc*100:.2f}%")

    best_k_cv = max(cv_results, key=cv_results.get)
    print(f"\nBest k from CV: {best_k_cv} with accuracy={cv_results[best_k_cv]*100:.2f}%")

    # Part 2C: train best k on full train, evaluate on test
    print("\nPart 2C: Final evaluation on TEST set using best k from CV")
    y_pred, test_acc = KNN_using_sklearn(X_train, y_train, X_test, y_test, best_k_cv)
    display_confusion_matrix_and_accuracy(best_k_cv, y_test, y_pred)


if __name__ == "__main__":
    main()
