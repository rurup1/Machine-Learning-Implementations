"""
Part 3: Scaling KNN with FAISS
Run: python part3_faiss.py
Compares sklearn KNN (for a chosen k) with FAISS L2 search + majority vote.
Note: FAISS may need to be installed. If faiss is not available, the script will explain how to install it.
"""
import time
import jax
import jax.numpy as jnp
from utils import load_har_dataset, accuracy, majority_vote, ScratchKNN

def scratch_knn_time_and_acc(X_tr, y_tr, X_ts, y_ts, k=5):
    t0 = time.time()

    # Your code goes here
    classifier = ScratchKNN(k)
    classifier.fit(X_tr, y_tr)
    y_pred = classifier.predict(X_ts)
    acc = accuracy(y_ts, y_pred)

    t1 = time.time()
    return (t1 - t0), acc

def faiss_knn_time_and_acc(X_tr, y_tr, X_ts, y_ts, k=5):
    try:
        import faiss
        import numpy as np
    except Exception as e:
        raise ImportError("faiss not found. Install with e.g. 'pip install faiss-cpu'") from e

    t0 = time.time()

    # Your code goes here
    X_tr = np.array(X_tr, dtype=np.float32)
    X_ts = np.array(X_ts, dtype=np.float32)
    d = X_tr.shape[1]

    quantizer = faiss.IndexHNSWFlat(d, 32)
    index = faiss.IndexIVFPQ(quantizer, d, 128, 17, 8)
    index.train(X_tr)
    index.add(X_tr)
    index.nprobe = 10

    # D -> Squared distances
    # I -> Indices of the nearest training vectors
    D, I = index.search(X_ts, k)
    neighbor_labels = y_tr[I]

    y_pred = jax.vmap(majority_vote)(neighbor_labels, D)
    acc = accuracy(y_ts, y_pred)

    t1 = time.time()
    return (t1 - t0), acc

def main():
    print("Part 3: FAISS scaling comparison")
    X_train, y_train, X_test, y_test = load_har_dataset()

    for k in [4, 6, 10, 20]:
        try:
            sk_time, sk_acc = scratch_knn_time_and_acc(X_train, y_train, X_test, y_test, k=k)
            print(f"scratch KNN k={k}: time={sk_time:.3f}s acc={sk_acc*100:.2f}%")
        except Exception as e:
            print("scratch KNN failed:", e)
            sk_time, sk_acc = None, None

        try:
            fa_time, fa_acc = faiss_knn_time_and_acc(X_train, y_train, X_test, y_test, k=k)
            print(f"faiss  k={k}: time={fa_time:.3f}s acc={fa_acc*100:.2f}%")
        except ImportError as e:
            print(str(e))
            fa_time, fa_acc = None, None

        print("-"*40)

if __name__ == '__main__':
    main()