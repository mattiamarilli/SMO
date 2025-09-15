import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time

from SMV_SMO import SMO_SVM

# === Global constants ===
TRAIN_PATH = "./datasets/a1a"  # Path to the training dataset
TEST_PATH = "./datasets/a1a.t"  # Path to the test dataset
VERBOSE = False  # Controls verbosity for the custom SMO solver


def rbf_kernel(gamma):
    # Returns a Radial Basis Function (RBF) kernel function with parameter gamma.
    # The kernel is defined as: exp(-gamma * ||x - y||^2)
    def kernel(x, y):
        diff = x - y
        return np.exp(-gamma * np.dot(diff, diff))

    return kernel


def benchmark_svm_varying_size(full_X, full_y, test_X, test_y, C=1.0, sizes=[500], verbose=False):

    #Benchmark custom SMO-based SVM vs scikit-learn's SVC using an RBF kernel.

    # Parameters:
    # full_X, full_y: training data and labels
    # test_X, test_y: test data and labels
    # C: regularization parameter
    # sizes: list of training set sizes to benchmark
    # verbose: whether to enable verbose output for the custom solver

    n_features = full_X.shape[1]
    gamma = 1.0 / n_features  # Common default gamma value for RBF kernel

    print(f"\nBenchmarking with variable training sizes (C={C}, RBF kernel, gamma={gamma}):")
    for size in sizes:
        if size > full_X.shape[0]:
            print(f"Skipping size {size}: not enough samples.")
            continue

        # Shuffle training data if using a subset
        if size < full_X.shape[0]:
            X_train, y_train = shuffle(full_X, full_y)
        else:
            X_train, y_train = full_X, full_y

        # Select the subset
        X_train = X_train[:size]
        y_train = y_train[:size]

        print(f"\nTraining size: {size}")

        # === Custom SMO implementation with RBF kernel ===
        kernel_rbf = rbf_kernel(gamma)
        svm = SMO_SVM(X_train, y_train, C=C, kernel_func=kernel_rbf, verbose=verbose)
        start = time.time()
        svm.fit()
        runtime = time.time() - start
        y_pred_custom = svm.predict(test_X)
        acc = accuracy_score(test_y, y_pred_custom)
        print(f"Custom SMO (RBF) runtime: {runtime:.5f}s, test accuracy: {acc:.4f}")

        # === Scikit-learn SVC with RBF kernel ===
        clf = SVC(C=C, kernel='rbf', gamma=gamma)
        start = time.time()
        clf.fit(X_train, y_train)
        runtime_sk = time.time() - start
        y_pred_sk = clf.predict(test_X)
        acc_sk = accuracy_score(test_y, y_pred_sk)
        print(f"sklearn SVC (RBF) runtime: {runtime_sk:.5f}s, test accuracy: {acc_sk:.4f}")


if __name__ == "__main__":
    # === Load training data ===
    full_X, full_y = load_svmlight_file(TRAIN_PATH)
    full_X = full_X.toarray()  # Convert sparse matrix to dense array
    full_y = full_y.astype(int)
    full_y[full_y == 0] = -1  # Convert labels {0,1} → {-1,1}

    # === Load test data ===
    test_X, test_y = load_svmlight_file(TEST_PATH)
    test_X = test_X.toarray()
    test_y = test_y.astype(int)
    test_y[test_y == 0] = -1

    # Ensure test and train sets have the same number of features
    if test_X.shape[1] < full_X.shape[1]:
        pad_width = full_X.shape[1] - test_X.shape[1]
        test_X = np.hstack([test_X, np.zeros((test_X.shape[0], pad_width))])
    elif test_X.shape[1] > full_X.shape[1]:
        test_X = test_X[:, :full_X.shape[1]]

    # Use only a subset of the test data for faster evaluation
    test_X_sub, test_y_sub = shuffle(test_X, test_y)
    test_X_sub, test_y_sub = test_X_sub[:1000], test_y_sub[:1000]

    # Run the benchmark with the given configuration
    benchmark_svm_varying_size(full_X, full_y, test_X_sub, test_y_sub,
                               C=1.0, sizes=[full_X.shape[0]], verbose=VERBOSE)
