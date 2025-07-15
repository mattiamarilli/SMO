from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time
import numpy as np

from platt_smo import SmoAlgorithm  # Assumendo che sia questa la tua classe

def rbf_kernel(gamma=1.0):
    def rbf(x, y):
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)
    return rbf

def benchmark_svm_varying_size(full_X, full_y, test_X, test_y, C=1.0, gamma=None, sizes=[500]):
    n_features = full_X.shape[1]
    if gamma is None:
        gamma = 1.0 / n_features

    print(f"\nBenchmarking with variable training sizes (C={C}, gamma={gamma}):")
    for size in sizes:
        if size > full_X.shape[0]:
            print(f"Skipping size {size}: not enough samples.")
            continue

        X_train, y_train = shuffle(full_X, full_y, random_state=42)
        X_train = X_train[:size]
        y_train = y_train[:size]

        print(f"\nTraining size: {size}")

        # Custom SMO
        smo_start = time.time()
        smo = SmoAlgorithm(X_train, y_train, C=C, tol=0.001,
                           kernel=rbf_kernel(gamma=gamma),
                           use_linear_optim=False)
        smo.main_routine()
        smo_time = time.time() - smo_start

        # Predizioni SMO
        smo_preds = np.array([np.sign(smo.output(x)) for x in test_X])
        smo_acc = accuracy_score(test_y, smo_preds)
        print(f"Custom SMO runtime: {smo_time:.2f}s, test accuracy: {smo_acc:.4f}")

        # Sklearn SVC
        clf = SVC(C=C, kernel='rbf', gamma=gamma)
        start = time.time()
        clf.fit(X_train, y_train)
        runtime_sk = time.time() - start
        y_pred_sk = clf.predict(test_X)
        acc_sk = accuracy_score(test_y, y_pred_sk)
        print(f"sklearn SVC runtime: {runtime_sk:.2f}s, test accuracy: {acc_sk:.4f}")


if __name__ == "__main__":
    # Percorsi dataset
    train_path = "./datasets/a9a"
    test_path = "./datasets/a9a.t"

    # Carica tutto il training set
    full_X, full_y = load_svmlight_file(train_path)
    full_X = full_X.toarray()
    full_y = full_y.astype(int)
    full_y[full_y == 0] = -1  # Etichette in {-1, 1}

    # Carica test set
    test_X, test_y = load_svmlight_file(test_path, n_features=full_X.shape[1])
    test_X = test_X.toarray()
    test_y = test_y.astype(int)
    test_y[test_y == 0] = -1

    # Benchmark su diverse dimensioni di training set
    benchmark_svm_varying_size(full_X, full_y, test_X, test_y,
                               C=1.0,
                               sizes=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])