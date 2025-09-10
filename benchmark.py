from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time

from SMV_SMO import SMO_SVM  # importa la tua implementazione

def benchmark_svm_varying_size(full_X, full_y, test_X, test_y, C=1.0, sizes=[500]):
    """
    Benchmark tra la nostra implementazione SMO e sklearn SVC su varie dimensioni del training set.
    """
    n_features = full_X.shape[1]
    gamma = 1.0 / n_features  # solo per riferimento se usiamo kernel RBF custom

    print(f"\nBenchmarking with variable training sizes (C={C}):")
    for size in sizes:
        if size > full_X.shape[0]:
            print(f"Skipping size {size}: not enough samples.")
            continue

        X_train, y_train = shuffle(full_X, full_y, random_state=42)
        X_train = X_train[:size]
        y_train = y_train[:size]

        print(f"\nTraining size: {size}")

        # # Custom SMO
        svm = SMO_SVM(X_train, y_train, C=C) # usa kernel lineare per default
        start = time.time()
        svm.fit()
        runtime = time.time() - start
        y_pred_custom = svm.predict(test_X)
        acc = accuracy_score(test_y, y_pred_custom)
        print(f"Custom SMO runtime: {runtime:.2f}s, test accuracy: {acc:.4f}")

        # Sklearn SVC (lineare per confronto equo)
        clf = SVC(C=C, kernel='linear')
        start = time.time()
        clf.fit(X_train, y_train)
        runtime_sk = time.time() - start
        y_pred_sk = clf.predict(test_X)
        acc_sk = accuracy_score(test_y, y_pred_sk)
        print(f"sklearn SVC runtime: {runtime_sk:.5f}s, test accuracy: {acc_sk:.4f}")


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

    test_X_sub, test_y_sub = shuffle(test_X, test_y, random_state=42)
    test_X_sub, test_y_sub = test_X_sub[:1000], test_y_sub[:1000]

    # Benchmark su diverse dimensioni di training set
    benchmark_svm_varying_size(full_X, full_y, test_X_sub, test_y_sub, C=1.0, sizes=[50,100,200,500,1000,32561])
