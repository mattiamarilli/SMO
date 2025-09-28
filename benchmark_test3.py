import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from test3 import SVM

def benchmark_svm(num_samples=100, num_features=10, test_size=0.2, random_state=42):
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=int(num_features * 0.6),
        n_redundant=int(num_features * 0.1), n_classes=2, n_clusters_per_class=1, random_state=random_state)

    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Benchmarking SVM Custom...")

    gamma = 1.0 / num_features

    model = SVM(c=1.0,kkt_thr=1e-3, max_iter=500, kernel_type='rbf', gamma_rbf=gamma)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (SVM personalizzato): {training_time:.3f} seconds")

    start_time = time.time()
    y_pred, scores = model.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = np.mean(y_pred == y_test)
    print(f"Prediction time (SVM personalizzato): {prediction_time:.3f} seconds")
    print(f"Accuracy (SVM personalizzato): {accuracy * 100:.2f}%")

    print("\n" + "-" * 50 + "\n")

    print("Benchmarking SVC (sklearn)...")

    svc_model = SVC(C=1.0, kernel='rbf',gamma=gamma)

    start_time = time.time()
    svc_model.fit(X_train, y_train)
    svc_training_time = time.time() - start_time
    print(f"Training time (SVC): {svc_training_time:.3f} seconds")

    start_time = time.time()
    y_pred_svc = svc_model.predict(X_test)
    svc_prediction_time = time.time() - start_time
    svc_accuracy = np.mean(y_pred_svc == y_test)
    print(f"Prediction time (SVC): {svc_prediction_time:.3f} seconds")
    print(f"Accuracy (SVC): {svc_accuracy * 100:.2f}%")

if __name__ == "__main__":
    benchmark_svm()
