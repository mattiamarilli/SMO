import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
import os
from smo import SVM

# List of datasets to process
DATASETS = [f"./datasets/a{i}a" for i in range(1, 7)]  # a1a → a2a (can extend up to a6a)
MAX_ITER = 2000  # Maximum number of iterations for the custom SMO
KERNL_TYPE = 'rbf'  # Kernel type: 'rbf' or 'linear'
KKT_THR = 1e-2  # Threshold for KKT condition in SMO

# Function to load a dataset from libsvm format
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    X, y = load_svmlight_file(path)  # Load the dataset (sparse format)
    X = X.toarray()  # Convert to dense array
    y = y.astype(int)
    y[y == 0] = -1  # Convert labels from {0,1} to {-1,1}
    return X, y

# Function to benchmark custom SMO SVM vs scikit-learn SVC
def benchmark_svm_split(X, y, dataset_name, C_values=[0.1, 1, 10, 100], test_size=0.2):
    n_features = X.shape[1]
    gamma = 1.0 / n_features  # Gamma parameter for RBF kernel
    report = []

    print(f"\n=== Dataset: {dataset_name} | samples={X.shape[0]}, features={n_features} ===")

    # Shuffle data and split into training/test sets
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    for C in C_values:
        print(f"\n--- C = {C} ---")

        # Custom SMO SVM
        svm = SVM(c=C, kkt_thr=KKT_THR, max_iter=MAX_ITER, kernel_type=KERNL_TYPE, gamma_rbf=gamma)
        start_time = time.time()
        svm.fit(X_train, y_train)  # Train custom SVM
        runtime = time.time() - start_time
        y_pred, scores = svm.predict(X_test)  # Predict using custom SVM
        custom_accuracy = np.mean(y_pred == y_test)
        print(f"Custom SMO runtime: {runtime:.5f}s, test accuracy: {custom_accuracy:.4f}")

        # scikit-learn SVC
        clf = SVC(C=C, kernel='rbf', gamma=gamma)
        start = time.time()
        clf.fit(X_train, y_train)  # Train scikit-learn SVM
        runtime_sk = time.time() - start
        y_pred_sk = clf.predict(X_test)  # Predict using scikit-learn SVM
        acc_sk = accuracy_score(y_test, y_pred_sk)
        print(f"sklearn SVC runtime: {runtime_sk:.5f}s, test accuracy: {acc_sk:.4f}")

        # Append results to report
        report.append({
            "dataset": dataset_name,
            "C": C,
            "custom_train_time": runtime,
            "custom_accuracy": custom_accuracy,
            "svc_train_time": runtime_sk,
            "svc_accuracy": acc_sk
        })
    return report

# Function to save benchmark results as a formatted text table
def save_report_txt_table(report, filename="report_libsvm_datasets.txt"):
    header = (
        f"{'Dataset':>10} {'C':>6} "
        f"{'Custom_Train(s)':>15} {'Custom_Acc(%)':>15} "
        f"{'SVC_Train(s)':>12} {'SVC_Acc(%)':>12}\n"
    )
    separator = "-" * 80 + "\n"

    with open(filename, "w") as f:
        f.write("SVM Benchmark Report - Datasets a1a → a6a (train/test split)\n")
        f.write("="*80 + "\n\n")
        f.write(header)
        f.write(separator)
        for r in report:
            f.write(f"{r['dataset']:>10} {r['C']:>6} "
                    f"{r['custom_train_time']:>15.5f} {r['custom_accuracy']*100:>15.2f} "
                    f"{r['svc_train_time']:>12.5f} {r['svc_accuracy']*100:>12.2f}\n")
    print(f"\nReport saved as '{filename}'")

if __name__ == "__main__":
    all_results = []

    # Iterate over all datasets
    for dataset_path in DATASETS:
        dataset_name = os.path.basename(dataset_path)
        try:
            X, y = load_dataset(dataset_path)  # Load dataset
        except FileNotFoundError as e:
            print(e)
            continue

        results = benchmark_svm_split(X, y, dataset_name)  # Run benchmark
        all_results.extend(results)

    # Save the results to a text file
    save_report_txt_table(all_results)
