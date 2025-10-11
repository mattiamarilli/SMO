import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from smo import SVM

# Function to benchmark custom SMO SVM vs scikit-learn SVC on synthetic datasets
def benchmark_svm(datasets, C_values=[0.1, 1, 10, 100], test_size=0.2, random_state=42):
    report = []

    # Iterate over different dataset sizes (num_samples, num_features)
    for num_samples, num_features in datasets:
        print(f"\nDataset: samples={num_samples}, features={num_features}")

        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=int(num_features * 0.6),  # 60% informative features
            n_redundant=int(num_features * 0.1),    # 10% redundant features
            n_classes=2,
            n_clusters_per_class=1,
            random_state=random_state
        )

        y = 2 * y - 1  # Convert labels from {0,1} to {-1,1} for SVM

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Standardize features (zero mean, unit variance)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gamma = 1.0 / num_features  # RBF kernel parameter

        # Iterate over different regularization values C
        for C in C_values:
            print(f"\n--- C = {C} ---")

            # ------------------------------
            # Custom SVM using SMO
            # ------------------------------
            model = SVM(c=C, kkt_thr=1e-1, max_iter=2000, kernel_type='rbf', gamma_rbf=gamma)
            start_time = time.time()
            model.fit(X_train, y_train)  # Train custom SVM
            training_time = time.time() - start_time

            start_time = time.time()
            y_pred, scores = model.predict(X_test)  # Predict using custom SVM
            prediction_time = time.time() - start_time
            accuracy = np.mean(y_pred == y_test)
            print(f"SVM custom        - Train: {training_time:.3f}s, Predict: {prediction_time:.3f}s, Acc: {accuracy*100:.2f}%")

            # ------------------------------
            # Scikit-learn SVC
            # ------------------------------
            svc_model = SVC(C=C, kernel='rbf', gamma=gamma)
            start_time = time.time()
            svc_model.fit(X_train, y_train)  # Train SVC
            svc_training_time = time.time() - start_time

            start_time = time.time()
            y_pred_svc = svc_model.predict(X_test)  # Predict with SVC
            svc_prediction_time = time.time() - start_time
            svc_accuracy = np.mean(y_pred_svc == y_test)
            print(f"SVC sklearn       - Train: {svc_training_time:.3f}s, Predict: {svc_prediction_time:.3f}s, Acc: {svc_accuracy*100:.2f}%")

            # Save results to report
            report.append({
                "samples": num_samples,
                "features": num_features,
                "C": C,
                "custom_train_time": training_time,
                "custom_pred_time": prediction_time,
                "custom_accuracy": accuracy,
                "svc_train_time": svc_training_time,
                "svc_pred_time": svc_prediction_time,
                "svc_accuracy": svc_accuracy
            })

    return report

# Function to save benchmark results as a formatted text table
def save_report_txt_table(report, filename="report_make_classification.txt"):
    # Table header
    header = (
        f"{'Samples':>7} {'Features':>8} {'C':>6} "
        f"{'Custom_Train(s)':>15} {'Custom_Pred(s)':>15} {'Custom_Acc(%)':>15} "
        f"{'SVC_Train(s)':>12} {'SVC_Pred(s)':>12} {'SVC_Acc(%)':>12}\n"
    )
    separator = "-" * (7+8+6+15+15+15+12+12+12 + 8*2) + "\n"

    with open(filename, "w") as f:
        f.write("SVM Benchmark Report\n")
        f.write("="*90 + "\n\n")
        f.write(header)
        f.write(separator)
        # Write each result row
        for r in report:
            f.write(f"{r['samples']:>7} {r['features']:>8} {r['C']:>6} "
                    f"{r['custom_train_time']:>15.3f} {r['custom_pred_time']:>15.3f} {r['custom_accuracy']*100:>15.2f} "
                    f"{r['svc_train_time']:>12.3f} {r['svc_pred_time']:>12.3f} {r['svc_accuracy']*100:>12.2f}\n")
    print(f"\nReport saved as '{filename}'")

if __name__ == "__main__":
    # List of synthetic dataset configurations: (num_samples, num_features)
    datasets = [
        (20, 50),
        (100, 50),
        (1000, 50),
    ]

    # Run benchmark and save results
    report = benchmark_svm(datasets)
    save_report_txt_table(report)
