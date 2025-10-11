from typing import Tuple
import numpy as np
import time


class SVM:
    def __init__(self, c: float = 1., kkt_thr: float = 1e-3, max_iter: int = 1e4, kernel_type: str = 'linear',
                 gamma_rbf: float = 1.) -> None:
        """
        Initialize the SVM model with hyperparameters.

        Args:
            c (float): Regularization parameter.
            kkt_thr (float): Threshold for KKT violation.
            max_iter (int): Maximum number of iterations.
            kernel_type (str): Type of kernel ('linear' or 'rbf').
            gamma_rbf (float): Gamma parameter for RBF kernel.
        """
        if kernel_type not in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))

        super().__init__()

        self.c = float(c)
        self.max_iter = int(max_iter)
        self.kkt_thr = kkt_thr

        # Set the kernel function
        if kernel_type == 'linear':
            self.kernel = self.linear_kernel
        elif kernel_type == 'rbf':
            self.kernel = self.rbf_kernel
            self.gamma_rbf = gamma_rbf

        # Initialize model parameters
        self.b = 0.0
        self.alpha = np.array([])
        self.support_vectors = np.array([])
        self.support_labels = np.array([])

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the labels for the given input.

        Args:
            x (np.ndarray): Input data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted labels and raw scores.
        """
        if self.alpha.shape[0] == 0:
            raise ValueError("Model not trained yet.")

        # Compute kernel between support vectors and input
        K_test = self.kernel(self.support_vectors, x)
        # Compute scores and predictions
        scores = np.dot(self.alpha * self.support_labels, K_test) + self.b
        pred = np.sign(scores)
        return pred, scores

    def mvp_selection(self, error_cache: np.ndarray) -> Tuple[int, int]:
        """
        Select the pair of alphas that most violate the KKT conditions.

        Args:
            error_cache (np.ndarray): Current error cache.

        Returns:
            Tuple[int, int]: Indices of the selected alphas (i, j).
        """
        alpha = self.alpha
        y = self.support_labels
        C = self.c

        # Determine sets L, U, Free
        L_indices = (alpha <= 1e-10)
        U_indices = (alpha >= C - 1e-10)
        Free_indices = (alpha > 1e-10) & (alpha < C - 1e-10)

        # Create masks for R and S sets
        R_mask = (L_indices & (y == 1)) | (U_indices & (y == -1)) | Free_indices
        S_mask = (L_indices & (y == -1)) | (U_indices & (y == 1)) | Free_indices

        R_indices = np.where(R_mask)[0]
        S_indices = np.where(S_mask)[0]

        # Remove NaN values from consideration
        R_indices = [i for i in R_indices if not np.isnan(error_cache[i])]
        S_indices = [i for i in S_indices if not np.isnan(error_cache[i])]

        if len(R_indices) == 0 or len(S_indices) == 0:
            return -1, -1

        # Check maximal KKT violation
        max_violation = np.max(error_cache[S_indices]) - np.min(error_cache[R_indices])
        if max_violation < self.kkt_thr:
            return -1, -1

        # Select indices with maximal violation
        i_mvp = R_indices[np.argmin(error_cache[R_indices])]
        j_mvp = S_indices[np.argmax(error_cache[S_indices])]

        return i_mvp, j_mvp

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model using SMO algorithm.

        Args:
            x_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
        """
        N, D = x_train.shape
        self.b = 0.0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train
        self.sumtimes = 0

        iter_idx = 0

        # Precompute kernel matrix and error cache
        if self.kernel == self.rbf_kernel:
            sq_norms = np.sum(x_train ** 2, axis=1)
            K = np.exp(-self.gamma_rbf * (sq_norms[:, None] + sq_norms[None, :] - 2 * x_train @ x_train.T))
            error_cache = (self.alpha * y_train) @ K + self.b - y_train
        else:
            K = x_train @ x_train.T
            error_cache = (self.alpha * y_train) @ K + self.b - y_train

        # Main SMO loop
        while iter_idx < self.max_iter:
            i_2, i_1 = self.mvp_selection(error_cache)
            if i_2 == -1 or i_1 == -1:
                break

            y_1, alpha_1 = self.support_labels[i_1], self.alpha[i_1]
            y_2, alpha_2 = self.support_labels[i_2], self.alpha[i_2]

            start_time = time.time()
            K_i1 = self.kernel_column(i_1)
            K_i2 = self.kernel_column(i_2)
            self.sumtimes += time.time() - start_time

            k11 = K_i1[i_1]
            k22 = K_i2[i_2]
            k12 = K_i1[i_2]

            # Compute bounds for alpha update
            L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)

            eta = k11 + k22 - 2 * k12
            if eta < 1e-10:
                continue

            # Compute errors
            E_1 = np.dot(self.alpha * self.support_labels, K_i1) + self.b - y_1
            E_2 = np.dot(self.alpha * self.support_labels, K_i2) + self.b - y_2

            # Update alphas
            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            alpha_2_new = np.clip(alpha_2_new, L, H)
            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)

            # Update bias term
            b1 = self.b - E_1 - y_1 * (alpha_1_new - alpha_1) * k11 - y_2 * (alpha_2_new - alpha_2) * k12
            b2 = self.b - E_2 - y_1 * (alpha_1_new - alpha_1) * k12 - y_2 * (alpha_2_new - alpha_2) * k22

            if 0 < alpha_1_new < self.c:
                self.b = b1
            elif 0 < alpha_2_new < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            # Save updated alphas
            self.alpha[i_1] = alpha_1_new
            self.alpha[i_2] = alpha_2_new

            # Update error cache
            delta_alpha_1 = alpha_1_new - alpha_1
            delta_alpha_2 = alpha_2_new - alpha_2
            error_cache += y_1 * delta_alpha_1 * K_i1 + y_2 * delta_alpha_2 * K_i2

            iter_idx += 1

        # Keep only support vectors
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

    def compute_boundaries(self, alpha_1, alpha_2, y_1, y_2) -> Tuple[float, float]:
        """
        Compute the lower and upper bounds for alpha update.

        Returns:
            Tuple[float, float]: (L, H) bounds
        """
        if y_1 == y_2:
            lb = max(0, alpha_1 + alpha_2 - self.c)
            ub = min(self.c, alpha_1 + alpha_2)
        else:
            lb = max(0, alpha_2 - alpha_1)
            ub = min(self.c, self.c + alpha_2 - alpha_1)
        return lb, ub

    def kernel_column(self, i: int) -> np.ndarray:
        """
        Compute the kernel values between a support vector and all support vectors.

        Args:
            i (int): Index of support vector.

        Returns:
            np.ndarray: Kernel column vector
        """
        x_i = self.support_vectors[i]
        return self.kernel(self.support_vectors, x_i)

    def rbf_kernel(self, u, v):
        """
        Compute RBF (Gaussian) kernel.

        Args:
            u (np.ndarray): First input.
            v (np.ndarray): Second input.

        Returns:
            np.ndarray: Kernel matrix
        """
        if np.ndim(v) == 1:
            v = v[np.newaxis, :]
        if np.ndim(u) == 1:
            u = u[np.newaxis, :]
        dist_squared = np.linalg.norm(u[:, :, np.newaxis] - v.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)
        return np.exp(-self.gamma_rbf * dist_squared)

    @staticmethod
    def linear_kernel(u, v) -> np.ndarray:
        """
        Compute linear kernel.

        Args:
            u (np.ndarray): First input.
            v (np.ndarray): Second input.

        Returns:
            np.ndarray: Kernel matrix
        """
        return np.dot(u, v.T)
