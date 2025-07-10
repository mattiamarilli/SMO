import numpy as np
import random

class KernelSVM_SMO:
    def __init__(self, C=1.0, gamma=None, tol=1e-3, max_passes=5, max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
        self._kernel_cache = {}
        self._error_cache = {}  # Cache for error values

    def kernel(self, i, j):
        if (i, j) in self._kernel_cache:
            return self._kernel_cache[(i, j)]
        if (j, i) in self._kernel_cache:
            return self._kernel_cache[(j, i)]
        k = np.exp(-self.gamma * np.sum((self.X[i] - self.X[j]) ** 2))
        self._kernel_cache[(i, j)] = k
        return k

    def decision_function(self, i):
        if self.alpha is None:
            return 0
        result = 0
        idx = np.where(self.alpha > 0)[0]
        for j in idx:
            result += self.alpha[j] * self.y[j] * self.kernel(j, i)
        return result + self.b

    def compute_error(self, i):
        if i in self._error_cache:
            return self._error_cache[i]
        error = self.decision_function(i) - self.y[i]
        self._error_cache[i] = error
        return error

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        if self.gamma is None:
            self.gamma = 1.0 / n_features

        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self._error_cache = {}  # Reset error cache

        passes = 0
        it = 0

        while passes < self.max_passes and it < self.max_iter:
            num_changed_alphas = 0

            # Outer loop heuristic: alternate between full dataset and non-bound examples
            if passes % 2 == 0:
                # Full pass over all examples
                examples_to_examine = range(n_samples)
            else:
                # Pass over non-bound examples (0 < alpha < C)
                examples_to_examine = [i for i in range(n_samples) if 0 < self.alpha[i] < self.C]

            # Shuffle to avoid bias
            examples_to_examine = list(examples_to_examine)
            random.shuffle(examples_to_examine)

            for i in examples_to_examine:
                E_i = self.compute_error(i)
                r_i = E_i * self.y[i]

                # Check KKT conditions
                if ((r_i < -self.tol and self.alpha[i] < self.C) or
                        (r_i > self.tol and self.alpha[i] > 0)):

                    # Second choice heuristic: select j to maximize step size
                    j, E_j = self.select_second_alpha(i, E_i)

                    # Try to take step with selected pair
                    if self.take_step(i, j):
                        num_changed_alphas += 1
                        continue

                    # If no progress, try random non-bound examples
                    non_bound_indices = [idx for idx in range(n_samples)
                                         if 0 < self.alpha[idx] < self.C]
                    if len(non_bound_indices) > 1:
                        random.shuffle(non_bound_indices)
                        for j in non_bound_indices:
                            if j == i:
                                continue
                            if self.take_step(i, j):
                                num_changed_alphas += 1
                                break

                    # If still no progress, try random examples from entire dataset
                    if num_changed_alphas == 0:
                        random_indices = list(range(n_samples))
                        random.shuffle(random_indices)
                        for j in random_indices:
                            if j == i:
                                continue
                            if self.take_step(i, j):
                                num_changed_alphas += 1
                                break

                    # If no progress at all, skip this i
                    if num_changed_alphas == 0:
                        continue

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

            it += 1

        self.support_ = np.where(self.alpha > 1e-8)[0]
        self.support_vectors_ = self.X[self.support_]
        self.support_alpha = self.alpha[self.support_]
        self.support_y = self.y[self.support_]

        return self

    def select_second_alpha(self, i, E_i):
        """Select the second alpha using the heuristic to maximize step size."""
        max_delta = 0
        j = -1
        best_E_j = 0

        # First try non-bound examples with maximum |E_i - E_j|
        non_zero_C = [idx for idx in range(len(self.alpha))
                      if 0 < self.alpha[idx] < self.C]

        if len(non_zero_C) > 1:
            # Find the example with maximum |E_i - E_j|
            for idx in non_zero_C:
                if idx == i:
                    continue
                E_j = self.compute_error(idx)
                delta = abs(E_i - E_j)
                if delta > max_delta:
                    max_delta = delta
                    j = idx
                    best_E_j = E_j
            if j >= 0:
                return j, best_E_j

        # If no good non-bound examples found, select from entire dataset
        for idx in range(len(self.alpha)):
            if idx == i:
                continue
            E_j = self.compute_error(idx)
            delta = abs(E_i - E_j)
            if delta > max_delta:
                max_delta = delta
                j = idx
                best_E_j = E_j

        return j, best_E_j

    def take_step(self, i, j):
        if i == j:
            return False

        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]
        y_i = self.y[i]
        y_j = self.y[j]
        E_i = self.compute_error(i)
        E_j = self.compute_error(j)
        s = y_i * y_j

        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)

        if L == H:
            return False

        k_ii = self.kernel(i, i)
        k_jj = self.kernel(j, j)
        k_ij = self.kernel(i, j)

        eta = k_ii + k_jj - 2 * k_ij
        if eta <= 0:
            return False

        alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)

        if abs(alpha_j_new - alpha_j) < 1e-5 * (alpha_j_new + alpha_j + 1e-5):
            return False

        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)

        b1 = self.b - E_i - y_i * (alpha_i_new - alpha_i) * k_ii - y_j * (alpha_j_new - alpha_j) * k_ij
        b2 = self.b - E_j - y_i * (alpha_i_new - alpha_i) * k_ij - y_j * (alpha_j_new - alpha_j) * k_jj

        if 0 < alpha_i_new < self.C:
            b_new = b1
        elif 0 < alpha_j_new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        # Update alpha and b
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        self.b = b_new

        # Update error cache for i and j
        self._error_cache[i] = self.decision_function(i) - self.y[i]
        self._error_cache[j] = self.decision_function(j) - self.y[j]

        return True

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            s = 0
            for alpha_i, y_i, x_i in zip(self.support_alpha, self.support_y, self.support_vectors_):
                s += alpha_i * y_i * np.exp(-self.gamma * np.sum((x - x_i) ** 2))
            y_pred[idx] = s + self.b
        return np.sign(y_pred)