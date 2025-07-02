import numpy as np

class KernelSVM_SMO:
    def __init__(self, C=1.0, gamma=None, tol=1e-3, max_passes=5, max_iter=10000):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.errors = None  # cache E_i = f(x_i) - y_i
        self._kernel_cache = {}

    def kernel(self, i, j):
        # Evaluate kernel K(x_i, x_j) on-demand with caching
        if (i, j) in self._kernel_cache:
            return self._kernel_cache[(i, j)]
        if (j, i) in self._kernel_cache:
            return self._kernel_cache[(j, i)]
        k = np.exp(-self.gamma * np.sum((self.X[i] - self.X[j]) ** 2))
        self._kernel_cache[(i, j)] = k
        return k

    def decision_function(self, i):
        # f(x_i) = sum_j alpha_j y_j K(x_j, x_i) + b
        # Efficiently compute with alpha and kernel on-demand
        if self.alpha is None:
            return 0
        result = 0
        # Compute only for alpha_j > 0 for speed
        idx = np.where(self.alpha > 0)[0]
        for j in idx:
            result += self.alpha[j] * self.y[j] * self.kernel(j, i)
        return result + self.b

    def compute_error(self, i):
        return self.decision_function(i) - self.y[i]

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        if self.gamma is None:
            self.gamma = 1.0 / n_features

        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self.errors = np.array([-y[i] for i in range(n_samples)], dtype=float)

        passes = 0
        it = 0

        while passes < self.max_passes and it < self.max_iter:
            num_changed_alphas = 0
            # Working set selection: find Most Violating Pair (i, j)
            # We follow Platt's original logic but with MVP heuristic:
            # Pick i from violators: alpha_i violates KKT most
            # For simplicity, scan all, but more efficient heuristics possible
            violators = []
            for i in range(n_samples):
                E_i = self.compute_error(i)
                r_i = E_i * self.y[i]
                if ((r_i < -self.tol and self.alpha[i] < self.C) or
                    (r_i > self.tol and self.alpha[i] > 0)):
                    violators.append(i)

            if len(violators) == 0:
                passes += 1
                it += 1
                continue

            # Find pair (i,j) that maximizes violation |E_i - E_j|
            max_delta = 0
            i_pair = -1
            j_pair = -1

            for i in violators:
                E_i = self.compute_error(i)
                for j in range(n_samples):
                    if i == j:
                        continue
                    E_j = self.compute_error(j)
                    delta = abs(E_i - E_j)
                    if delta > max_delta:
                        max_delta = delta
                        i_pair, j_pair = i, j

            if i_pair == -1 or j_pair == -1:
                passes += 1
                it += 1
                continue

            # Optimize pair (i_pair, j_pair)
            if self.take_step(i_pair, j_pair):
                num_changed_alphas += 1
            else:
                passes += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            it += 1

        # After training, store support vectors info
        self.support_ = np.where(self.alpha > 1e-8)[0]
        self.support_vectors_ = self.X[self.support_]
        self.support_alpha = self.alpha[self.support_]
        self.support_y = self.y[self.support_]

        return self

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

        # Compute L and H (box constraints for alpha_j)
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)

        if L == H:
            return False

        # Compute kernel values
        k_ii = self.kernel(i, i)
        k_jj = self.kernel(j, j)
        k_ij = self.kernel(i, j)

        eta = k_ii + k_jj - 2 * k_ij
        if eta <= 0:
            return False

        # Compute new alpha_j
        alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
        # Clip alpha_j_new within L and H
        alpha_j_new = np.clip(alpha_j_new, L, H)
        if abs(alpha_j_new - alpha_j) < 1e-5 * (alpha_j_new + alpha_j + 1e-5):
            return False

        # Compute new alpha_i
        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)

        # Update threshold b
        b1 = self.b - E_i - y_i * (alpha_i_new - alpha_i) * k_ii - y_j * (alpha_j_new - alpha_j) * k_ij
        b2 = self.b - E_j - y_i * (alpha_i_new - alpha_i) * k_ij - y_j * (alpha_j_new - alpha_j) * k_jj

        if 0 < alpha_i_new < self.C:
            b_new = b1
        elif 0 < alpha_j_new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        # Update model parameters
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        self.b = b_new

        return True

    def predict(self, X):
        # Predict labels for new data points
        y_pred = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            s = 0
            for alpha_i, y_i, x_i in zip(self.support_alpha, self.support_y, self.support_vectors_):
                s += alpha_i * y_i * np.exp(-self.gamma * np.sum((x - x_i) ** 2))
            y_pred[idx] = s + self.b
        return np.sign(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
