from typing import Tuple

import numpy as np

class SVM:
    def __init__(self, c: float = 1.,kkt_thr: float = 1e-3, max_iter: int = 1e4, kernel_type: str = 'linear', gamma_rbf: float = 1.) -> None:
        if not kernel_type in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))

        super().__init__()

        self.c = float(c)
        self.max_iter = max_iter
        self.kkt_thr = kkt_thr
        if kernel_type == 'linear':
            self.kernel = self.linear_kernel
        elif kernel_type == 'rbf':
            self.kernel = self.rbf_kernel
            self.gamma_rbf = gamma_rbf

        self.b = np.array([])
        self.alpha = np.array([])
        self.support_vectors = np.array([])
        self.support_labels = np.array([])

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = self.support_labels * self.alpha
        x = self.kernel(self.support_vectors, x)

        scores = np.matmul(w, x) + self.b
        pred = np.sign(scores)

        return pred, scores

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train

        iter_idx = 0
        non_kkt_array = np.arange(N)
        error_cache = self.predict(x_train)[1] - y_train

        print("SVM training using SMO algorithm - START")
        while iter_idx < self.max_iter:
            i_2, non_kkt_array = self.i2_heuristic(non_kkt_array)
            if i_2 == -1:
                break

            i_1 = self.i1_heuristic(i_2, error_cache)

            if i_1 == i_2:
                continue

            x_1, y_1, alpha_1 = self.support_vectors[i_1, :], self.support_labels[i_1], self.alpha[i_1]
            x_2, y_2, alpha_2 = self.support_vectors[i_2, :], self.support_labels[i_2], self.alpha[i_2]

            L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)
            if L == H:
                continue

            eta = self.compute_eta(x_1, x_2)
            if eta == 0:
                continue

            _, score_1 = self.predict(x_1)
            _, score_2 = self.predict(x_2)

            E_1 = score_1 - y_1
            E_2 = score_2 - y_2

            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            alpha_2_new = np.minimum(alpha_2_new, H)
            alpha_2_new = np.maximum(alpha_2_new, L)

            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)

            self.compute_b(alpha_1_new, alpha_2_new, E_1, E_2, i_1, i_2)

            self.alpha[i_1] = alpha_1_new
            self.alpha[i_2] = alpha_2_new

            error_cache[i_1] = self.predict(x_1)[1] - y_1
            error_cache[i_2] = self.predict(x_2)[1] - y_2

            iter_idx = iter_idx + 1

        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        print(f"Training summary: {iter_idx} iterations, {self.alpha.shape[0]} supprts vectors")
        print("SVM training using SMO algorithm - DONE!")

    def i1_heuristic(self, i_2, error_cache) -> int:
        E_2 = error_cache[i_2]
        non_bounded_idx = np.argwhere((0 < self.alpha) & (self.alpha < self.c)).reshape((1, -1))[0]

        if non_bounded_idx.shape[0] > 0:
            if E_2 >= 0:
                i_1 = non_bounded_idx[np.argmin(error_cache[non_bounded_idx])]
            else:
                i_1 = non_bounded_idx[np.argmax(error_cache[non_bounded_idx])]
        else:
            i_1 = np.argmax(np.abs(error_cache - E_2))

        return i_1

    def i2_heuristic(self, non_kkt_array: np.ndarray) -> int:
        i_2 = -1
        for idx in non_kkt_array:
            if not self.check_kkt(idx):
                i_2 = idx
                break

        if i_2 == -1:
            idx_array = np.arange(self.alpha.shape[0])
            non_kkt_array = idx_array[~(self.check_kkt(idx_array))]
            if non_kkt_array.shape[0] > 0:
                np.random.shuffle(non_kkt_array)
                i_2 = non_kkt_array[0]
                non_kkt_array = non_kkt_array[1:-1]

        return i_2, non_kkt_array

    def check_kkt(self, check_idx: int) -> np.ndarray:
        alpha_idx = self.alpha[check_idx]
        _, score_idx = self.predict(self.support_vectors[check_idx, :])
        y_idx = self.support_labels[check_idx]
        r_idx = y_idx * score_idx - 1
        cond_1 = (alpha_idx < self.c) & (r_idx < - self.kkt_thr)
        cond_2 = (alpha_idx > 0) & (r_idx > self.kkt_thr)

        return ~(cond_1 | cond_2)

    def compute_boundaries(self, alpha_1, alpha_2, y_1, y_2) -> Tuple[float, float]:
        if y_1 == y_2:
            lb = np.max([0, alpha_1 + alpha_2 - self.c])
            ub = np.min([self.c, alpha_1 + alpha_2])
        else:
            lb = np.max([0, alpha_2 - alpha_1])
            ub = np.min([self.c, self.c + alpha_2 - alpha_1])
        return lb, ub

    def compute_eta(self, x_1, x_2) -> float:
        return self.kernel(x_1, x_1) + self.kernel(x_2, x_2) - 2 * self.kernel(x_1, x_2)

    def compute_b(self, alpha_1_new, alpha_2_new, E_1, E_2, i_1, i_2) -> None:
        x_1 = self.support_vectors[i_1]
        x_2 = self.support_vectors[i_2]

        b1 = self.b - E_1 - self.support_labels[i_1] * (alpha_1_new - self.alpha[i_1]) * self.kernel(x_1, x_1) - \
             self.support_labels[i_2] * (alpha_2_new - self.alpha[i_2]) * self.kernel(x_1, x_2)

        b2 = self.b - E_2 - self.support_labels[i_1] * (alpha_1_new - self.alpha[i_1]) * self.kernel(x_1, x_2) - \
             self.support_labels[i_2] * (alpha_2_new - self.alpha[i_2]) * self.kernel(x_2, x_2)

        if 0 < alpha_1_new < self.c:
            self.b = b1
        elif 0 < alpha_2_new < self.c:
            self.b = b2
        else:
            self.b = np.mean([b1, b2])

    def rbf_kernel(self, u, v):
        if np.ndim(v) == 1:
            v = v[np.newaxis, :]

        if np.ndim(u) == 1:
            u = u[np.newaxis, :]

        dist_squared = np.linalg.norm(u[:, :, np.newaxis] - v.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)

        return np.exp(-self.gamma_rbf * dist_squared)

    @staticmethod
    def linear_kernel(u,v) -> np.ndarray:
        return np.dot(u, v.T)