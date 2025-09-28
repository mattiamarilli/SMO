from typing import Tuple
import numpy as np
import time


class SVM:
    def __init__(self, c: float = 1., kkt_thr: float = 1e-3, max_iter: int = 1e4, kernel_type: str = 'linear', gamma_rbf: float = 1.) -> None:
        if kernel_type not in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))

        super().__init__()

        self.c = float(c)
        self.max_iter = int(max_iter)
        self.kkt_thr = kkt_thr

        if kernel_type == 'linear':
            self.kernel = self.linear_kernel
        elif kernel_type == 'rbf':
            self.kernel = self.rbf_kernel
            self.gamma_rbf = gamma_rbf

        self.b = 0.0
        self.alpha = np.array([])
        self.support_vectors = np.array([])
        self.support_labels = np.array([])

    # ------------------- PREDICT -------------------
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.alpha.shape[0] == 0:
            raise ValueError("Model not trained yet.")
        K_test = self.kernel(self.support_vectors, x)
        scores = np.dot(self.alpha * self.support_labels, K_test) + self.b
        pred = np.sign(scores)
        return pred, scores

    def mvp_heuristic(self, error_cache: np.ndarray) -> Tuple[int, int]:
        alpha = self.alpha
        y = self.support_labels
        C = self.c

        L_indices = (alpha == 0)
        U_indices = (alpha == C)
        Free_indices = (alpha > 0) & (alpha < C)

        R_mask = (L_indices & (y == 1)) | (U_indices & (y == -1)) | Free_indices
        S_mask = (L_indices & (y == -1)) | (U_indices & (y == 1)) | Free_indices

        R_indices = np.where(R_mask)[0]
        S_indices = np.where(S_mask)[0]

        # filtrare fuori i NaN
        R_indices = [i for i in R_indices if not np.isnan(error_cache[i])]
        S_indices = [i for i in S_indices if not np.isnan(error_cache[i])]

        if len(R_indices) == 0 or len(S_indices) == 0:
            return -1, -1

        i_mvp = R_indices[np.argmin(error_cache[R_indices])]
        j_mvp = S_indices[np.argmax(error_cache[S_indices])]

        return i_mvp, j_mvp

    # ------------------- FIT -------------------
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0.0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train
        self.sumtimes = 0

        iter_idx = 0
        non_kkt_array = np.arange(N)
        start_time_cache = time.time()
        #error_cache = self.predict(x_train)[1] - y_train
        error_cache = np.zeros(N)
        for i in range(N):
            K_i = self.kernel_column(i)  # calcola solo K[:, i]
            error_cache[i] = np.dot(self.alpha * self.support_labels, K_i) + self.b - y_train[i]
        end_time_cache = time.time()

        print("SVM training using SMO algorithm - START")
        start_time_fit = time.time()

        while iter_idx < self.max_iter:
            # if iter_idx % 50 == 0:
            #     for i in range(N):
            #         K_i = self.kernel_column(i)
            #         error_cache[i] = np.dot(self.alpha * self.support_labels, K_i) + self.b - y_train[i]

            i_2, i_1 = self.mvp_heuristic(error_cache)  # Seleziona i_MVP e j_MVP

            if i_2 == -1 or i_1 == -1:
                break

            if i_1 == i_2:
                continue

            y_1, alpha_1 = self.support_labels[i_1], self.alpha[i_1]
            y_2, alpha_2 = self.support_labels[i_2], self.alpha[i_2]

            # --- Precalcolo colonne kernel ---
            start_time = time.time()
            K_i1 = self.kernel_column(i_1)
            K_i2 = self.kernel_column(i_2)
            self.sumtimes += time.time() - start_time
            k11 = K_i1[i_1]
            k22 = K_i2[i_2]
            k12 = K_i1[i_2]

            # --- Calcolo boundaries ---
            L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)

            # --- Calcolo eta ---
            eta = k11 + k22 - 2 * k12
            if eta < 1e-12:
                continue

            # --- Calcolo errori ---
            E_1 = np.dot(self.alpha * self.support_labels, K_i1) + self.b - y_1
            E_2 = np.dot(self.alpha * self.support_labels, K_i2) + self.b - y_2

            # --- Aggiornamento alpha ---
            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            alpha_2_new = np.clip(alpha_2_new, L, H)
            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)

            # --- Aggiornamento b ---
            b1 = self.b - E_1 - y_1 * (alpha_1_new - alpha_1) * k11 - y_2 * (alpha_2_new - alpha_2) * k12
            b2 = self.b - E_2 - y_1 * (alpha_1_new - alpha_1) * k12 - y_2 * (alpha_2_new - alpha_2) * k22

            if 0 < alpha_1_new < self.c:
                self.b = b1
            elif 0 < alpha_2_new < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            # --- Salvataggio alpha aggiornati ---
            self.alpha[i_1] = alpha_1_new
            self.alpha[i_2] = alpha_2_new

            # --- Aggiornamento cache errori ---
            delta_alpha_1 = alpha_1_new - alpha_1
            delta_alpha_2 = alpha_2_new - alpha_2
            error_cache += y_1 * delta_alpha_1 * K_i1 + y_2 * delta_alpha_2 * K_i2

            iter_idx += 1


        # --- Filtraggio support vectors ---
        end_time_fit = time.time()
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        print(f"Training summary: {iter_idx} iterations");
        print(f"Tempo calcolo colonne interno: {self.sumtimes }")
        print(f"Tempo calcolo colonne per cache iniziale: {end_time_cache - start_time_cache}")
        print(f"Tempo calcolo colonne totale: {self.sumtimes + (end_time_cache - start_time_cache)}")
        print("SVM training using SMO algorithm - DONE!")


    # ------------------- HEURISTICS -------------------
    def i1_heuristic(self, i_2, error_cache) -> int:
        E_2 = error_cache[i_2]
        non_bounded_idx = np.argwhere((0 < self.alpha) & (self.alpha < self.c)).reshape(-1)

        if non_bounded_idx.size > 0:
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
            non_kkt_array = np.delete(non_kkt_array, np.argwhere(non_kkt_array == idx))
            if not self.check_kkt(idx):
                i_2 = idx
                break

        if i_2 == -1:
            idx_array = np.arange(self.alpha.shape[0])
            non_kkt_array = idx_array[~(self.check_kkt(idx_array))]
            if non_kkt_array.size > 0:
                np.random.shuffle(non_kkt_array)
                i_2 = non_kkt_array[0]
                non_kkt_array = non_kkt_array[1:-1]

        return i_2, non_kkt_array

    # ------------------- KKT CHECK -------------------
    def check_kkt(self, check_idx: int) -> np.ndarray:
        alpha_idx = self.alpha[check_idx]
        _, score_idx = self.predict(self.support_vectors[check_idx, :])
        y_idx = self.support_labels[check_idx]
        r_idx = y_idx * score_idx - 1
        cond_1 = (alpha_idx < self.c) & (r_idx < - self.kkt_thr)
        cond_2 = (alpha_idx > 0) & (r_idx > self.kkt_thr)
        return ~(cond_1 | cond_2)

    # ------------------- BOUNDS -------------------
    def compute_boundaries(self, alpha_1, alpha_2, y_1, y_2) -> Tuple[float, float]:
        if y_1 == y_2:
            lb = max(0, alpha_1 + alpha_2 - self.c)
            ub = min(self.c, alpha_1 + alpha_2)
        else:
            lb = max(0, alpha_2 - alpha_1)
            ub = min(self.c, self.c + alpha_2 - alpha_1)
        return lb, ub

    # ------------------- KERNEL COLUMN -------------------
    def kernel_column(self, i: int) -> np.ndarray:
        x_i = self.support_vectors[i]
        return self.kernel(self.support_vectors, x_i)

    # ------------------- KERNELS -------------------
    def rbf_kernel(self, u, v):
        if np.ndim(v) == 1:
            v = v[np.newaxis, :]
        if np.ndim(u) == 1:
            u = u[np.newaxis, :]
        dist_squared = np.linalg.norm(u[:, :, np.newaxis] - v.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)
        return np.exp(-self.gamma_rbf * dist_squared)

    @staticmethod
    def linear_kernel(u, v) -> np.ndarray:
        return np.dot(u, v.T)
