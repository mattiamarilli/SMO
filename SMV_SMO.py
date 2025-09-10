import numpy as np

class SMO_SVM:
    def __init__(self, X, y, C, kernel_func=None, tol=1e-3, max_iter=1000):
        """
        Inizializza l'implementazione SMO per SVM senza memorizzare Q.

        Args:
            X (np.array): Matrice dei dati di input (N_samples, N_features).
            y (np.array): Vettore delle etichette (+1 o -1) (N_samples,).
            C (float): Parametro di regolarizzazione (box constraint per alpha).
            kernel_func (callable, optional): Funzione kernel. Se None, usa kernel lineare.
            tol (float): Tolleranza per il criterio di arresto.
            max_iter (int): Numero massimo di iterazioni.
        """
        self.X = X
        self.y = y
        self.C = C
        self.kernel_func = kernel_func if kernel_func is not None else self._linear_kernel
        self.tol = tol
        self.max_iter = max_iter
        self.N = X.shape[0]

        self.alpha = np.zeros(self.N)
        self.b = 0.0
        self.gradients = -np.ones(self.N)

    def _linear_kernel(self, u, v):
        return np.dot(u, v)

    def _Q(self, i, j):
        """Calcola Q_ij al volo: Q_ij = y_i * y_j * k(x_i, x_j)"""
        return self.y[i] * self.y[j] * self.kernel_func(self.X[i], self.X[j])

    def _update_gradients_after_step(self, i, j, alpha_i_old, alpha_j_old):
        alpha_i_new = self.alpha[i]
        alpha_j_new = self.alpha[j]
        for k in range(self.N):
            self.gradients[k] += (alpha_i_new - alpha_i_old) * self._Q(k, i) + \
                                 (alpha_j_new - alpha_j_old) * self._Q(k, j)

    def _select_mvp(self):
        L_alpha_idx = np.where(self.alpha == 0)[0]
        U_alpha_idx = np.where(self.alpha == self.C)[0]
        Free_alpha_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]

        L_plus_idx = L_alpha_idx[np.where(self.y[L_alpha_idx] == 1)]
        L_minus_idx = L_alpha_idx[np.where(self.y[L_alpha_idx] == -1)]
        U_plus_idx = U_alpha_idx[np.where(self.y[U_alpha_idx] == 1)]
        U_minus_idx = U_alpha_idx[np.where(self.y[U_alpha_idx] == -1)]

        R_alpha_indices = np.concatenate((L_plus_idx, U_minus_idx, Free_alpha_idx))
        S_alpha_indices = np.concatenate((L_minus_idx, U_plus_idx, Free_alpha_idx))

        if len(R_alpha_indices) == 0 or len(S_alpha_indices) == 0:
            return None, None

        neg_grad_y_ratio = -self.gradients / self.y

        i_star_idx_in_R = np.argmax(neg_grad_y_ratio[R_alpha_indices])
        i_star = R_alpha_indices[i_star_idx_in_R]

        j_star_idx_in_S = np.argmin(neg_grad_y_ratio[S_alpha_indices])
        j_star = S_alpha_indices[j_star_idx_in_S]

        M_alpha = neg_grad_y_ratio[i_star]
        m_alpha = neg_grad_y_ratio[j_star]

        if M_alpha - m_alpha <= self.tol:
            return None, None

        return i_star, j_star

    def _solve_2_variable_subproblem(self, i, j):
        if i == j:
            return self.alpha[i], self.alpha[j]

        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
        y_i, y_j = self.y[i], self.y[j]
        Q_ii, Q_jj, Q_ij = self._Q(i, i), self._Q(j, j), self._Q(i, j)
        s = y_i * y_j

        grad_i, grad_j = self.gradients[i], self.gradients[j]

        if s == 1:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        else:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)

        if L == H:
            return alpha_i_old, alpha_j_old

        eta = Q_ii + Q_jj - 2 * Q_ij
        if eta <= 0:
            return alpha_i_old, alpha_j_old

        alpha_j_new_unclipped = alpha_j_old + y_j * ((grad_i / y_i) - (grad_j / y_j)) / eta
        alpha_j_new = np.clip(alpha_j_new_unclipped, L, H)

        if np.abs(alpha_j_new - alpha_j_old) < 1e-5:
            return alpha_i_old, alpha_j_old

        alpha_i_new = alpha_i_old + s * (alpha_j_old - alpha_j_new)
        return alpha_i_new, alpha_j_new

    def _get_kernel_row(self, k_idx):
        return np.array([self.kernel_func(self.X[k_idx], self.X[j]) for j in range(self.N)])

    def fit(self):
        iteration = 0
        while iteration < self.max_iter:
            i_star, j_star = self._select_mvp()
            if i_star is None:
                print(f"Convergenza raggiunta dopo {iteration} iterazioni.")
                break

            alpha_i_old, alpha_j_old = self.alpha[i_star], self.alpha[j_star]
            alpha_i_new, alpha_j_new = self._solve_2_variable_subproblem(i_star, j_star)

            if np.abs(alpha_i_new - alpha_i_old) < 1e-5 and \
               np.abs(alpha_j_new - alpha_j_old) < 1e-5:
                iteration += 1
                continue

            self.alpha[i_star] = alpha_i_new
            self.alpha[j_star] = alpha_j_new

            self._update_gradients_after_step(i_star, j_star, alpha_i_old, alpha_j_old)

            non_bound_alphas_idx = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]
            if len(non_bound_alphas_idx) > 0:
                k_for_b = non_bound_alphas_idx[0]
                w_dot_x_k = np.sum(self.alpha * self.y * self._get_kernel_row(k_for_b))
                self.b = self.y[k_for_b] - w_dot_x_k

            iteration += 1

        if iteration >= self.max_iter:
            print(f"Raggiunto il numero massimo di iterazioni ({self.max_iter}).")

    def decision_function(self, X_test):
        if self.alpha is None or self.N == 0:
            raise RuntimeError("Il modello non è stato ancora addestrato o i dati di addestramento sono vuoti.")

        scores = np.zeros(X_test.shape[0])
        for i, x_test_sample in enumerate(X_test):
            kernel_values = np.array([self.kernel_func(self.X[k], x_test_sample) for k in range(self.N)])
            scores[i] = np.sum(self.alpha * self.y * kernel_values) + self.b
        return scores

    def predict(self, X_test):
        scores = self.decision_function(X_test)
        return np.sign(scores)
