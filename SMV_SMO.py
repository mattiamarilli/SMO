import numpy as np

class SMO_SVM:
    def __init__(self, X, y, C, kernel_func=None, tol=1e-5, max_iter=1000, verbose=False):
        # Store training data, labels, regularization parameter C, and kernel
        # If no kernel is provided, default is linear: K(u,v) = u·v
        self.X = X
        self.y = y
        self.C = C
        self.kernel_func = kernel_func if kernel_func is not None else self._linear_kernel
        self.tol = tol                   # Numerical tolerance for stopping criterion
        self.max_iter = max_iter         # Maximum number of iterations
        self.N = X.shape[0]              # Number of training samples
        self.verbose = verbose

        # Initialize Lagrange multipliers α_i to 0
        self.alpha = np.zeros(self.N)
        # Initialize bias term b = 0
        self.b = 0.0
        # Initialize gradients (derivatives of dual objective w.r.t α_i)
        # At start, ∇W(α) = -1 for all i
        self.gradients = -np.ones(self.N)

    def _linear_kernel(self, u, v):
        # Default linear kernel: K(u,v) = u·v
        return np.dot(u, v)

    def _Q(self, i, j):
        # Computes Q_ij = y_i * y_j * K(x_i, x_j)
        # This is the Hessian entry of the dual optimization problem
        return self.y[i] * self.y[j] * self.kernel_func(self.X[i], self.X[j])

    def _update_gradients_after_step(self, i, j, alpha_i_old, alpha_j_old):
        # After updating α_i and α_j, the gradients for all α_k must be updated
        # ∇_k W(α) changes according to (Δα_i * Q_ki + Δα_j * Q_kj)
        alpha_i_new = self.alpha[i]
        alpha_j_new = self.alpha[j]
        for k in range(self.N):
            self.gradients[k] += (alpha_i_new - alpha_i_old) * self._Q(k, i) + \
                                 (alpha_j_new - alpha_j_old) * self._Q(k, j)

    def _select_mvp(self):
        # MVP = Most Violating Pair
        # Chooses (i*, j*) that maximally violates KKT conditions

        # Split α into sets:
        # - L set: α = 0
        # - U set: α = C
        # - Free set: 0 < α < C
        L_alpha_idx = np.where(self.alpha == 0)[0]
        U_alpha_idx = np.where(self.alpha == self.C)[0]
        Free_alpha_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]

        # Partition indices further by label sign
        L_plus_idx = L_alpha_idx[np.where(self.y[L_alpha_idx] == 1)]
        L_minus_idx = L_alpha_idx[np.where(self.y[L_alpha_idx] == -1)]
        U_plus_idx = U_alpha_idx[np.where(self.y[U_alpha_idx] == 1)]
        U_minus_idx = U_alpha_idx[np.where(self.y[U_alpha_idx] == -1)]

        # Construct R and S sets (see SMO algorithm derivation)
        R_alpha_indices = np.concatenate((L_plus_idx, U_minus_idx, Free_alpha_idx))
        S_alpha_indices = np.concatenate((L_minus_idx, U_plus_idx, Free_alpha_idx))

        if len(R_alpha_indices) == 0 or len(S_alpha_indices) == 0:
            if self.verbose:
                print("[MVP] No valid candidate for R or S.")
            return None, None

        # Negative gradient scaled by labels: -∇W(α)/y
        neg_grad_y_ratio = -self.gradients / self.y

        # i* = argmax over R, j* = argmin over S
        i_star = R_alpha_indices[np.argmax(neg_grad_y_ratio[R_alpha_indices])]
        j_star = S_alpha_indices[np.argmin(neg_grad_y_ratio[S_alpha_indices])]

        M_alpha = neg_grad_y_ratio[i_star]
        m_alpha = neg_grad_y_ratio[j_star]
        if self.verbose:
            print(f"[MVP] M={M_alpha:.6f}, m={m_alpha:.6f}, i={i_star}, j={j_star}, α_i={self.alpha[i_star]:.6f}, α_j={self.alpha[j_star]:.6f}")

        # If gap is smaller than tolerance, stop (KKT satisfied)
        if M_alpha - m_alpha <= self.tol:
            if self.verbose:
                print("[MVP] No significant violation. Convergence.")
            return None, None

        return i_star, j_star

    def _solve_2_variable_subproblem(self, i, j):
        # Solves optimization restricted to two variables α_i and α_j
        if i == j:
            if self.verbose:
                print("[SUBPROBLEM] Same index, no update.")
            return self.alpha[i], self.alpha[j]

        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
        y_i, y_j = self.y[i], self.y[j]
        Q_ii, Q_jj, Q_ij = self._Q(i, i), self._Q(j, j), self._Q(i, j)
        s = y_i * y_j

        grad_i, grad_j = self.gradients[i], self.gradients[j]

        # Compute feasible interval [L,H] for α_j
        if s == 1:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        else:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)

        if self.verbose:
            print(f"[SUBPROBLEM] L={L:.6f}, H={H:.6f}")

        if L == H:
            if self.verbose:
                print("[SUBPROBLEM] L == H, skipping update.")
            return alpha_i_old, alpha_j_old

        # Compute eta = Q_ii + Q_jj - 2Q_ij (curvature of objective in 2D subspace)
        eta = Q_ii + Q_jj - 2 * Q_ij
        if self.verbose:
            print(f"[SUBPROBLEM] eta={eta:.6f}")
        if eta < self.tol:
            if self.verbose:
                print("[SUBPROBLEM] eta too small, skipping update.")
            return alpha_i_old, alpha_j_old

        # Compute new α_j before clipping
        alpha_j_new_unclipped = alpha_j_old + y_j * ((grad_i / y_i) - (grad_j / y_j)) / eta
        if self.verbose:
            print(f"[SUBPROBLEM] α_j (unclipped): {alpha_j_new_unclipped:.6f}")

        # Clip α_j within [L,H]
        alpha_j_new = np.clip(alpha_j_new_unclipped, L, H)

        # If change is too small, skip
        if np.abs(alpha_j_new - alpha_j_old) < self.tol:
            if self.verbose:
                print("[SUBPROBLEM] Δα_j too small, skipping update.")
            return alpha_i_old, alpha_j_old

        # Update α_i based on constraint y_i α_i + y_j α_j = constant
        alpha_i_new = alpha_i_old + s * (alpha_j_old - alpha_j_new)
        return alpha_i_new, alpha_j_new

    def _get_kernel_row(self, k_idx):
        # Compute the kernel vector K(x_k, X) for a given index k
        return np.array([self.kernel_func(self.X[k_idx], self.X[j]) for j in range(self.N)])

    def fit(self):
        # Train the SVM by iteratively selecting violating pairs and solving subproblems
        iteration = 0
        no_progress_count = 0
        max_no_progress = 50

        while iteration < self.max_iter:
            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")
            i_star, j_star = self._select_mvp()
            if i_star is None:
                if self.verbose:
                    print(f"[STOP] Convergence reached after {iteration} iterations.")
                break

            alpha_i_old, alpha_j_old = self.alpha[i_star], self.alpha[j_star]
            alpha_i_new, alpha_j_new = self._solve_2_variable_subproblem(i_star, j_star)

            # Changes in α values
            delta_i = np.abs(alpha_i_new - alpha_i_old)
            delta_j = np.abs(alpha_j_new - alpha_j_old)
            if self.verbose:
                print(f"[UPDATE] Δα_i={delta_i:.6f}, Δα_j={delta_j:.6f}")

            # If both updates are negligible, increment stall counter
            if delta_i < self.tol and delta_j < self.tol:
                no_progress_count += 1
                if self.verbose:
                    print(f"[SKIP] No significant update. Stall count: {no_progress_count}")
                if no_progress_count >= max_no_progress:
                    if self.verbose:
                        print("[STOP] Stalling detected. Forced stop.")
                    break
                iteration += 1
                continue
            else:
                no_progress_count = 0

            # Update α values
            self.alpha[i_star] = alpha_i_new
            self.alpha[j_star] = alpha_j_new

            # Update gradients efficiently
            self._update_gradients_after_step(i_star, j_star, alpha_i_old, alpha_j_old)

            # Update bias term b using one free α
            non_bound_alphas_idx = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]
            if len(non_bound_alphas_idx) > 0:
                k_for_b = non_bound_alphas_idx[0]
                w_dot_x_k = np.sum(self.alpha * self.y * self._get_kernel_row(k_for_b))
                self.b = self.y[k_for_b] - w_dot_x_k
                if self.verbose:
                    print(f"[BIAS] Updated b: {self.b:.6f}")
            else:
                if self.verbose:
                    print("[BIAS] No free α to update b.")

            iteration += 1

        if iteration >= self.max_iter:
            if self.verbose:
                print(f"[STOP] Max iterations reached ({self.max_iter}).")

    def decision_function(self, X_test):
        # Compute raw decision function f(x) = Σ α_i y_i K(x_i, x) + b
        if self.alpha is None or self.N == 0:
            raise RuntimeError("Model not trained or empty data.")

        scores = np.zeros(X_test.shape[0])
        for i, x_test_sample in enumerate(X_test):
            kernel_values = np.array([self.kernel_func(self.X[k], x_test_sample) for k in range(self.N)])
            scores[i] = np.sum(self.alpha * self.y * kernel_values) + self.b
        return scores

    def predict(self, X_test):
        # Predict class labels: sign(f(x))
        scores = self.decision_function(X_test)
        return np.sign(scores)
