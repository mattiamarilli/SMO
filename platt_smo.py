import numpy as np
from random import randrange

class SmoAlgorithm:
    def __init__(self, X, y, C, tol, kernel):
        self.X = X
        self.y = y
        self.m, self.n = np.shape(self.X)
        self.alphas = np.zeros(self.m)

        self.kernel = kernel
        self.C = C
        self.tol = tol

        self.errors = np.zeros(self.m)
        self.eps = 1e-3

        self.b = 0
        self.w = np.zeros(self.n)

    def output(self, x):
        if isinstance(x, (int, np.integer)):
                return np.sum([
                    self.alphas[j] * self.y[j] * self.kernel(self.X[j], self.X[x])
                    for j in range(self.m)
                ]) - self.b
        else:
                return np.sum([
                    self.alphas[j] * self.y[j] * self.kernel(self.X[j], x)
                    for j in range(self.m)
                ]) - self.b

    def take_step(self, i1, i2):
        if i1 == i2:
            return False

        a1 = self.alphas[i1]
        y1 = self.y[i1]
        X1 = self.X[i1]
        E1 = self.get_error(i1)

        s = y1 * self.y2

        # Determine the bounds for alpha2 (Equation 13 and 14)
        if y1 != self.y2:
            L = max(0, self.a2 - a1)
            H = min(self.C, self.C + self.a2 - a1)
        else:
            L = max(0, self.a2 + a1 - self.C)
            H = min(self.C, self.a2 + a1)

        if L == H:
            return False

        k11 = self.kernel(X1, X1)
        k12 = self.kernel(X1, self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])

        # Compute η (eta), the second derivative of the objective function (Equation 15)
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            # Compute new alpha2 analytically (Equation 16)
            a2_new = self.a2 + self.y2 * (E1 - self.E2) / eta

            # Clip a2_new into [L, H] (Equation 17)
            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        else:
            # Handle non-positive eta case (Equation 19)
            f1 = y1 * (E1 + self.b) - a1 * k11 - s * self.a2 * k12
            f2 = self.y2 * (self.E2 + self.b) - s * a1 * k12 - self.a2 * k22
            L1 = a1 + s * (self.a2 - L)
            H1 = a1 + s * (self.a2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * k11 + \
                   0.5 * L ** 2 * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * k11 + \
                   0.5 * H ** 2 * k22 + s * H * H1 * k12

            if Lobj < Hobj - self.eps:
                a2_new = L
            elif Lobj > Hobj + self.eps:
                a2_new = H
            else:
                a2_new = self.a2

        # Check for sufficient change in alpha2
        if abs(a2_new - self.a2) < self.eps * (a2_new + self.a2 + self.eps):
            return False

        # Compute new alpha1 (Equation 18)
        a1_new = a1 + s * (self.a2 - a2_new)

        # Compute new bias term b (Equation 20, 21)
        new_b = self.compute_b(E1, a1, a1_new, a2_new, k11, k12, k22, y1)
        delta_b = new_b - self.b
        self.b = new_b

        # Update error cache for non-bound alphas
        delta1 = y1 * (a1_new - a1)
        delta2 = self.y2 * (a2_new - self.a2)

        for i in range(self.m):
            if 0 < self.alphas[i] < self.C:
                self.errors[i] += delta1 * self.kernel(X1, self.X[i]) + \
                                  delta2 * self.kernel(self.X2, self.X[i]) - \
                                  delta_b

        self.errors[i1] = 0
        self.errors[i2] = 0

        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        return True

    def compute_b(self, E1, a1, a1_new, a2_new, k11, k12, k22, y1):
        # Compute b1 (Equation 20)
        b1 = E1 + y1 * (a1_new - a1) * k11 + \
             self.y2 * (a2_new - self.a2) * k12 + self.b

        # Compute b2 (Equation 21)
        b2 = self.E2 + y1 * (a1_new - a1) * k12 + \
             self.y2 * (a2_new - self.a2) * k22 + self.b

        if 0 < a1_new < self.C:
            return b1
        elif 0 < a2_new < self.C:
            return b2
        else:
            return (b1 + b2) / 2.0

    def get_error(self, i1):
        if 0 < self.alphas[i1] < self.C:
            return self.errors[i1]
        else:
            return self.output(i1) - self.y[i1]

    def second_heuristic(self, non_bound_indices):
        i1 = -1
        if len(non_bound_indices) > 1:
            max_step = 0
            for j in non_bound_indices:
                E1 = self.errors[j] - self.y[j]
                step = abs(E1 - self.E2)
                if step > max_step:
                    max_step = step
                    i1 = j
        return i1

    def examine_example(self, i2):
        self.y2 = self.y[i2]
        self.a2 = self.alphas[i2]
        self.X2 = self.X[i2]
        self.E2 = self.get_error(i2)

        r2 = self.E2 * self.y2

        # Check KKT conditions
        if not ((r2 < -self.tol and self.a2 < self.C) or
                (r2 > self.tol and self.a2 > 0)):
            return 0

        # Heuristic 2A: choose alpha1 to maximize |E1 - E2|
        non_bound_idx = list(self.get_non_bound_indexes())
        i1 = self.second_heuristic(non_bound_idx)
        if i1 >= 0 and self.take_step(i1, i2):
            return 1

        # Heuristic 2B: loop over all non-bound alphas in random order
        if len(non_bound_idx) > 0:
            rand_i = randrange(len(non_bound_idx))
            for i1 in non_bound_idx[rand_i:] + non_bound_idx[:rand_i]:
                if self.take_step(i1, i2):
                    return 1

        # Heuristic 2C: loop over all training examples in random order
        rand_i = randrange(self.m)
        all_indices = list(range(self.m))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.take_step(i1, i2):
                return 1
        return 0

    def error(self, i2):
        return self.output(i2) - self.y2

    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.alphas > 0, self.alphas < self.C))[0]

    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indexes()
        for i in non_bound_idx:
            num_changed += self.examine_example(i)
        return num_changed

    def main_routine(self):
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.m):
                    num_changed += self.examine_example(i)
            else:
                num_changed += self.first_heuristic()

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
