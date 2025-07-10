import numpy as np

class KernelSVM_SMO:
    def __init__(self, C=1.0, gamma=None, tol=1e-3, max_passes=5, max_iter=1000):
        # C: parametro di regolarizzazione che bilancia il margine e gli errori di classificazione
        # gamma: parametro del kernel RBF, che controlla la "larghezza" della gaussiana
        # tol: tolleranza per controllare la violazione delle condizioni KKT
        # max_passes: numero massimo di iterazioni senza modifiche agli alpha per terminare l'algoritmo
        # max_iter: numero massimo di iterazioni totali per la convergenza
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.alpha = None  # moltiplicatori di Lagrange, vettore di pesi per i dati di supporto
        self.b = 0.0       # bias/intercetta della funzione decisionale
        self.X = None      # dati di addestramento (feature)
        self.y = None      # etichette di addestramento (+1 o -1)
        self._kernel_cache = {}  # cache per memorizzare valori del kernel già calcolati

    def kernel(self, i, j):
        # Kernel RBF (Radial Basis Function) tra i-esimo e j-esimo campione
        # K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
        # Cache per evitare ricalcoli costosi
        if (i, j) in self._kernel_cache:
            return self._kernel_cache[(i, j)]
        if (j, i) in self._kernel_cache:
            return self._kernel_cache[(j, i)]
        k = np.exp(-self.gamma * np.sum((self.X[i] - self.X[j]) ** 2))
        self._kernel_cache[(i, j)] = k
        return k

    def decision_function(self, i):
        # Calcola la funzione decisionale f(x_i) = sum_j alpha_j y_j K(x_j, x_i) + b
        # solo per j con alpha_j > 0 (support vectors)
        if self.alpha is None:
            return 0
        result = 0
        idx = np.where(self.alpha > 0)[0]
        for j in idx:
            result += self.alpha[j] * self.y[j] * self.kernel(j, i)
        return result + self.b

    def compute_error(self, i):
        # Errore di classificazione per il punto i: E_i = f(x_i) - y_i
        # Serve per valutare se alpha_i viola le condizioni KKT
        return self.decision_function(i) - self.y[i]

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        # Se gamma non è specificato, lo si calcola come inverso del numero di features
        if self.gamma is None:
            self.gamma = 1.0 / n_features

        self.alpha = np.zeros(n_samples)  # inizializza alpha a zero (nessun support vector)
        self.b = 0.0

        passes = 0  # contatore di iterazioni senza cambiamenti agli alpha
        it = 0      # contatore iterazioni totali

        # Ciclo principale di ottimizzazione SMO
        while passes < self.max_passes and it < self.max_iter:
            num_changed_alphas = 0

            # Identifica i "violators", ovvero punti che violano le condizioni KKT:
            # Condizioni KKT per alpha_i:
            # 1) Se alpha_i è tra 0 e C, allora E_i * y_i = 0 (vicino a zero)
            # 2) Se alpha_i = 0, allora E_i * y_i >= 0
            # 3) Se alpha_i = C, allora E_i * y_i <= 0
            violators = []
            for i in range(n_samples):
                E_i = self.compute_error(i)
                r_i = E_i * self.y[i]
                # Se viola le condizioni KKT entro la tolleranza tol, viene considerato un violatore
                # Il confronto con la tolleranza mi garantisce di essere sicuramente nel caso 2 o 3 e non nell'intorno
                # di 0. La prima condizione viene indirettamente controllata
                if ((r_i < -self.tol and self.alpha[i] < self.C) or
                    (r_i > self.tol and self.alpha[i] > 0)):
                    violators.append(i)

            # Se non ci sono violatori, incrementa passes (iterazioni senza cambiamenti)
            if len(violators) == 0:
                passes += 1
                it += 1
                continue

            # Scegli la coppia (i, j) di moltiplicatori più "violanti" (Most Violating Pair)
            # la coppia con la differenza di errori |E_i - E_j| massima per massimizzare il passo di aggiornamento
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

            # Se non trova una coppia valida, incrementa passes e continua
            if i_pair == -1 or j_pair == -1:
                passes += 1
                it += 1
                continue

            # Ottimizza la coppia di moltiplicatori (i_pair, j_pair)
            if self.take_step(i_pair, j_pair):
                num_changed_alphas += 1
            else:
                passes += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0  # resetta passes se ci sono stati cambiamenti

            it += 1  # incremento iterazioni totali

        # Dopo il training, memorizza i support vectors (alpha > 0)
        self.support_ = np.where(self.alpha > 1e-8)[0]
        self.support_vectors_ = self.X[self.support_]
        self.support_alpha = self.alpha[self.support_]
        self.support_y = self.y[self.support_]

        return self

    def take_step(self, i, j):
        # Procedura di aggiornamento degli alpha_i e alpha_j basata su SMO
        if i == j:
            return False

        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]
        y_i = self.y[i]
        y_j = self.y[j]
        E_i = self.compute_error(i)
        E_j = self.compute_error(j)
        s = y_i * y_j  # prodotto etichette

        # Calcola i limiti L e H per alpha_j (vincoli box)
        # Se y_i != y_j:
        # L = max(0, alpha_j - alpha_i)
        # H = min(C, C + alpha_j - alpha_i)
        # Altrimenti:
        # L = max(0, alpha_j + alpha_i - C)
        # H = min(C, alpha_j + alpha_i)
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)

        if L == H:
            return False

        # Calcola i valori del kernel necessari
        k_ii = self.kernel(i, i)
        k_jj = self.kernel(j, j)
        k_ij = self.kernel(i, j)

        # Calcola eta = K_ii + K_jj - 2 K_ij, coefficiente della quadraticità
        # Eta > 0 per la convexità della funzione da ottimizzare
        eta = k_ii + k_jj - 2 * k_ij
        if eta <= 0:
            return False

        # Calcola il nuovo alpha_j usando la formula derivata dal vincolo ottimale:
        # alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
        alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
        # Clippa alpha_j_new nel range [L, H]
        alpha_j_new = np.clip(alpha_j_new, L, H)

        # Controlla se il cambiamento è significativo, altrimenti ritorna False
        if abs(alpha_j_new - alpha_j) < 1e-5 * (alpha_j_new + alpha_j + 1e-5):
            return False

        # Calcola il nuovo alpha_i usando il vincolo di uguaglianza:
        # alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)
        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)

        # Calcola i nuovi valori di soglia b, b1 e b2,
        # basati sugli errori e variazioni degli alpha
        b1 = self.b - E_i - y_i * (alpha_i_new - alpha_i) * k_ii - y_j * (alpha_j_new - alpha_j) * k_ij
        b2 = self.b - E_j - y_i * (alpha_i_new - alpha_i) * k_ij - y_j * (alpha_j_new - alpha_j) * k_jj

        # Scegli il nuovo b in base a quali alpha sono tra 0 e C,
        # altrimenti media tra b1 e b2
        if 0 < alpha_i_new < self.C:
            b_new = b1
        elif 0 < alpha_j_new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        # Aggiorna i parametri del modello
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        self.b = b_new

        return True

    def predict(self, X):
        # Predice le etichette per i nuovi dati X
        y_pred = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            s = 0
            # Calcola f(x) = sum_i alpha_i y_i K(x_i, x) + b usando solo support vectors
            for alpha_i, y_i, x_i in zip(self.support_alpha, self.support_y, self.support_vectors_):
                s += alpha_i * y_i * np.exp(-self.gamma * np.sum((x - x_i) ** 2))
            y_pred[idx] = s + self.b
        # Restituisce la classe +1 o -1 in base al segno di f(x)
        return np.sign(y_pred)