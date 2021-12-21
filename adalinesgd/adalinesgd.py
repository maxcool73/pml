import numpy as np


class AdalineSGD:
    """Классификатор на основе персептрона.
    Параметры
    ---------
    eta: float              Скорость обучения (между О.О и 1.0)
    n_iter: int             Проходы по обучающему набору данных.
    random_state: int       Начальное значение генератора случайных чисел для инициализации случайными весами.

    Атрибуты
    --------
    w_:                     одномерный массив   Веса после подгонки.
    cost:                   список Значение функции издержек на основе суммы квадратов в каждой эпохе ."""

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """Подгоняет к обучающим данным .
            Параметры
            Х: {подобен массиву}, форма = [n_examples , n_features]
            Обучающие векторы, где n_examples - количество образцов
            и n_features - количество признаков.

            у: подобен массиву, форма = [n_examples]
            Целевые значения.
            Возвращает
            self: object"""
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Подгоняет к обучающим данным без повторной
        подгонки весов"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
            else:
                self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Тасует обучающие данные"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Инициализирует веса небольшими случайными числами"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ Применяет правило обучения Adaline для обновления весов"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def activation(self, X):
        """Вычисляет линейную активацию"""
        return X

    def net_input(self, X):
        """Вычисляет общий вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Возвращает метку класса после единичного шага"""
        return np.where(self.net_input(X) >= 0, 1, -1)
