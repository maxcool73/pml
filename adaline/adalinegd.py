import numpy as np


class AdalineGD:
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

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self, X):
        """Вычисляет линейную активацию"""
        return X

    def net_input(self, X):
        """Вычисляет общий вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Возвращает метку класса после единичного шага"""
        return np.where(self.net_input(X) >= 0, 1, -1)
