import numpy as np


class LogisticRegressionGD:
    """Классификатор на основе логистической реrрессии,
        использующий градиентный спуск.
    Параметры
    eta : float
        Скорость обучения (между О. О и 1. 0).
    п_iter : int
        Проходы по обучающему набору данных.
    random_state : int
        Начальное значение генератора случайных чисел
        для инициализации случайными весами.
    Атрибуты
    w_ : одномерный массив
        Веса после подгонки.
    cost_ : list
        Значение логистической функции издержек в каждой эпохе.
    """

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
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # обратите внимание, что теперь мы вычисляем
            # логистические 'издержки', а не издержки в виде
            # суммы квадратичных ошибок
            cost = (-y.dot(np.log(output)) -
                    ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Вычисляет общий вход"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Вычисляет логистическую сигмоидальную активацию"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Возвращает метку класса после единичного шага"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # эквивалентно:
        # return np.where(self.activation(self.net_input(X))
        #                 >= 0.5, 1, 0)
