import numpy as np
import sys


class NeuralNetMLP(object):
    """Нейронная сеть прямого распространения/ классификатор
    на основе многослойного персептрона.
    Параметры
    n_hidden : int (по умолчанию: 30)
        Количество скрытых элементов.
    l2 : float (по умолчанию: О.)
        Значения лямбда для регуляризации L2.
        Регуляризация отсутствует, если l2=0 (принято по умолчанию).
    epochs : int (по умолчанию: 100)
        Количество проходов по обучающему набору.
    eta : float (по умолчанию: О. 001)
        Скорость обучения.
    shuffle : bool (по умолчанию: True)
        Если True, тогда обучающие данные тасуются
        каждую эпоху, чтобы предотвратить циклы.
    minibatch_size : int (по умолчанию: 1)
        Количество обучающих образцов на мини-пакет.
    seed : int (по умолчанию: None)
        Случайное начальное значение для инициализации весов
        и тасования.
    Атрибуты
    eval : dict
        Словарь, в котором собираются показатели издержек,
        правильности при обучении и правильности при испытании
        для каждой эпохи во время обучения.
    """

    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """ Кодирует метки в представление с унитарным кодом
        Параметры
        у : массив, форма = [n_examples]
            Целевые значения.
        Возвращает
        onehot : массив, форма (n_examples, n_labels)
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _siqmoid(self, z):
        """Вычисляет логистическую (сигмоидальную) функцию"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Вычисляет шаг прямого распространения"""
        # шаг 1 : общий вход скрытого слоя
        # скалярное произведение {n_examples, n_features}
        # и {n_features, n_hidden}
        # -> {n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h
        # шаг 2: активация скрытого слоя
        a_h = self._siqmoid(z_h)
        # шаг 3: общий вход выходного слоя
        # скалярное произведение {n_examples, n_hidden}
        # и [n_hidden, п classlabels}
        # -> [n_examples, n_classlabels]
        z_out = np.dot(a_h, self.w_out) + self.b_out
        # шаг 4: активация выходного слоя
        a_out = self._siqmoid(z_out)
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """ Вычисляет функцию издержек.
        Параметры
        у_епс : массив, форма = (n_examples, n_labels)
            Метки классов в унитарном коде.
        output: массив, форма = [n_examples, n_output_units]
            Активация выходного слоя(прямое распространение)
        Возвращает
        cost : float
            Регуляризированные издержки
        """
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """Прогнозирует метки классов.
        Параметры
        Х : массив, форма = [n_examples, n_features]
            Входной слой с первоначальными признаками.
        Возвращает
        y__pred : массив, форма = [n_examples]
            Спрогнозированные метки классов."""
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Выясняет веса из данных.
        Параметры
        X_train : массив, форма = [n_examples, n_features]
            Входной слой с первоначальными признаками.
        y_train : массив, форма = [n_examples]
            Целевые метки классов.
        X_valid: array, shape = [n_examples, n_features]
            Признаки образцов для проверки во время обучения
        y_valid : массив, форма= [n_examples]
            Метки образцов для проверки во время обучения
        Возвращает
        self
        """
        n_output = np.unique(y_train).shape[0]  # количество меток классов
        n_features = X_train.shape[1]
        #######################
        # Инициализация весов #
        #######################
        # веса для входного слоя -> скрытого слоя
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        # веса для скрытого слоя -> выходного слоя
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))
        epoch_strlen = len(str(self.epochs))  # для форма та
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self._onehot(y_train, n_output)
        # итерация по эпохам обучения
        for i in range(self.epochs):
            # итерация по мини-пакетам
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                # прямое распространение
                z_h, a_h, z_out, a_out = self._forward(X_train.iloc[batch_idx])
                ############################
                # Обратное распространение #
                ############################
                # {n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]
                # {n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)
                # скалярное произведение {n_examples, п classlabels]
                # и [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h)
                # скалярное произведение [n_features, n_examples]
                # и [n_examples, n_hidden]
                # -> {n_features, n_hidden]
                grad_w_h = np.dot((X_train.iloc[batch_idx]).to_numpy(copy=True).T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)
                # скалярное произведение [n_hidden, n_examples]
                # и {n_examples, n_classlabels]
                # -> {n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)
                # Регуляризация и обновления весов
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h  # смещение не регуляризируется
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out  # смещение не регуляризируется
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
            ##########
            # Оценка #
            ##########
            # Оценка после каждой эпохи во время обучения
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid.shape[0])
            sys.stderr.write('\r%0*d/%d | Издержки: %.2f | Правильность при обучении/при проверке: %.2f%%/%.2f%%' % (
                epoch_strlen, i + 1, self.epochs, cost, train_acc * 100, valid_acc * 100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
        return self
