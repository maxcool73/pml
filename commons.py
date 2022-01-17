import math
from math import comb
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Настроить генератор маркеров и карту цветов
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Вывести образцы по классам
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


def rbf_kernel_pca(X, gamma, n_components):
    """Реализация алгоритма РСА с ядром RBF.
    Параметры
    ---------
    Х: {NumPy ndarray}, форма[n_examples, n_features]
    gamma: float
        Параметр настройки ядра RBF
    n_components: int
        Количество главных компонентов, подлежащих возвращению
    Возвращает
    Х_рс: {NumPy ndarray}, форма= {n_examples, k_features}
        Спроецированный набор данных
    """
    # Вычислить попарные квадратичные евклидовы расстояния
    # в МхN-мерном наборе данных.
    sq_dists = pdist(X, 'sqeuclidean')
    # Преобразовать попарные расстояния в квадратную матрицу.
    mat_sq_dists = squareform(sq_dists)
    # Вычислить симметричную матрицу ядра.
    K = exp(-gamma * mat_sq_dists)
    # Центрировать матрицу ядра.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # Получить собственные пары из центрированной матрицы ядра;
    # scipy. linalg. eigh возвращает их в порядке по возрастанию.
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    # Собрать верхние k собственных векторов (спроецированных образцов).
    X_pc = np.column_stack([eigvecs[:, i] for i in range(n_components)])
    return X_pc


def rbf_kernel_pca2(X, gamma, n_components):
    """Реализация алгоритма РСА с ядром RBF.
    Параметры
    ---------
    Х: {NumPy ndarray}, форма[n_examples, n_features]
    gamma: float
        Параметр настройки ядра RBF
    n_components: int
        Количество главных компонентов, подлежащих возвращению
    Возвращает
    ----------
    alphas: {NumPy ndarray}, форма
        Спроецированный набор данных
    lambdas: список
        Собственные значения
    """
    # Вычислить попарные квадратичные евклидовы расстояния
    # в МхN-мерном наборе данных.
    sq_dists = pdist(X, 'sqeuclidean')
    # Преобразовать попарные расстояния в квадратную матрицу.
    mat_sq_dists = squareform(sq_dists)
    # Вычислить симметричную матрицу ядра.
    K = exp(-gamma * mat_sq_dists)
    # Центрировать матрицу ядра.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # Получить собственные пары из центрированной матрицы ядра;
    # scipy. linalg. eigh возвращает их в порядке по возрастанию.
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    # Собрать верхние k собственных векторов
    # (спроецированных образцов).
    alphas = np.column_stack([eigvecs[:, i] for i in range(n_components)])
    # Собрать соответствующие собственные значения.
    lambdas = [eigvals[i] for i in range(n_components)]
    return alphas, lambdas


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) *
             error ** k *
             (1 - error) ** (n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

def preprocessor (text):
    text = re.sub( '<[^>]*>', '', text)
    emoticons = re.findall ('(?::|;|=) (?:-)?(?:\)|\(|D|Р)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text