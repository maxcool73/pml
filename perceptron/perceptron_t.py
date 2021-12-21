import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from commons import plot_decision_regions
from perceptron import Perceptron

# s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
s = '/home/user/Документы/Projects/Python/pml/perceptron/iris.data'
print('URL: ', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
# df.tail()

y = df.iloc[0: 100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0: 100, [0, 2]].values

plt.scatter(X[: 50, 0], X[: 50, 1],
            color='red', marker='o', label='щетинистый')
plt.scatter(X[50: 100, 0], X[50: 100, 1],
            color='blue', marker='x', label='разноцветный')
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1),
         ppn.errors_, marker='o')
plt.xlabel('Эпoxи')
plt.ylabel('Количество обновлений')
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()