import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adalinegd import AdalineGD

s = '/home/user/Документы/Projects/Python/pml/perceptron/iris.data'
print('URL: ', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0: 100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0: 100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
           np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Cyммa квадратичных ошибок)')
ax[0].set_title('Adaline - скорость обучения 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
           ada2.cost_, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('log(Cyммa квадратичных ошибок)')
ax[1].set_title('Adaline - скорость обучения 0.0001')

plt.show()
