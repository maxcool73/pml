import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from adalinesgd import AdalineSGD
from commons import plot_decision_regions

s = r'C:\Users\maxim.korolev\PycharmProjects\pml\perceptron\iris.data'
print('URL: ', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0: 100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0: 100, [0, 2]].values
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd )
plt.title('Adaline - стохастический градиентный спуск')
plt.xlabel('длина чашелистика [стандартизированная]')
plt.ylabel('длина лепестка [стандартизированная]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel ('Эпохи')
plt.ylabel ('Усредненные издержки')
plt.tight_layout()
plt.show()
