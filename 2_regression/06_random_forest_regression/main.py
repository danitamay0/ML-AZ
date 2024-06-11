# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



from sklearn.ensemble import RandomForestRegressor
regresor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regresor = regresor.fit(X, y)

y_pred = regresor.predict([[6.5]])


X_grid = np.arange(min(X), max(X), 0.1 ) # creamos un rango y rellenamos
X_grid = X_grid.reshape(len(X_grid), 1) # lo convertimos en una matriz

print(y_pred)

plt.scatter(X, y, color="red")
plt.plot(X_grid, regresor.predict(X_grid), color="blue")
plt.scatter([[6.5]], regresor.predict([[6.5]]),  color = "green")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()

