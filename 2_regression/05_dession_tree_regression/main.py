# -*- coding: utf-8 -*-

"""
Regresión con árboles de decisión
"""

import numpy as np
import matplotlib.pyplot as pl
import pandas as p


#import dataset

dataset = p.read_csv("Position_Salaries.csv")


### localizar elementos por posicion 

# definir matriz de variables independientes SIEMPRE DEBE SER UNA MATRIZ
X = dataset.iloc[:,1:2].values # llamar todas las filas, llamar todas las columnas excepto la ultima

# definir vector de variable dependiente
y = dataset.iloc[:, -1].values




### Dividir el dataset en conjunto de entrenamiento y conjunto de testing
"""from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
"""


### Ajustar la regresión 

from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X, y)



### Predicción de nuestros modelos

y_pred = regression.predict([[12]])
### Suavizar la visualización del polinomio extendiendo los valores




 
### Modelo suavizada suavizar y sin invertir la transformación
X_grid = np.arange(min(X), max(X), 0.1 ) # creamos un rango y rellenamos
X_grid = X_grid.reshape(len(X_grid), 1) # lo convertimos en una matriz


pl.scatter(X, y, color = "red")
pl.plot(X_grid, regression.predict( X_grid ), color="blue")
pl.scatter([[14]], regression.predict([[14]]),  color = "green")

pl.title("Modelo de regresión SVR")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()

