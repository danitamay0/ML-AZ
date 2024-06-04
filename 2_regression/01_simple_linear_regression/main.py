# -*- coding: utf-8 -*-


### Modelo lineal sirve para predecir datos lineales


"""
Seccion 2 Pre Procesado de Datos
"""

import numpy as np
import matplotlib.pyplot as pl
import pandas as p


#import dataset

dataset = p.read_csv("Salary_Data.csv")


### localizar elementos por posicion 

# definir matriz de variables independientes
x = dataset.iloc[:,:-1].values # llamar todas las filas, llamar todas las columnas excepto la ultima

# definir vector de variable dependiente
y = dataset.iloc[:,1].values




### Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)

### Crear modelo de regresion lineal simple  con el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)

### Predecir el conjunto de test

y_pred = regression.predict(x_test)

specific_prediction = regression.predict([[6.7]])


### Visualizar los resultados de entrenamiento
# Nube de puntos
pl.scatter(x_train, y_train, color="red")

# Recta de regresesión
pl.plot(x_train, regression.predict(x_train), color="blue")

pl.title("Sueldo vs Años de experiencia [Conjunto de entrenamiento]" )
pl.xlabel("Años de experiencia")
pl.xlabel("Sueldo en $")
pl.show()


### Visualizar los resultados de testing
# Nube de puntos
pl.scatter(x_test, y_test, color="red")

# Recta de regresesión se puede dejar la recta de train
""" Se toma el conjunto faltante para validar que la recta no sufra de overfitting"""
pl.plot(x_train,  regression.predict(x_train), color="blue")

pl.title("Sueldo vs Años de experiencia [Conjunto de testing]" )
pl.xlabel("Años de experiencia")
pl.xlabel("Sueldo en $")
pl.show()


### Visualizar los resultados de predictivos
# Nube de puntos
pl.scatter(x_test, y_pred, color="red")

# Recta de regresesión se puede dejar la recta de train
pl.plot(x_train,  regression.predict(x_train), color="blue")

pl.title("Sueldo vs Años de experiencia [Conjunto de testing]" )
pl.xlabel("Años de experiencia")
pl.xlabel("Sueldo en $")
pl.show()

""" 
LinearRegression fits a linear model with coefficients w = (w1, …, wp) 
to minimize the residual sum of squares between the observed targets in the dataset, 
and the targets predicted by the linear approximation.
"""



