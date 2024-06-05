# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Seccion 2 Pre Procesado de Datos
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

# no es necesario escalar variables, por que el algoritmo necesita establecer relaciones no lineales, por lo tanto no es necesario


### Ajustar la regresión lineal con el dataset (ejemplo de que pasaria si uso una regresion lineal en un modelo polinomico)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


### Ajustar la regresión polinómica con el dataset

from sklearn.preprocessing import PolynomialFeatures
# creamos la matriz polinomiales, pasamos los grados de las caracteristicas de X y sus cuadrados
poly_reg = PolynomialFeatures(degree=4) # regresion polinomica de grado 2
# de una vez se agrega la columna de constante y las variables
X_poly = poly_reg.fit_transform(X) # aplicamos el fit y el transform de 

# previamente la matriz de caracteristicas debe ser modelada, de modo que se incluyan las variables indepentiendes con sus 
# transformaciones polinomicas

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


### Visualización de los resultados del model lineal


### Predicción de nuestros modelos

lin_reg.predict([[6.5]])

### Dato correcto
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

# recta de regression Linear
# pintar nubes de puntos
pl.scatter(X, y, color = "red")
pl.plot(X, lin_reg.predict(X), color="blue")
pl.title("Modelo de regresión lineal")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()
### Visualización de los resultados del model polinómico

pl.scatter(X, y, color = "red")
#NOTA: PARA USAR LA PREDICCION SE USA X_poly por que tiene la misma cantidad de columnas que fue entrenado el modelo
pl.plot(X, lin_reg_2.predict(X_poly), color="blue")
pl.title("Modelo de regresión polinomica")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()



### Suavizar la visualización del polinomio extendiendo los valores

X_grid = np.arange(min(X), max(X), 0.1 ) # creamos un rango y rellenamos
X_grid = X_grid.reshape(len(X_grid), 1) # lo convertimos en una matriz
# Visualización de los resultados del model polinómico

pl.scatter(X, y, color = "red")
#NOTA: PARA USAR LA PREDICCION SE USA X_poly por que tiene la misma cantidad de columnas que fue entrenado el modelo
pl.plot(X_grid, lin_reg_2.predict( poly_reg.fit_transform(X_grid) ), color="blue")
pl.title("Modelo de regresión polinomica")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()


