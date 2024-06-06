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

### Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


### Ajustar la regresión 

from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X, y)



### Predicción de nuestros modelos

y_pred = regression.predict(sc_x.transform([[6.5]])).reshape(-1,1)
y_pred = sc_y.inverse_transform(y_pred)
#sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))
### Suavizar la visualización del polinomio extendiendo los valores



#X_grid = np.arange(min(X), max(X), 0.1 ) # creamos un rango y rellenamos
#X_grid = X_grid.reshape(len(X_grid), 1) # lo convertimos en una matriz
# Visualización de los resultados del model

### Modelo sin suavizar y sin invertir la transformación
pl.scatter(X, y, color = "red")
#NOTA: PARA USAR LA PREDICCION SE USA X_poly por que tiene la misma cantidad de columnas que fue entrenado el modelo
pl.plot(X, regression.predict( X ), color="blue")
pl.title("Modelo de regresión SVR")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()

 
### Modelo suavizada suavizar y sin invertir la transformación
X_grid = np.arange(min(X), max(X), 0.1 ) # creamos un rango y rellenamos
X_grid = X_grid.reshape(len(X_grid), 1) # lo convertimos en una matriz


pl.scatter(X, y, color = "red")
pl.plot(X_grid, regression.predict( X_grid ), color="blue")
pl.title("Modelo de regresión SVR")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()


### Modelo suavizada suavizar y sin invertir la transformación

X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

pl.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
pl.scatter([[6.5]],[y_pred], color = "green")
#NOTA: PARA USAR LA PREDICCION SE USA X_poly por que tiene la misma cantidad de columnas que fue entrenado el modelo
pl.plot( X_grid,  sc_y.inverse_transform(regression.predict( sc_x.transform(X_grid)).reshape(-1,1)), color="blue")
pl.title("Modelo de regresión SVR")
pl.xlabel("Posicion del empleado")
pl.ylabel("Sueldo en $")
pl.show()