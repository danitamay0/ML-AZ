# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""
Seccion 5 Regresion lineal multiple
"""

import numpy as np
import matplotlib.pyplot as pl
import pandas as p


#import dataset

dataset = p.read_csv("50_Startups.csv")


### localizar elementos por posicion 

# definir matriz de variables independientes
x = dataset.iloc[:,:-1].values # llamar todas las filas, llamar todas las columnas excepto la ultima

# definir vector de variable dependiente
y = dataset.iloc[:,4].values




### Tratamiento de datos categoricos 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

ct = ColumnTransformer(
    [('encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
x = np.array(ct.fit_transform(x))


### Evitar la trampa de las variables ficticias, multi-colinealidad 

x = x[:,1:]

### Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


### Ajustar el modelo de regresiojn lineal multiple con el modelo de entrenamiento

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)


y_pred = regression.predict(x_test)


### Construir el modelo optimo de RLM utilizando la eliminación hacia atrás

import statsmodels.api as sm

# añadimos el valor de 1 para hacer mension al termino dependiente o ordenada en el origen
# 50 es el numero de filas del dataset original
x = np.append(np.ones((50,1)).astype(int) , x , axis=1)


# PASO 1
SSL = 0.05


from functions import backwardElimination

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, y,  SSL)

x_train_opt, x_test_opt, y_train_opt, y_test_opt = train_test_split(x_Modeled, y, test_size=0.2, random_state=0)

regression = LinearRegression()
regression.fit(x_train_opt, y_train_opt)

y_pred_opt = regression.predict(x_test_opt)



