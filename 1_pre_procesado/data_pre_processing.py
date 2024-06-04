# -*- coding: utf-8 -*-
"""
Seccion 2 Pre Procesado de Datos
"""

import numpy as np
import matplotlib.pyplot as pl
import pandas as p


#import dataset

dataset = p.read_csv("Data.csv")


### localizar elementos por posicion 

# definir matriz de variables independientes
x = dataset.iloc[:,:-1].values # llamar todas las filas, llamar todas las columnas excepto la ultima

# definir vector de variable dependiente
y = dataset.iloc[:,3].values




### Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


 
"""

### Escalado de variables
from sklearn.preprocessing import StandardScaler

#Fit to data, then transform it.
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

# only transform, el ajuste de la data queda en la instancia y el escalado va a ser en base a x_traain
x_test = sc_x.transform(x_test)

"""
