# Plantilla pre procesado - Datos categoricos

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



### Tratamiento de datos categoricos 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

ct = ColumnTransformer(
    [('encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
x = np.array(ct.fit_transform(x))


y = LabelEncoder().fit_transform(y)
