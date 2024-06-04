
# Plantilla pre procesado - Datos faltantes

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



### Tratamientos de NAs


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# axis es en que va a tomar el valor 0 = columna 1 = filas

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean',  )

# ajustar los datos que se van a trabajar, las columnas que se van a seleccionar
imp_mean.fit(x[:,1:3])

# transformar las columnas
x[:, 1:3]  = imp_mean.transform(x[:, 1:3])
