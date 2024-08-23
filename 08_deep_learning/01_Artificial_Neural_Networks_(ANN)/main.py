# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


# Codificar datos categoricos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

label_X_1 = LabelEncoder()
X[:,1] = label_X_1.fit_transform(X[:,1])

label_X_2 = LabelEncoder()
X[:,2] = label_X_2.fit_transform(X[:,2])


# Variables dummy
ct = ColumnTransformer(
    [('encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X))

#Overfiting

X = X[:,1:]



# Dividir dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)



# Pre procesado de datos

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




# BUild la RNA

# Keras y librerias

import keras
from keras import Sequential
from keras.layers import Dense

# Inicial la RNA

''' 
Existen 2 maneras de inicializar la RNA
*1 : Definir la secuencia de capas 
2 : Definir el grafo de como se relacionan
'''

classifier = Sequential()


'''
Paso 1 Inicializar los pesos aleatoriamente con valores cercanos a 0

Unit = N nodos que hay en la siguiente capa
Regla de oro (No obligatoria):En la capa oculta ponemos la media de la Capa de entrada y la capa de salida
ejemplo: 11 variables indepedientes y variable dependiente = 6


kernel_initializer = como inicializamos los pesos   con valores cercanos a 0

activation = la funcion de activacion  relu = rectificadora


input_dim = numero de nodos en la capa actual
'''

# Añadir primera capa de entrada y Añadir primera capa oculta
classifier.add(
    Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim = 11 ) # Dense crea la zona de sinapsis de conexiones entre capas, 
    )

# Añadir primera capa oculta

'''
Ahora la capa sabe que tiene una dimension de 6 por que viene de la capa anterior, por ende
la nueva capa oculta ya sabe lo que espera de la capa anterior, por ende input_dim=6 pero ya no es necesario
'''
classifier.add(
    Dense(units=6, kernel_initializer="uniform", activation="relu" ) # Dense crea la zona de sinapsis de conexiones entre capas, 
    )



# Añadir la capa final de salida
'''
sigmoid, sirve bien para resultados binarios 1/0

si queremos añadir otra categoria en la misma variable ejmpplo Exited= 1/2/3 tenemos que cambiar unit=3 y 
la funcion sigmoide no seria la mas adecuada, podria ser un escalon para saber donde activarse cada una,
Si usamos la funcion sigmoide posiblemente deberíamos utilizar una funcion softman, que todas las posibilidades sumaran 1


'''
classifier.add(
    Dense(units=1, kernel_initializer="uniform", activation="sigmoid" ) 
    )


'''
Paso 2 Introducir la primera observacion del dataset en la capa de entrada. Cada caracteristica es un node de entrada
'''

# Compilar la red neuronal artifical
'''
Optimizer: como construir las ponderaciones de los pesos - grandiente decendiente, etc... tiene por default
adam recomendado

lcss: Seleccionar la  funcion de costes

metrics: medir la precision del algoritmo
'''

classifier.compile( optimizer = "adam", loss="binary_crossentropy" , metrics=["accuracy" ])

'''
Paso 3 Propagacion hacia adelante de izquierda a derecha, la neurona se activa de modo
que cada activacion de cada una se limita por los pesos. propaga las activaciones hasta obetener la prediccion y^

En este caso se usará la función de activación rectificador
y para la salida la funcion sigmoidea para obtener la probabilidad 0-1 


Paso 4 Compraramos la predicci[on con el resultado real. se mide entonces el error generado
                     

Paso 5 Propagación haciua atrás derecha a izquierda. propagando el error y actualizando los pesos

Paso 6 repetir pasos 1 a 5 y se actualizan despues de cada operacion (reinforcement lerning) o se actualiza los pesos despues
de un conjunto de observaciones (batch learning)

batch= numero de batch para ejecutar batch learning por defecto es None, ejemplo 10 elementos y luego procesar los pesos

Paso 7 Cuando todo el conjuinto de entrenamiento ha pasado por la RNA se completa un epoch. hacer mas epochs
epoch= pasadas del conjunto de datos (tener cuidado, muchas pasadas puede generar overfiting)
'''

classifier.fit(X_train, y_train, batch_size=10, epochs=100 )



y_pred = classifier.predict(X_test)

# tenemos resultados probabilisticios necesitamos generar un umbral de cuanto
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac *  100)




