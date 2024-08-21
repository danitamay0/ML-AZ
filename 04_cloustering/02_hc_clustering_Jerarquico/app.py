# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values


# Utilizar el dendograma para encontrar el numero obtimo de clusters
from scipy.cluster import hierarchy as sch

## algoritmo aglonerativo
link = sch.linkage(X, method="ward") # intentamos minimizar la varianza entre los puntos del cluster
dendrogram = sch.dendrogram(link)

plt.title("Dendograma")
plt.xlabel("Clientes")

plt.ylabel("Distancia Euclididea")
plt.show()

k = 5

## crear cluster jerarquico aglomerativo

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)


plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c="red", label="Cautos" )
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1],  s=100, c="blue", label="Standard" )
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1],  s=100, c="green", label="Objetivo" )
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1],  s=100, c="cyan", label="Descuidado" )
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1],  s=100, c="magenta", label="Conservadores" )

plt.title("Closuter de clientes")
plt.xlabel("ingresos anuales (miles de $)")
plt.ylabel("Puntuación de gastos (1-100)")
plt.legend()
plt.show()