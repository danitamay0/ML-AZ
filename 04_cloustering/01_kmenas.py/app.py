
# K-Means


import numpy as np
import matplotlib.pyplot as plt
import pandas as p

# Cargamos datos

dataset = p.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values


# Metodo del codo para averiguar

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i , init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # suma de cuadrados
    
print(wcss)   

plt.plot(range(1,11),wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clousters")
plt.ylabel("WCSS(K)")
plt.show()


# Aplicar el metodo de kmeans para segmentar el dataset

k = 5

kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visializacion de los clousters
print(kmeans.cluster_centers_)

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c="red", label="Cluster 1" )
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1],  s=100, c="blue", label="Cluster 2" )
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1],  s=100, c="green", label="Cluster 3" )
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1],  s=100, c="cyan", label="Cluster 4" )
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1],  s=100, c="magenta", label="Cluster 5" )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1] ,  s = 300 ,c="yellow", label="Baricentros" )
plt.title("Closuter de clientes")
plt.xlabel("ingresos anuales (miles de $)")
plt.ylabel("Puntuación de gastos (1-100)")
plt.legend()
plt.show()
