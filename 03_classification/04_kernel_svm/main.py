# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as p

dataset = p.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Dividir los conjuntos de datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
cs_X = StandardScaler()
X_train = cs_X.fit_transform(X_train)
X_test = cs_X.transform(X_test)


# Ajustar modelo de regresión logística
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# predicción

y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[: ,0].min() -1, stop = X_set[: ,0].max() +1, step = 0.01),
                     np.arange(start = X_set[: ,0].min() -1, stop = X_set[: ,1].max() +1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#752F13', 'green']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    a = (y_set==j).ravel()
    plt.scatter(X_set[a,0],X_set[a,1],
                c = ListedColormap(['#bf420f', 'green'])(i), label = j)
plt.title('SVM(Trainning set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = cs_X.inverse_transform(X_test), y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(cs_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
# Visualising the Training set results


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[: ,0].min() -1, stop = X_set[: ,0].max() +1, step = 0.01),
                     np.arange(start = X_set[: ,0].min() -1, stop = X_set[: ,1].max() +1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#752F13', 'green']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    a = (y_set==j).ravel()
    plt.scatter(X_set[a,0],X_set[a,1],
                c = ListedColormap(['#bf420f', 'green'])(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
