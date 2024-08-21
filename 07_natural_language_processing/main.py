# -*- coding: utf-8 -*-



import pandas as pd

# Es importante que el archivo sea delimitado por tabulador, pa evitar las commas 

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t" , quoting=3) # tratar de ignorar las commillas dobles por que puede generar error


"""
***Limpieza de Textos***
Remplazar los infinitivos, conjugadas, etc por solo los verbos
eliminar numeros, Conectores 
etc
"""

#regx
from re import sub
from nltk import download


download('stopwords') # bajar palabras irrelevantes

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer() # eliminar las raices de las palabras

"""
Corpus: Colección de texto que puede ser usado para cualquier algoritmio/ que ya han sido tratados"""
corpus = []


#range es lo mas optimo en tiempo
values = dataset.values
for i in range(len(values)):
    review = sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words("english")) ] # buscar en sets es mas optimo
    review = " ".join(review)
    corpus.append(review)


# Crear el Bag of Words
# tokenizacion, crear vectores de frecuencia con respecto a las palabras
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



