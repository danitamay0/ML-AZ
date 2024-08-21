

# Sistemas de recomendaciónes para entender el compertamiento

import numpy as np
import matplotlib as np
import pandas as pd

## este dataset no tiene cabecera
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

"""
Este dataset es un excel de ventas donde cada columna es el producto de la tienda
no es el formato que necesita el algoritmo de apriori por que no es representativo, no se 
necesitan los nan por cada compra, simplente los productos que compró.
Para ello necesitamos hacer una transformación de los datos 
"""
print(dataset)

transactions = []

for index, row in dataset.iterrows():
    
    data = []
    for  col in row:
        if str(col) != 'nan':
            data.append(col)          
            
    transactions.append(data)
    
    
from apyori import apriori

'''
 min_support: el total de transacciones que tienen un determinado item dividido en el numero total de items, ej un item que se compra 3 veces a la semana
min_confidence: en que porcentaje de las compras aparece, lo ideal es empezar con un rango alto ej 80% e ir bajando, no se puede ser tan estricto y si es muy bajo no  hay coincidencias y no se dan valores logicos
min_linft: ranking de las mejores reglas, es decir las mas relevantes o mas fuertes

 '''
print(len(transactions))

# mininmo 3 veces al dia en 7 dias de la semana

min_support = 3 * 7 / (len(transactions) - 1) # = 0.0028

rules = apriori(transactions, min_support = min_support , min_confidence=0.4 , min_lift=3 , min_lenght=2)



results = list(rules)




rule = list()
support = list()
confidence = list()
lift = list()
 
for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    
    
    rule.append( [ f"{x} - " for x in items ] )
    
    #second index of the inner list
    support.append(str(item[1]))
 
    #third index of the list located at 0th
    #of the third index of the inner list
 
    confidence.append(item[2][0][2])
    lift.append(item[2][0][3])
 
output_ds  = pd.DataFrame({'rule': rule,
                           'support': support,
                           'confidence': confidence,
                           'lift': lift
                          }).sort_values(by = 'lift', ascending = False)