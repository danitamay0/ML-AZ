# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, log

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# hacemos una estrategia para saber cuales vamos a mostrar, basado en las 10 rondas,
# dependiendo de cada ronda hacemos el refuerzo y se decide cual mostrar

""" 
Vamos a comprar si seleccionamos de manera aleatoria Vs con el refuerzo 

randmom sekection 1227 han hecho click de manera leatoria y cambia cada vez que se ejcuta
"""

"""
N_i(n)
R_i(n)
"""
N = len(dataset.values)
d = len(dataset.values[0])
number_of_selections = [0] * d
sums_of_rewars = [0] * d
adds_selected = []
total_reward = 0
"""
paso 2 

"""

for n in  range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        
        if number_of_selections[i] > 0:
        
            """
             r_i(n) = R_i(n) / N_i(n)
            """
            average_reward  = sums_of_rewars[i] / number_of_selections[i]
            
        
            """ delta_i(n) = """
            delta_i = sqrt( 3/2*log(n + 1 )/number_of_selections[i] )
            
            upper_bound= average_reward + delta_i
            
        else:
            upper_bound = 1e400 # se inicia con un limite superior muy alto 
            
        
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    adds_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewars[ad] = sums_of_rewars[ad] + reward
    total_reward = total_reward+ reward
            

# histograma de resultados

plt.hist(adds_selected)
plt.title("Histograma de anuncio")
plt.xlabel("ID del anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()