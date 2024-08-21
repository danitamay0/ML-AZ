# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from random import  betavariate
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# hacemos una estrategia para saber cuales vamos a mostrar, basado en las 10 rondas,
# dependiendo de cada ronda hacemos el refuerzo y se decide cual mostrar

""" 

randmom sekection 1227 , UCB 2178, thomson 2586
"""

"""
N_i^1(n) el numero de veces que el anuncio i recibe una recompensa 1 hasta la ronda n
N_i^0(n) el numero de veces que el anuncio i recibe una recompensa 0 hasta la ronda n
"""
N = len(dataset.values)
d = len(dataset.values[0])

number_of_reward_1 = [0] * d
number_of_reward_0 = [0] * d

adds_selected = []
total_reward = 0    
"""
paso 2 

"""

for n in  range(0, N):
    max_random = 0 # mayor distribución de exito en base a una elaboración teorica
    ad = 0
    for i in range(0, d):
        
        #distribución beta -- investigar
        random_beta = betavariate( number_of_reward_1[i] +1, number_of_reward_0[i] +1 )
        
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    adds_selected.append(ad)
    reward = dataset.values[n,ad]
    total_reward = total_reward+ reward
    
    if reward == 1:
        number_of_reward_1[ad] =  number_of_reward_1[ad] +1
    else:
        number_of_reward_0[ad] =  number_of_reward_0[ad] +1
            

# histograma de resultados

plt.hist(adds_selected)
plt.title("Histograma de anuncio")
plt.xlabel("ID del anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()