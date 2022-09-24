from __future__ import division
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from ACO import ACA_KP

POP = 50
ITERATIONS = 25

coord = pd.read_csv("files/KP_coord.txt", sep=",")	
points_coordinate = coord.to_numpy(dtype = np.float64)

weights = pd.read_csv("files/KP_weights.txt", sep=",")
weights_matrix = weights.to_numpy(dtype = np.float64)

value = pd.read_csv("files/KP_value.txt", sep=",")
value_matrix = value.to_numpy(dtype = np.float64)

names = ["Bolsa de Centenarios", "Fajo de billetes de $1000", "Joyero Grande", 
         "Joyero Pequeño", "Coleccion de Estampillas", "Obra de arte", "Pisapapeles de Oro"]

num_points = len(points_coordinate)

def calc_total_sum(routine):
    num_points, = routine.shape
    return sum([weights_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def printBackPack(best):
    print('\n#hormigas: %d --- #iteraciones: %d' %(POP, ITERATIONS))
    print('\nLa mochila peso %.2fkg. con un valor total de $%.2f'%(best[1], best[2]))
    print('\nSu contenido era el siguiente:\n')
    for i in range(len(best[0])-1):
        #print(best[0][i+1])
        print('%s - $%.2f - %.2fKg.'%(names[best[0][i+1]-1], value_matrix[0][best[0][i+1]], weights_matrix[0][best[0][i+1]]))
    print('\n')

def main():
    aca = ACA_KP(func=calc_total_sum, n_dim=num_points,
                  size_pop=POP, max_iter=ITERATIONS,
                  weights_matrix=weights_matrix, value_matrix=value_matrix,
                  beta=1,rho=0.5)
    best = aca.run()
    #print('Prueba #', i+1)
    printBackPack(best)
    #result, rte = calc_gain(best_x)
    #print('\n$',result[0],'\n', rte,'\n',result[1],'kg.\n')
    # Plot the result
    #fig, ax = plt.subplots(1, 2)
    #best_points_ = best[0]
    #best_points_coordinate = points_coordinate[best_points_, :]
    #for index in range(0, len(best_points_)):
        #ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    #ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    #pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    #ax[0].set_title('Population %.2f, Iteration %.2f' % (POP, ITERATIONS))
    #ax[1].set_title('Weight: %.2f, Value: $%.2f' % (best[1], best[2]))
    #plt.show()

if __name__ == "__main__":
    main()