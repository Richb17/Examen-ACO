from __future__ import division
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
from ACO_KP import ACO_KP

POP = 2
ITERATIONS = 300

coord = pd.read_csv("files/KP_coord.txt", sep=",")	
points_coordinate = coord.to_numpy(dtype = np.float64)

weights = pd.read_csv("files/KP_weights.txt", sep=",")
weights_matrix = weights.to_numpy(dtype = np.float64)

value = pd.read_csv("files/KP_value.txt", sep=",")
value_matrix = value.to_numpy(dtype = np.float64)

num_points = len(points_coordinate)

def calc_total_sum(routine):
    num_points, = routine.shape
    return sum([weights_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def 

def main():
    aca = ACA_TSP(func=calc_total_sum, n_dim=num_points,
                  size_pop=POP, max_iter=ITERATIONS,
                  distance_matrix=weights_matrix,)
    best_x, best_y = aca.run()
    # Plot the result
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    for index in range(0, len(best_points_)):
        ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    ax[0].set_title('Population %.2f, Iteration %.2f' % (POP, ITERATIONS))
    ax[1].set_title('Result: %.2f' % (best_y))
    plt.show()


if __name__ == "__main__":
    main()