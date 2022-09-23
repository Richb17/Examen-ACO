# @Time    : 22/09/2022
# @Author  : github.com/Richb17
import numpy as np
import random 

class ACO_KP:
    def __init__(self, func, n_dim,
                 population=10, iterations=100,
                 weights_matrix=None, value_matrix=None,
                 alpha=1, beta=1, rho=0.2,
                 capacity = 10, ):
        self.func = func
        self.n_dim = n_dim #number of dimensions
        self.value_matrix = value_matrix.reshape(n_dim)
        self.population = population #number of ants
        self.iterations = iterations #number of iterations
        self.alpha = alpha #degree of importance of pheromones
        self.beta = beta #degree of importance of adaptability
        self.rho = rho #Pheromone volatilization constant
        self.tau = np.ones((n_dim, n_dim))  #Pheromone matrix, updated with each step
        self.prob_matrix_weights = 1 / (weights_matrix + 1e-10 * np.eye(n_dim, n_dim))  # Avoiding division by zero errors
        self.Table = np.zeros((population, n_dim)).astype(np.int)  # The crawling path of each ant in a certain generation
        self.capacity = capacity #Maximum weight of the backpack
        self.y = None  # Total distance crawled by each ant in a certain generation
        self.generation_best_X, self.generation_best_Y, self.generation_best_value = [], [], []  # Record the best of each generation
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y # Historical reasons, in order to maintain unity
        self.value_best_history = self.generation_best_value
        self.best_value = None
        self.best_x, self.best_y = None, None
        
    def run(self, iterations=None):
        self.iterations = iterations or self.iterations
        for i in range(self.iterations):
            probability_matrix = (self.tau ** self.alpha) * (self.prob_matrix_weights) ** self.beta
            for j in range(self.population):  # For each ant
                self.Table[j,0] = 0
                for k in range(self.n_dim - 1):  # For each object
                    passed_set = set(self.Table[j, :k + 1]) #objects passed
                    not_passed_list = list(set(range(self.n_dim)) - passed_set) #objects not yet passed
                    prob = probability_matrix[self.Table[j,k],not_passed_list]
                    prob = prob / prob.sum()
                    next_point = np.random.choice(not_passed_list, size=1,p=prob)[0]
                    self.Table[j,k+1] = next_point
            # Calculate distance
            route, weight = np.array([self.func(i,self.capacity) for i in self.Table])
            value = 0
            print(route)
            #for it in range(len(route)):
             #   value += self.value_matrix[route[it]]
            # By the way, record the best historical situation
            x_best, y_best = route, value
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)
            self.generation_best_value.append(value)  
            # Calculation of the pheromone that needs to be newly applied
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.population):  # Each ant
                for k in range(self.n_dim - 1):  # Per Node
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # Ants crawl from n1 node to n2 node
                    delta_tau[n1, n2] += 1 / weight[1][j]  # Applied pheromones
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # Ants crawl back to the first node from the last node
                delta_tau[n1, n2] += 1 / weight[j]  # Applying pheromones        
            # Pheromone drifting + pheromone smearing
            self.tau = (1 - self.rho) * self.tau + delta_tau
        best_generation = np.array(self.generation_best_Value).argmax()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        self.best_value = self.generation_best_value[best_generation]
        return self.best_x, self.best_y, self.best_value
    fit = run