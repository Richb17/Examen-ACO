# @Time    : 22/09/2022
# @Author  : github.com/Richb17
import numpy as np

def calc_gain(best_x, value_matrix, weights_matrix):
    currentCapacity = 10
    result = [0]
    route = []
    for i in range(best_x.shape[0]):
        result[0] += value_matrix[0][best_x[i]]
        currentCapacity -= weights_matrix[0][best_x[i]]
        route.append(best_x[i])
        if currentCapacity-weights_matrix[0][best_x[i+1]] < 0:
            break 
    result.append(10-currentCapacity)
    return result, route

class ACA_KP:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 value_matrix=None,
                 weights_matrix=None,
                 alpha=1, beta=2, rho=0.1, capacity=10
                 ):
        self.func = func
        self.value_matrix = value_matrix
        self.weights_matrix = weights_matrix
        self.n_dim = n_dim  # number of dimensions (variables)
        self.size_pop = size_pop  # number of ants
        self.max_iter = max_iter  # iterations
        self.capacity = capacity # capacity of the backpack 
        self.alpha = alpha  # Degree of importance of pheromones
        self.beta = beta  # Importance of Adaptability
        self.rho = rho  # Pheromone volatilization rate     
        self.prob_matrix_weights = 1 / (weights_matrix + 1e-10 * np.eye(n_dim, n_dim))  # Avoiding division by zero errors
        self.Tau = np.ones((n_dim, n_dim))  #Pheromone matrix, updated with each iteration
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)  # The crawling path of each ant in a certain generation
        self.y = None  # Total distance crawled by each ant in a certain generation
        self.generation_best_X, self.generation_best_Y, self.generation_best_Value = [], [], []  # Record the best of each generation
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y  # Historical reasons, in order to maintain unity
        self.value_best_history = self.generation_best_Value
        self.best = []
        
    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):  # For each iteration
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_weights) ** self.beta  # Transfer probabilities without normalization.
            for j in range(self.size_pop):  # For each ant
                self.Table[j, 0] = 0  # start point, in fact, can be random, but there is no difference
                for k in range(self.n_dim - 1):  # Each node reached by ants
                    taboo_set = set(self.Table[j, :k + 1])  # Points already passed and the current point, can not pass again
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # Choose between these points
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # Probabilistic normalization
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point       
            # Calculate distance
            y = np.array([self.func(i) for i in self.Table]) 
            # By the way, record the best historical situation
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            result, route = calc_gain(x_best, self.value_matrix, self.weights_matrix)
            self.generation_best_X.append(route)
            self.generation_best_Y.append(result[1])       
            self.generation_best_Value.append(result[0])
            # Calculation of the pheromone that needs to be newly applied
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # Each ant
                for k in range(self.n_dim - 1):  # Per Node
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # Ants crawl from n1 node to n2 node
                    delta_tau[n1, n2] += 1 / y[j]  # Applied pheromones
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # Ants crawl back to the first node from the last node
                delta_tau[n1, n2] += 1 / y[j]  # Applying pheromones        
            # Pheromone drifting + pheromone smearing
            self.Tau = (1 - self.rho) * self.Tau + delta_tau
        best_generation = np.array(self.generation_best_Value).argmax()
        self.best.append(self.generation_best_X[best_generation])
        self.best.append(self.generation_best_Y[best_generation])
        self.best.append(self.generation_best_Value[best_generation])
        return self.best
    fit = run