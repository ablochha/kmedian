import time
import torch
import pandas as pd
import numpy as np

class CohenAddad:
    """
    This class performs a hopfield-huge.txt search to improve a solution to the facility location problem.
    It contains a vertex array where each vertex has a value 0 for client or 1 for facility.
    It operates by randomly selecting a client and swapping it with a facility if the total distance is
    reduced.
    """

    def __init__(self, graph, n, k, max_time, vertices):
        self.graph = graph
        self.n = n
        self.k = k
        self.max_time = max_time
        self.vertices = torch.tensor(vertices)
        self.check_counter = 0
        self.swap_counter = 0
        
        if n > 6000:
            self.maxTime = 200
        elif n < 1000:
            self.maxTime = 5
        elif n > 1000 and k < 1000:
            self.maxTime = 10
        elif n > 1000 and k >= 1000:
            self.maxTime = 15
        else:
            self.maxTime = 15

    def run(self):
        """
        Randomly select a client and compare swapping it with each facility.
        If the distance is improved, we select the swap pair that decreases the distance the most.
        Otherwise, keep the original state.

        :return: None
        """
        start_time = time.time()
        best_distance = self.calculate_distance()
        while True:
            has_swapped = False
            for client in range(self.n):
                if self.vertices[client] == 0:
                    self.check_counter += 1
                    # check every possible swap
                    for facility in range(self.n):
                        if time.time() - start_time >= self.maxTime:
                            break
                        if self.vertices[facility] == 1:
                            self.vertices[facility] = 0
                            self.vertices[client] = 1
                            new_distance = self.calculate_distance()
                            # if our change in distance reaches a certain threshold we stop.
                            # We use the formula from 'Effectiveness of Local Search for Geometric Optimization'
                            # (Cohen-Addad and Mathieu 2015).
                            if new_distance < (1 - (1 / self.n)) * best_distance:
                                best_distance = new_distance
                                has_swapped = True
                                break
                            else:
                                self.vertices[facility] = 1
                                self.vertices[client] = 0 
                    if has_swapped is True:
                        self.swap_counter += 1
                        break
            # if we have no better candidate we stop
            if not has_swapped:
                break   

    def calculate_distance(self):
        """
        Helper function to get the total distance of all clients to their nearest facility.

        :param vertices: The solution to check
        :return: The sum of distances from clients to their nearest facility.
        """
        values, _ = torch.topk(self.vertices * (1 - self.graph._normalized_distances), k=2, dim=1)        
        closest = 1 - values[:,0]
        secondClosest = 1 - values[:,1]
        costs = closest + (0.2 * torch.minimum(secondClosest,(3 * closest)))
        ret = torch.sum(costs)
        return ret