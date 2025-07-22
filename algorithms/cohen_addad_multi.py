import time
import torch
import pandas as pd
import numpy as np
import random

class CohenAddadMulti:
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
        
        if n > 6000 and k > 2000:
            self.maxTime = 200
        elif n > 6000 and k <= 2000:
            self.maxTime = 100
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
        best_distance = self.calculate_distance(self.vertices)
        while True:
            if time.time() - start_time >= self.maxTime:
                break
            open_locations = [i for i in range(self.n) if self.vertices[i] == 0]
            facilities = [i for i in range(self.n) if self.vertices[i] == 1]
            has_swapped = False
            while has_swapped is False:
                if time.time() - start_time >= self.maxTime:
                    break
                client1, client2 = random.sample(open_locations, k=2)
                facility1, facility2 = random.sample(facilities, k=2)
                self.check_counter += 1
                temp_vertices = self.vertices.detach().clone()
                temp_vertices[facility1] = 0
                temp_vertices[facility2] = 0
                temp_vertices[client1] = 1
                temp_vertices[client2] = 1
                new_distance = self.calculate_distance(temp_vertices)
                if new_distance < (1 - (1 / self.n)) * best_distance:
                    best_distance = new_distance
                    self.vertices[facility1] = 0
                    self.vertices[facility2] = 0
                    self.vertices[client1] = 1
                    self.vertices[client2] = 1
                    has_swapped = True
                    self.swap_counter += 1   

    def calculate_distance(self, vertices):
        """
        Helper function to get the total distance of all clients to their nearest facility.

        :param vertices: The solution to check
        :return: The sum of distances from clients to their nearest facility.
        """
        values, _ = torch.topk(vertices * (1 - self.graph._normalized_distances), k=2, dim=1)        
        closest = 1 - values[:,0]
        secondClosest = 1 - values[:,1]
        costs = closest + (0.2 * torch.minimum(secondClosest,(3 * closest)))
        ret = torch.sum(costs)
        return ret