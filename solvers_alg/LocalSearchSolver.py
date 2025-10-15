import random
import sys
import time

import torch

from solvers_alg.KMPSolver import KMPSolver


class LocalSearchSolver(KMPSolver):
    def __init__(self, graph, n, k, max_time, solution=None ):
        # Initialize Variables for Solver
        self._name = "Local Search"
        self._solutionValue = 0
        self._selectedFacilities = []

        self._graph = graph
        self._n = n
        self._k = k
        self._max_time = max_time
        self._check_counter = 0
        self._swap_counter = 0

        # prepare the initial vertices
        if solution:
            vertices = []
            for i in range(self._n):
                if i in solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(self._n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, self._n)], k=self._k):
                vertices[value] = 1

        self._vertices = torch.tensor(vertices)

        if n < 1000:
            self._maxTime = 5
        elif n > 1000 and n < 1500:
            self._maxTime = 1
        elif n > 1500 and n < 3000:
            self._maxTime = 2
        elif n > 3000 and n < 5000:
            self._maxTime = 3
        elif n > 5000 and n < 6000:
            self._maxTime = 20
        elif n > 6000 and n < 15000 and k < 1000:
            self._maxTime = 50
        elif n > 6000 and n < 15000 and k == 1000:
            self._maxTime = 75
        elif n > 6000 and n < 15000 and k == 2000:
            self._maxTime = 100
        elif n > 6000 and n < 15000 and k > 2000:
            self._maxTime = 200

    def getName(self):
        return self._name

    def getSolutionValue(self):
        return self._solutionValue

    def getSelectedFacilities(self):
        return self._selectedFacilities
    
    def solve(self):
        start_time = time.time()
        best_distance = self.calculate_distance()
        while True:
            has_swapped = False
            for client in range(self._):
                if self._vertices[client] == 0:
                    self._check_counter += 1
                    # check every possible swap
                    for facility in range(self._n):
                        if time.time() - start_time >= self._maxTime:
                            break
                        if self._vertices[facility] == 1:
                            self._vertices[facility] = 0
                            self._vertices[client] = 1
                            new_distance = self.calculate_distance()
                            # if our change in distance reaches a certain threshold we stop.
                            # We use the formula from 'Effectiveness of Local Search for Geometric Optimization'
                            # (Cohen-Addad and Mathieu 2015).
                            if new_distance < (1 - (1 / self.n)) * best_distance:
                                best_distance = new_distance
                                has_swapped = True
                                break
                            else:
                                self._vertices[facility] = 1
                                self._vertices[client] = 0
                    if has_swapped is True:
                        self._swap_counter += 1
                        break
            # if we have no better candidate we stop
            if not has_swapped:
                break

        self._selectedFacilities = [i for i in range(self._n) if self._vertices[i] == 1]
        self._solutionValue = best_distance


    def calculate_distance(self):
        """
        Helper function to get the total distance of all clients to their nearest facility.

        :param vertices: The solution to check
        :return: The sum of distances from clients to their nearest facility.
        """
        # The graph contains a normalized NxN array of distance values. We use facility_tensor to isolate only the
        # distance values for the clients to facilities. Then instead of getting the min (which will include the
        # unused facilities) we use the max and flip the distance with 1 - distances.
        # The max values of this will be the minimum distance values of the actual graph.
        max_values, _ = torch.max(self._vertices * (1 - self._graph._normalized_distances), dim=1)        
        # For now, flip back to the min values by subtracting 1 again. This is superfluous and can be changed but
        # will require changing the check in the loop as well.
        return torch.sum(1 - max_values)

   
