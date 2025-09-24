import time
import torch
import sys


class LocalSearch:
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
        
        """
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
        """
        if n < 1000:
            self.maxTime = 5
        elif n > 1000 and n < 1500:
            self.maxTime = 1
        elif n > 1000 and n < 3000:
            self.maxTime = 2
        elif n > 1000 and n < 5000:
            self.maxTime = 3
        elif n > 1000 and n < 6000: 
            self.maxTime = 20
        elif n > 1000 and n < 15000 and k < 1000:
            self.maxTime = 50
        elif n > 1000 and n < 15000 and k == 1000:
            self.maxTime = 75
        elif n > 1000 and n < 15000 and k == 2000:
            self.maxTime = 100
        elif n > 1000 and n < 15000 and k > 2000:
            self.maxTime = 200

    """
    Randomly select a client and compare swapping it with each facility.
    If the distance is improved, we select the swap pair that decreases the distance the most.
    Otherwise, keep the original state.
    :return: None
    """
    def run(self):
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
        # The graph contains a normalized NxN array of distance values. We use facility_tensor to isolate only the
        # distance values for the clients to facilities. Then instead of getting the min (which will include the
        # unused facilities) we use the max and flip the distance with 1 - distances.
        # The max values of this will be the minimum distance values of the actual graph.
        max_values, _ = torch.max(self.vertices * (1 - self.graph._normalized_distances), dim=1)        
        # For now, flip back to the min values by subtracting 1 again. This is superfluous and can be changed but
        # will require changing the check in the loop as well.
        return torch.sum(1 - max_values)