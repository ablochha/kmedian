import time
import torch
import random

"""
This class performs a hopfield-huge.txt search to improve a solution to the facility location problem.

It contains a vertex array where each vertex has a value 0 for client or 1 for facility.
It operates by randomly selecting a client and swapping it with a facility if the total distance is
reduced.
"""
class AryaMulti:

    def __init__(self, graph, n, k, max_time, vertices):
    
        self.graph = graph
        self.n = n
        self.k = k
        self.max_time = max_time
        self.vertices = vertices
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
        
    """
    Randomly select a client and compare swapping it with each facility.
    If the distance is improved, we select the swap pair that decreases the distance the most.
    Otherwise, keep the original state.

    :return: None
    """
    def run(self):

        #print("START")
        start_time = time.time()
        best_distance = self.calculate_distance(self.vertices)
        #print("Max time:", self.max_time)
        
        while True:
        
            #print("IN LOOP")
            open_locations = [i for i in range(self.n) if self.vertices[i] == 0]
            facilities = [i for i in range(self.n) if self.vertices[i] == 1]
            has_swapped = False
            
            while has_swapped is False:
            
                client1, client2 = random.sample(open_locations, k=2)
                facility1, facility2 = random.sample(facilities, k=2)
                self.check_counter += 1
                #facility = None
            
                temp_vertices = self.vertices.copy()
                temp_vertices[facility1] = 0
                temp_vertices[facility2] = 0
                temp_vertices[client1] = 1
                temp_vertices[client2] = 1
                new_distance = self.calculate_distance(temp_vertices)
            
                if new_distance < (1 - (1 / self.n)) * best_distance:
                        
                    #print("UPDATE SOLUTION")
                    best_distance = new_distance
                    self.vertices[facility1] = 0
                    self.vertices[facility2] = 0
                    self.vertices[client1] = 1
                    self.vertices[client2] = 1
                    has_swapped = True
                    self.swap_counter += 1   
                
                if time.time() - start_time >= self.maxTime:
                    
                    #print("HIT TIME LIMIT")
                    break 
                    
            if time.time() - start_time >= self.maxTime:
                    
                #print("HIT TIME LIMIT")
                break       

    """
    Helper function to get the total distance of all clients to their nearest facility.
    :param vertices: The solution to check
    :return: The sum of distances from clients to their nearest facility.
    """
    def calculate_distance(self, vertices):
    
        facility_tensor = torch.tensor(vertices)
        max_values, _ = torch.max(facility_tensor * (1 - self.graph._normalized_distances), dim=1)
        return torch.sum(1 - max_values)
