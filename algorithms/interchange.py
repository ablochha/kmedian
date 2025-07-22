import time
import torch
import sys

class Interchange:

    def __init__(self, n, k, graph, use_gpu, vertices):
        self.graph = graph
        self.n = n
        self.k = k
        self.vertices = torch.tensor(vertices)
        self.check_counter = 0
        self.swap_counter = 0
        
        #fast interchange tensors
        self.gain = torch.zeros(size=(self.n,))
        self.loss = torch.zeros(size=(self.n,))
        closest_values, closest_indices = torch.topk(self.vertices * (1 - self.graph._normalized_distances), k=2, dim=1) 
        closest_values = 1 - closest_values
        self.d1 = closest_values[:,0]
        self.d1_indices = closest_indices[:,0]
        self.d2 = closest_values[:,1]
        self.d2_indices = closest_indices[:,1]
        self.extra = torch.zeros(size=(self.n, self.n))
        #print("Normalized Distances:")
        #print(self.graph._normalized_distances)
        #print("Vertices:")
        #print(self.vertices)
        #print("Closest Values:")
        #print(closest_values)
        #print("D1 indices:")
        #print(self.d1_indices)
        #print("D2 indices:")
        #print(self.d2_indices)
        
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
        
    def run(self, runs):
        self.start_time = time.time()
        self.affected_clients = [i for i in range(self.n) if self.vertices[i] == 0]
        self.resetStructures()
        while True:
            for client in self.affected_clients: self.updateStructures(client)
            if time.time() - self.start_time >= self.maxTime:
                break
            removed_facility, inserted_facility, profit = self.findBestNeighbor()
            if time.time() - self.start_time >= self.maxTime:
                break
            if profit <= 0: break
            self.affected_clients = []
            for client in range(self.n):
                if self.vertices[client] == 0:
                    if removed_facility == self.d1_indices[client] or removed_facility == self.d2_indices[client] or self.graph._normalized_distances[client,inserted_facility] < self.graph._normalized_distances[client,self.d2_indices[client]]:
                        self.affected_clients.append(client)
            for client in self.affected_clients: self.undoUpdateStructures(client)
            if time.time() - self.start_time >= self.maxTime:
                break
            self.vertices[inserted_facility] = 1
            self.vertices[removed_facility] = 0
            self.updateClosest()
        
        
    def resetStructures(self):
        self.gain = torch.zeros(size=(self.n,))
        self.loss = torch.zeros(size=(self.n,))
        self.extra = torch.zeros(size=(self.n, self.n))
        
    def updateStructures(self, client):
        facility_to_remove = self.d1_indices[client]
        self.loss[facility_to_remove] += self.d2[client] - self.d1[client]
        for facility_to_insert in range(self.n):
            if time.time() - self.start_time >= self.maxTime:
                break
            if self.vertices[facility_to_insert] == 0:
                if self.graph._normalized_distances[client, facility_to_insert] < self.d2[client]:
                    self.gain[facility_to_insert] += max(0, self.d1[client] - self.graph._normalized_distances[client,facility_to_insert])
                    self.extra[facility_to_insert,facility_to_remove] += self.d2[client] - max(self.graph._normalized_distances[client,facility_to_insert], self.d1[client])
        
    def undoUpdateStructures(self, client):
        facility_to_remove = self.d1_indices[client]
        self.loss[facility_to_remove] -= self.d2[client] - self.d1[client]
        for facility_to_insert in range(self.n):
            if time.time() - self.start_time >= self.maxTime:
                break
            if self.vertices[facility_to_insert] == 0:
                if self.graph._normalized_distances[client, facility_to_insert] < self.d2[client]:
                    self.gain[facility_to_insert] -= max(0, self.d1[client] - self.graph._normalized_distances[client,facility_to_insert])
                    self.extra[facility_to_insert,facility_to_remove] -= self.d2[client] - max(self.graph._normalized_distances[client,facility_to_insert], self.d1[client])
        
    def findBestNeighbor(self):
        currProfit = -1
        bestProfit = -1
        bestInsert = -1
        bestRemove = -1
        for currInsert in range(self.n):
            if time.time() - self.start_time >= self.maxTime:
                break
            if self.vertices[currInsert] == 0:
                for currRemove in range(self.n):
                    if self.vertices[currRemove] == 1:
                        currProfit = self.gain[currInsert] - self.loss[currRemove] + self.extra[currInsert,currRemove]
                        if currProfit > bestProfit:
                            bestProfit = currProfit
                            bestInsert = currInsert
                            bestRemove = currRemove
        return bestRemove, bestInsert, bestProfit  
        
    def updateClosest(self):
        closest_values, closest_indices = torch.topk(self.vertices * (1 - self.graph._normalized_distances), k=2, dim=1) 
        closest_values = 1 - closest_values
        self.d1 = closest_values[:,0]
        self.d1_indices = closest_indices[:,0]
        self.d2 = closest_values[:,1]
        self.d2_indices = closest_indices[:,1]