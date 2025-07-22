import random
import torch
import time

from solvers.brute_solver import calculate_distance

class Dominguez:
    def __init__(self, n, k, graph, use_gpu):
        self.verbose = False
        self.n = n
        self.k = k
        self.graph = graph
        self.size = (n, k)
        
        self.distanceValues = graph._normalized_distances.detach().clone()
        self.facilityAssignments = torch.zeros(self.size)
        self.clientAssignments = torch.zeros(self.size)
        
        self.mathRowIndices = torch.arange(n)
        self.mathColIndices = torch.arange(k)
        
        self.vertices = [0] * n
        
        """
        if self.verbose is True:
            print("These are the initial (normalized) distance values")
            print(self.distanceValues)
            print()
        """
        for i in range(n):
            self.vertices[i] = 0
        
    def run(self, runs):
        best_facilities = None
        best_distance = None
        for r in range(runs):
            self.initializePerRunArrays()
            updateClients = True
            stabilized = False
            numIterations = 0
            while not stabilized:
                numIterations += 1
                energyBefore = self.calculateDistance()[0]
                #print("Start of the loop, EnergyBefore: ",energyBefore)
                if updateClients:
                    self.updateClients()
                else:
                    self.updateFacilities()
                updateClients = not updateClients
                energyAfter = self.calculateDistance()[0]
                #print("EnergyAfter: ",energyAfter)
                if updateClients and energyAfter >= energyBefore:
                    stabilized = True
                    #print("Stabilized:",stabilized)
                    #print("Finished in ",numIterations," iterations")
            distance, selectedFacilities = self.calculateDistance()
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_facilities = selectedFacilities
        """
        if self.verbose is True:
            print("Returning values:",self.vertices)
        """
        return best_facilities
            
    def initializePerRunArrays(self):
        self.facilityAssignments = torch.zeros(size=self.size)
        self.clientAssignments = torch.zeros(size=self.size)
        self.activeFacilityList = []
        
        # Randomly assign a facility to each cluster
        selectedFacilities = random.sample(range(self.n), self.k)
        for index, val in enumerate(selectedFacilities):
            self.facilityAssignments[val,index] = 1
            self.activeFacilityList.append(val)

        # Randomly assign clients to clusters
        for i in range(self.n):
            index = random.randint(0, self.k - 1)
            self.clientAssignments[i,index] = 1
            
        """
        if self.verbose is True:
            print("These are the initial facility assignments")
            print(self.facilityAssignments)
            print()
        """
        
        """
        if self.verbose is True:
            print("These are the initial client assignments")
            print(self.clientAssignments)
            print()
        """
            
    def updateClients(self):
        self.clientAssignments = -self.distanceValues @ self.facilityAssignments
        """
        if self.verbose is True:
            print("Update Clients. Client values were:")
            print(self.clientAssignments)
            print()
        """
        maxIndices = torch.argmax(self.clientAssignments, dim=1)
        """
        if self.verbose is True:
            print("Max Indices are: ",maxIndices)
            print()
        #"""
        self.clientAssignments = torch.zeros(size=self.size)
        self.clientAssignments[self.mathRowIndices,maxIndices] = 1
        """
        if self.verbose is True:
            print("Client values are now:")
            print(self.clientAssignments)
            print()
        #"""
    
    def updateFacilities(self):      
        self.facilityAssignments = -self.distanceValues.T @ self.clientAssignments
        """
        if self.verbose is True:
            print("Update Facilities. Facility values were:")
            print(self.facilityAssignments)
            print()
        #"""
        self.activeFacilityList = torch.argmax(self.facilityAssignments, dim=0).tolist()
        """
        if self.verbose is True:
            print("Active facilities are:",self.activeFacilityList)
            print()
        #"""
        self.facilityAssignments = torch.zeros(size=self.size)
        self.facilityAssignments[self.activeFacilityList,self.mathColIndices] = 1     
        """
        if self.verbose is True:
            print("Facility values are now:")
            print(self.facilityAssignments)
            print()
        """        
        
    def calculateDistance(self):
        client_clusters = torch.argmax(self.clientAssignments, dim=1)
        facility_indices = torch.argmax(self.facilityAssignments, dim=0)
        total_cost = 0.0
        for i in range(self.n):
            q = client_clusters[i].item()
            j = facility_indices[q].item()
            total_cost += self.distanceValues[i,j].item()
        return total_cost, facility_indices