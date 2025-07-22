import random

import numpy as np
import torch

from algorithms.search_tree import SearchAgent
from solvers.brute_solver import calculate_distance

# Constants for update type.
FACILITY = 1
CLIENT = 0


class Hopfield:
    def __init__(self, n, k, graph, use_gpu, search_tree_config=None):
    
        self.verbose = True
        self._n = n
        self._k = k
        self._graph = graph
        self._search_tree_config = search_tree_config
        self._num_rows = n
        self._num_cols = n + 1
        self._size = (self._num_rows, self._num_cols)
        self._facility_update_value = 1.0
        self._client_update_value = 1.0
        
        # This variable keeps track of how many facilities we are evaluating.
        self._facility_length = n    
        # this variable keeps track of the facilities we are evaluating                
        self._facility_list = [i for i in range(n)] 
        # This keeps track of the specific facility neighbour groups
        self._neighbour_sets = {}  
        # Cache the mapping from a facility to it's selected row group.                 
        self._selected_row_cache = {} 
        # CPU/GPU toggle              
        self._use_gpu = use_gpu                      
        
        # If we select the GPU and cuda is not available, fail loudly.
        if use_gpu:                                 
            self._device = 'cuda' if torch.cuda.is_available() else None
            assert self._device is not None
        else:
            self._device = 'cpu' 
          
        # The graph should contain a normalized torch array of distances.
        # Subtract 1 in order to prioritize smaller distances
        if self._use_gpu:
            self._full_distance_values = 1 - graph._gpu_normalized_distances
        else:
            #self._full_distance_values = torch.tensor(1 - graph._normalized_distances)
            self._full_distance_values = (1 - graph._normalized_distances).clone().detach()
        """
        if self.verbose is True:
            print("These are the initial (normalized) distance values")
            print(self._full_distance_values)
            print()
        """
              
        self._distance_values = None
        
        # Value Tensors/Matrices
        #self._full_activation_values = torch.empty(size=self._size, device=self._device)
        self._activation_values = torch.zeros(size=self._size, device=self._device)
        #self._full_inner_values = torch.empty(size=self._size, device=self._device)
        self._inner_values = torch.zeros(size=self._size, device=self._device)
        
        # Constant Tensors/Matrices
        #self._client_addition_values = torch.full(fill_value=self._client_update_value * 2, size=(self._n,), device=self._device)
        #self._sparse_column_indices = torch.arange(start=0, end=self._n, device=self._device)
        
        # we want the columns from 1 to the end of the array (i.e., only the client values)
        self._math_column_indices = torch.arange(start=1, end=self._n + 1, device=self._device)
        
        # Test tensors to see if making comparisons on the GPU is faster
        #self._k_on_gpu = torch.tensor(self._k, device=self._device)
        #self._n_on_gpu = torch.tensor(self._n, device=self._device)
        #self._1_on_gpu = torch.tensor(1.0, device=self._device)
        
        # Caching the sorted list of a facility inner values in order to decrease the number of sort calls
        self._sorted_facility_inner_values = None
        self._sorted_facility_indices = None

    def run(self, runs, starter_facilities=None):
                   
        best_facilities = starter_facilities
        best_distance = calculate_distance(self._graph, best_facilities, self._n) if starter_facilities else None

        completed = 0
        import time
        start_time = time.time()
        max_time = 235
        
        for r in range(runs):
        
            current_time = time.time()
            
            #if current_time - start_time >= max_time:
            #    break
                
            # Initialize our per run variables.
            self._initialize_per_run_arrays(r, runs)
            
            facility_stabilized = False
            #counter = 0
            
            while not facility_stabilized:
            
                #counter = counter + 1
                # Since we start with a facility update we need to sort the facility values at the start
                self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(self._inner_values[:, 0])
                #self._sorted_facility_inner_values = self._sorted_facility_inner_values[-self._k:]
                self._sorted_facility_indices = self._sorted_facility_indices[-self._k:]
                
                worstFacility = self._sorted_facility_indices[0]
                #temp = np.where(self._inner_values[:, 0] == 0, 2, self._inner_values[:, 0])
                #worstFacility = torch.argmin(torch.from_numpy(temp))
                self._activation_values[worstFacility, 0] = 0
                
                """
                if self.verbose is True:
                    print("Beginning of the loop")
                    print("Here is the list of sorted facility inner values")
                    print(self._sorted_facility_inner_values)
                    print()
                
                    print("Here is the list of sorted facility indices")
                    print(self._sorted_facility_indices)
                    print()
                
                    print("The active facility with the worst inner value is: ", worstFacility.item())
                    print()
                """
                
                sumBefore = torch.sum(self._inner_values[:,0]).item()
                
                """
                if self.verbose is True:
                    print("Here is the sum of facility inner values before deactivation")
                    print(sumBefore)
                    print()
                
                    print("These are the activation values")
                    print(self._activation_values)
                    print()
                """
                
                self._inner_values[:, 0] = 0
                self._inner_values[worstFacility, 1:] = 0
                
                """
                if self.verbose is True:
                    print("Zeroed out the facility inner values")
                    print(self._inner_values)
                    print()
                """
                
                #self._calculate_client_values()
                
                """
                if self.verbose is True:
                    print("These are the client inner values after deactivating the worst facility")
                    print(self._inner_values)
                    print()
                """
            
                max_values, max_indices = torch.max(self._inner_values[:, 1:], dim=0)
                
                """
                if self.verbose is True:
                    print("These are the max values in each column")
                    print(max_values)
                    print()
                """
                
                facility_values = torch.reshape(self._activation_values[:, 0], (self._facility_length, 1))
                facility_values = (facility_values - 1) * -1
                #print("Facility values")
                #print(facility_values)
                self._inner_values[:, 1:] = facility_values * self._distance_values
                
                #print("These are the *modified* client inner values")
                #print(self._inner_values)
                #print()
                
                facility_values = torch.reshape(facility_values, (1, self._facility_length))
                #print("Facility values")
                #print(facility_values)
                self._inner_values[:, 1:] = facility_values * self._inner_values[:, 1:] 
                
                #print("These are the *modified* client inner values")
                #print(self._inner_values)
                #print()
                
                #vals = self._inner_values[:, 1:]
                #np.where(self._inner_values[:, 1:] > max_values, self._inner_values[:, 1:] + (self._inner_values[:, 1:] - max_values), 0)
                #np.where(self._inner_values[:, i for i in range (1, self._n)] > max_values[i-1], self._inner_values[:, 1:] + (self._inner_values[:, 1:] - max_values), 0)
                
                #for i in range(1, self._n):
                    #self._inner_values[:,i] = torch.from_numpy(np.where(self._inner_values[:,i] > max_values[i-1], self._inner_values[:,i] + (self._inner_values[:,i] - max_values[i-1]), 0))
                 
                self._inner_values[:,1:] = torch.from_numpy(np.where(self._inner_values[:,1:] > max_values, self._inner_values[:,1:] - max_values, 0))   
                 
                """
                if self.verbose is True:
                    print("Final *modified* client inner values")
                    print(self._inner_values)
                    print()
                """
                
                self._inner_values[:, 0] = torch.sum(self._inner_values[:, 1:], dim=1)
                
                """
                if self.verbose is True:
                    print("Updated facility inner values")
                    print(self._inner_values)
                    print()
                """
                
                # Sort the facilities again
                #self._sorted_facility_inner_values, self._sorted_facility_indices = torch.sort(self._inner_values[:, 0])
                #bestFacility = self._sorted_facility_indices[self._n-1]
                bestFacility = torch.argmax(self._inner_values[:, 0])               
                
                #if bestFacility.item() == worstFacility.item():
                #    facility_stabilized = True
                #    print("Stabilized")
                
                self._activation_values[bestFacility, 0] = 1
                #These are not the correct sorted inner values, because we used argmax for this calculation!
                #print("Here is the list of sorted facility inner values")
                #print(self._sorted_facility_inner_values)
                #print()
                
                """
                if self.verbose is True:
                    print("Comparing worstFacility to bestFacility")
                    print("worstFacility", worstFacility.item())
                    print("bestFacility", bestFacility.item())
                    print()
                """
                
                #print("The active facility with the best inner value is: ", bestFacility.item())
                #print()
                
                self._calculate_client_values()
                
                """
                if self.verbose is True:
                    print("These are the client inner values")
                    print(self._inner_values)
                    print()
                """
                
                self._update_client()
                
                """
                if self.verbose is True:
                    print("These are the client activation values")
                    print(self._activation_values)
                    print()
                """
                
                self._calculate_facility_values()
                
                """
                if self.verbose is True:
                    print("These are the facility inner values")
                    print(self._inner_values)
                    print()
                """
                
                sumAfter = torch.sum(self._inner_values[:,0]).item()
                
                """
                if self.verbose is True:
                    print("Here is the sum of facility inner values after activation")
                    print(sumAfter)
                    print()
                """
                
                if sumBefore >= sumAfter:
                    facility_stabilized = True
                    self._activation_values[bestFacility.item(),0] = 0
                    self._activation_values[worstFacility.item(),0] = 1
                    
                    """
                    if self.verbose is True:
                        print("Stabilized")
                    """
                    
            #print("Number of iterations of the main loop: ", counter)
            selected_facilities, selected_distance = self._calculate_facilities_and_distance()
            
            """
            if self.verbose is True:
                print("Current distance", selected_distance)
                print("Best distance", best_distance)
                print()
            """

            # update our best value
            if best_distance is None or selected_distance < best_distance:
            
                if best_distance is None:
                
                    pass
                    
                else:
                
                    new_ratio = round(best_distance / selected_distance, 3)
                    #print(f"{r}: {selected_distance} - {new_ratio}")
                    
                best_distance = selected_distance
                best_facilities = selected_facilities
                
                """
                if self.verbose is True:
                    print("These are the activation values")
                    print(self._activation_values)
                    print()   
                """

            completed += 1

        #import sys
        #sys.exit(0)
        #print(completed)
        return best_facilities

    def _initialize_per_run_arrays(self, r, runs):
    
        self._activation_values = torch.zeros(size=self._size, device=self._device)
        #self._inner_values = torch.zeros(size=self._size, device=self._device)
        
        # randomly pick k vertices as the starting facilities
        for value in random.sample([i for i in range(0, self._n)], k=self._k):
            self._activation_values[value, 0] = 1
            
        """
        Maximum Distance Initialization
        """
        #distanceSums = torch.sum(self._full_distance_values[:, :], dim=1)    
        #sumVales, sumIndices = torch.sort(distanceSums)
        #for i in range(self._k):
        #    self._activation_values[i] = 1
        
        """
        if self.verbose is True:
            print("These are the initial random activation values")
            print(self._activation_values)
            print("")
        """
        
        self._distance_values = self._full_distance_values
        self._facility_list = [i for i in range(self._n)]
        self._facility_length = self._n
        
        self._calculate_client_values()
        
        """
        if self.verbose is True:
            print("These are the initial client inner values")
            print(self._inner_values)
            print()
        """
        
        self._update_client()
        
        """
        if self.verbose is True:
            print("These are the inital client activation values")
            print(self._activation_values)
            print()
        """

        self._calculate_facility_values()
        
        """
        if self.verbose is True:
            print("These are the initial facility inner values")
            print(self._inner_values)
            print()
        """

    def _calculate_facilities_and_distance(self):
    
        selected_facilities = []
        
        for i in range(self._facility_length):
        
            if self._activation_values[i, 0] == 1:
            
                selected_facilities.append(self._facility_list[i])
                
        selected_distance = calculate_distance(self._graph, selected_facilities, self._n)
        
        return selected_facilities, selected_distance

    def _create_distance_array(self):
    
        return torch.tensor(self._graph.distance_cache, device=self._device)

    def _calculate_client_values(self):
    
        facility_values = torch.reshape(self._activation_values[:, 0], (self._facility_length, 1))
        self._inner_values[:, 1:] = facility_values * self._distance_values

    def _calculate_facility_values(self):
    
        self._inner_values[:, 0] = torch.sum(self._activation_values[:, 1:] * self._distance_values, dim=1)

    def _update_client_math(self):
    
        """
        This method sees if using indexing for calculations is faster
        for updating the client values.
        """
        max_indices = torch.argmax(self._inner_values[:, 1:], dim=0)
        
        # decrease all values
        self._activation_values[:, 1:] = 0.0
        
        # increase the max values
        self._activation_values[max_indices, self._math_column_indices] = 1.0

    def _update_client_original(self):
    
        """
        Original method to be used as a baseline for speed testing
        """
        max_indices = torch.argmax(self._inner_values[:, 1:], dim=0)
        self._activation_values[:, 1:] -= self._client_update_value
        
        for place, i in enumerate(max_indices):
        
            self._activation_values[i, place + 1] += (self._client_update_value * 2)

    def _update_client(self):
    
        old_values = self._activation_values[:, 1:].clone()
        self._update_client_math()
        return torch.ne(old_values, self._activation_values[:, 1:]).any()