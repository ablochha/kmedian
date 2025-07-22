import random

from algorithms.haralampiev import calculate_temperature_decay, HaralampievNetwork
from algorithms.hopfield import Hopfield
from algorithms.local_search import LocalSearch
from algorithms.arya_multi import AryaMulti
from algorithms.mra import MRA
from algorithms.cohen_addad import CohenAddad
from algorithms.cohen_addad_multi import CohenAddadMulti
from algorithms.hopfield_original2nk import HopfieldOriginal
from algorithms.hopfield_2nkBestHalf import HopfieldBestHalfSingle
from algorithms.hopfield_2nkMultiBestHalf import HopfieldBestHalfMulti
from algorithms.hopfield_2nkSecondClosestBestHalf import HopfieldBestHalfSecondClosest
from algorithms.hopfield_2nkTimesK import HopfieldExhaustive
from algorithms.interchange import Interchange
from algorithms.dominguez import Dominguez
from solvers.ilp_solver import FacilityLocationSolver

from solvers.brute_solver import get_facilities, calculate_distance

"""
n: the number of clients
k: the number of facilities
"""

class FacilityLocationAlgorithm:
    def name(self):
        raise NotImplementedError

    def run(self, graph, n, k, solution=None):
        raise NotImplementedError


class RandomChoiceAlgorithm(FacilityLocationAlgorithm):
    """
    This class exists for two purposes:
    1. See if an approximation is doing well or just appearing to do well.
    2. This class runs quickly, so it can be used to test that datasets have been constructed properly.
    """

    def name(self):
        return "Random Choice"

    def run(self, graph, n, k, solution=None):
        selection = random.sample([i for i in range(n)], k=k)
        return selection
    
class HopfieldAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Hopfield"

    def run(self, graph, n, k, solutions=None):
        instance = Hopfield(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs, solutions)

        return selected_facilities
    
class HaralampievAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, temperature, epoch_length, decay_interval, runs):
        self.temperature = temperature
        self.epoch_length = epoch_length  # Set to none in order to dynamically get n at runtime
        self.alpha = calculate_temperature_decay(self.temperature, decay_interval)
        self.runs = runs

    def name(self):
        return "Haralampiev Network"

    def run(self, graph, n, k, solution=None):
        instance = HaralampievNetwork(n, k, graph)
        warm_instance = RandomChoiceAlgorithm()
        warm_solution = warm_instance.run(graph, n, k)
        instance.warm_start(warm_solution)
        # Epoch length can be variable so we allow a None assignment in order to signal that we should use n
        if self.epoch_length is None:
            epoch_length = n
        else:
            epoch_length = self.epoch_length

        best_facilities = None
        best_distance = None
        for _ in range(self.runs):
            instance.run(temperature=self.temperature, epoch_length=epoch_length, alpha=self.alpha)
            facilities = get_facilities(instance, n, k)

            current_distance = calculate_distance(graph, facilities, n)
            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_facilities = facilities

            instance.reset()
            instance.warm_start(best_facilities)

        return best_facilities

class LocalSearchAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, max_time):
        self.max_time = max_time

    def name(self):
        return "Local Search"

    def run(self, graph, n, k, solution=None):
        # prepare the initial vertices
        if solution:
            vertices = []
            for i in range(n):
                if i in solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, n)], k=k):
                vertices[value] = 1

        instance = LocalSearch(graph, n, k, self.max_time, vertices)
        instance.run()
        facilities = [i for i in range(n) if instance.vertices[i] == 1]
        return facilities
        
class AryaMultiAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, max_time):
        self.max_time = max_time

    def name(self):
        return "Arya Multi"

    def run(self, graph, n, k, solution=None):
        # prepare the initial vertices
        if solution:
            vertices = []
            for i in range(n):
                if i in solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, n)], k=k):
                vertices[value] = 1

        instance = AryaMulti(graph, n, k, self.max_time, vertices)
        instance.run()
        facilities = [i for i in range(n) if instance.vertices[i] == 1]
        return facilities
        
        
class ZhuAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, max_time):
        self.max_time = max_time

    def name(self):
        return "Zhu's Algorithm (MRA)"

    def run(self, graph, n, k, solution=None):
        # prepare the initial vertices
        if solution:
            vertices = []
            for i in range(n):
                if i in solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, n)], k=k):
                vertices[value] = 1

        instance = MRA(graph, n, k, self.max_time, vertices)
        instance.run()
        facilities = [i for i in range(n) if instance.vertices[i] == 1]
        return facilities
        
class CohenAddadAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, max_time):
        self.max_time = max_time

    def name(self):
        return "Cohen-Addad Local Search"

    def run(self, graph, n, k, solution=None):
        # prepare the initial vertices
        if solution:
            vertices = []
            for i in range(n):
                if i in solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, n)], k=k):
                vertices[value] = 1

        instance = CohenAddad(graph, n, k, self.max_time, vertices)
        instance.run()
        facilities = [i for i in range(n) if instance.vertices[i] == 1]
        return facilities
        
class CohenAddadMultiAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, max_time):
        self.max_time = max_time

    def name(self):
        return "Cohen-Addad Multi"

    def run(self, graph, n, k, solution=None):
        # prepare the initial vertices
        if solution:
            vertices = []
            for i in range(n):
                if i in solution:
                    vertices.append(1)
                else:
                    vertices.append(0)
        else:
            vertices = [0 for _ in range(n)]
            # randomly pick k vertices
            for value in random.sample([i for i in range(0, n)], k=k):
                vertices[value] = 1

        instance = CohenAddadMulti(graph, n, k, self.max_time, vertices)
        instance.run()
        facilities = [i for i in range(n) if instance.vertices[i] == 1]
        return facilities
        
class HopfieldOriginalAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Hopfield (original 2nk)"

    def run(self, graph, n, k, solutions=None):
        instance = HopfieldOriginal(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs, solutions)

        return selected_facilities
        
class HopfieldBestHalfSingleAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Hopfield 2nk Best Half Single)"

    def run(self, graph, n, k, solutions=None):
        instance = HopfieldBestHalfSingle(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs, solutions)

        return selected_facilities        
        
class HopfieldBestHalfMultiAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Hopfield 2nk Best Half Multi"

    def run(self, graph, n, k, solutions=None):
        instance = HopfieldBestHalfMulti(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs, solutions)

        return selected_facilities 

class HopfieldBestHalfSecondClosestAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Hopfield 2nk Best Half Second Closest"

    def run(self, graph, n, k, solutions=None):
        instance = HopfieldBestHalfSecondClosest(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs, solutions)

        return selected_facilities 

class HopfieldExhaustiveAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Hopfield 2nk Exhaustive"

    def run(self, graph, n, k, solutions=None):
        instance = HopfieldExhaustive(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs, solutions)

        return selected_facilities 
        
class InterchangeAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Fast Interchange"

    def run(self, graph, n, k, solutions=None):
        vertices = [0 for _ in range(n)]
        # randomly pick k vertices
        for value in random.sample([i for i in range(0, n)], k=k):
            vertices[value] = 1
        instance = Interchange(n, k, graph, self._use_gpu, vertices)
        instance.run(self._runs)
        facilities = [i for i in range(n) if instance.vertices[i] == 1]
        return facilities

        return selected_facilities 
        
class DominguezAlgorithm(FacilityLocationAlgorithm):
    def __init__(self, runs, use_gpu):
        self._runs = runs
        self._use_gpu = use_gpu

    def name(self):
        return "Dominguez"

    def run(self, graph, n, k, solutions=None):
        instance = Dominguez(n, k, graph, self._use_gpu)
        selected_facilities = instance.run(self._runs)

        return selected_facilities 
        
class ILPSolverAlgorithm(FacilityLocationAlgorithm):
    def __init__(self):
        self.unused = None
        
    def name(self):
        return "ILPSolver"
        
    def run(self, graph, n, k):
        instance = FacilityLocationSolver(graph, n, k)
        selected_facilities = instance.solve()
        return selected_facilities