import json

from problems.KMProblem import KMProblem
from reader.InputReader import InputReader
from solvers.brute_solver import calculate_distance
from utils.graph import DistanceGraph


class KMPJSONReader(InputReader):
    def setNext(self, next):
        return super().setNext(next)

    def read(self, input):
        return super().read(input)

    def canRead(self, input):
        try:
            with open(input) as f:
                data = json.load(f)
                return len(data) > 0 and data['format'] == 1
        except (FileNotFoundError, json.JSONDecodeError, IndexError):
            return False
        
    # returns an instance of the problem
    def parse(self, input, use_gpu):
        with open(input) as f:
            data = json.load(f)
        name = data['name']
        n = data['n']
        k = data['k']
        distances = data['distances']
        graph = DistanceGraph(distances, use_gpu)
        optimal_distance = data['optimal_solution']

        problem = KMProblem(name ,graph, n, k, optimal_distance)

        return problem