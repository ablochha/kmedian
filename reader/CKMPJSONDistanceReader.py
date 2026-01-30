import json

from problems.CKMProblem import CKMProblem
from reader.InputReader import InputReader
from utils.graph import DistanceGraph


class CKMPJSONDistanceReader(InputReader):
    def __init__(self):
        self._next = None

    def setNext(self, next):
        return super().setNext(next)

    def read(self, input):
        return super().read(input)

    def canRead(self, input):
        try:
            with open(input) as f:
                data = json.load(f)
                return len(data) > 0 and data['format'] == 7
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
        capacity = data['capacity']

        problem = CKMProblem(name ,graph, n, k, capacity, optimal_distance)

        return problem