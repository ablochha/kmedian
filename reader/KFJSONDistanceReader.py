import json

from problems.KFProblem import KFProblem
from reader.InputReader import InputReader
from utils.graph import DistanceGraph


class KFJSONDistanceReader(InputReader):
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
                return len(data) > 0 and data['format'] == 9
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
        costs = data['costs']

        problem = KFProblem(name ,graph, n, k, optimal_distance, costs)

        return problem