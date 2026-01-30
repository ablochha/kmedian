from problems.Problem import Problem


class CKMProblem(Problem):
    
    def __init__(self, name, graph, n, k, capacity, optimal):
        self._name = name
        self._graph = graph
        self._n = n
        self._k = k
        self._capacity = capacity
        self._optimal = optimal

    def getName(self):
        return self._name

    def getOptimal(self):
        return self._optimal

    def setOptimal(self, optimal):
        self._optimal = optimal

    def getN(self):
        return self._n

    def setN(self, n):
        self._n = n

    def getK(self):
        return self._k
    
    def setK(self, k):
        self._k = k

    def getGraph(self):
        return self._graph
    
    def setGraph(self, graph):
        self._graph = graph

    def getCapacity(self):
        return self._capacity
    
    def setCapacity(self, capacity):
        self._capacity = capacity