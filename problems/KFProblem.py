from problems.Problem import Problem


class KFProblem(Problem):
    
    def __init__(self, name, graph, n, k, optimal, costs):
        self._name = name
        self._graph = graph
        self._n = n
        self._k = k
        self._optimal = optimal
        self._costs = costs

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

    def getCosts(self):
        return self._costs
    
    def setCosts(self, costs):
        self._costs = costs