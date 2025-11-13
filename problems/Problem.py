# Interface for different problems
from abc import ABC, abstractmethod


class Problem(ABC):

    @abstractmethod
    def getName(self):
        pass

    @abstractmethod
    def getOptimal(self):
        pass

    def setOptimal(self):
        pass

    @abstractmethod
    def getN(self):
        pass

    def setN(self):
        pass
