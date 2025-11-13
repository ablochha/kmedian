from abc import ABC, abstractmethod


class InputReader(ABC):
    @abstractmethod
    def __init__(self):
        self._next = None

    @abstractmethod
    def setNext(self, next):
        self._next = next
        return next
    
    @abstractmethod
    def read(self, input):
        if(self.canRead(input)):
            return self.parse(input)
        elif(self._next is not None):
            return self._next.read(input)
        else:
            raise Exception(f"Unsupported file format: {input}")
        

    @abstractmethod
    def canRead(self, input):
        pass

    @abstractmethod
    def parse(self, input):
        pass