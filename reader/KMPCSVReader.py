import json

from reader.InputReader import InputReader


class KMPCSVReader(InputReader):
    def setNext(self, next):
        return super().setNext(next)

    def read(self, input):
        return super().read(input)

    def canRead(self, input):
        try:
            with open(input) as f:
                data = json.load(f)
                return len(data) > 0 and data[0] == 2
        except (FileNotFoundError, json.JSONDecodeError, IndexError):
            return False