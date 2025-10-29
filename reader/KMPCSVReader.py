import csv

from reader.InputReader import InputReader


class KMPCSVReader(InputReader):
    def setNext(self, next):
        return super().setNext(next)

    def read(self, input):
        return super().read(input)

    def canRead(self, input):
        try:
            with open(input, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[0].strip().lower() == "format":
                        return row[1].strip() == "3"
                return False  # "format" key not found
        except (FileNotFoundError, csv.Error, IndexError):
            return False
        
    def parse(self, input):
        pass