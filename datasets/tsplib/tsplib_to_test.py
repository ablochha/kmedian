"""
This script takes a tsplib file and converts it into the test format that can be read by the standard dataset
loader class.

"""

import csv
import json

#from datasets.tsplib.load_tsplib_test import load_test
from load_tsplib_test import load_test
#from utils.graph import CoordinateGraph
from graph import CoordinateGraph

# Get the value data from the CSV file
# The format will be: test name, n, k, optimal (or best) value
test_values = []
with open('test_values.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        test_values.append(row)

    # remove the header row
    test_values.pop(0)

# Go through the CSV file list and for each test load the data and save it in our test
# format
for test in test_values:
    test_name, n, k, optimum = test
    if test_name != "rl5934":
        continue
    n = int(n)
    k = int(k)
    optimum = int(optimum)

    # some names appear to have typos:
    if test_name == 'v1748':
        test_name = 'vm1748'

    _, x, y = load_test(test_name)

    graph = CoordinateGraph(x, y, use_gpu=False)

    # convert into .json testing format
    json_dict = {
        'x': x,
        'y': y,
        'is_optimal': True,
        'distance': optimum,
        'n': n,
        'k': k
    }

    # save the test to disk
    filepath = f'tests/{test_name}_k{k}.json'
    with open(filepath, 'w') as test_file:
        test_file.write(json.dumps(json_dict))


