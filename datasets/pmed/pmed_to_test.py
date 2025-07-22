"""
This script takes a pmed file and converts it into the test format that can be read by the standard dataset
loader class.

The files contain a subset of the total number of edges AND may contain duplicate edges but with different
values. The dataset instructions state to read in the last edge value as the correct value.

After getting this subset of edges, Floyd's algorithm is to be used in order to create the total
edge cost matrix.
"""

import json
import numpy as np

# open the optimum values file first so that we have that data on hand when we store
# the test instances.

optimal_distances = {}
with open('original_files/pmedopt.txt', 'r') as f:
    # skip the headers
    f.readline()

    while True:
        data = f.readline()
        if not data:
            break

        # line format is: test_name value
        data = data.split()
        test_name = data[0]
        value = int(data[1])
        optimal_distances[test_name] = value

for filename in optimal_distances.keys():
    with open(f'original_files/{filename}.txt', 'r') as f:
        # get the header data, the format is: number_of_vertices number_of_edges p
        headers = f.readline().split()
        num_vertices = int(headers[0])
        num_edges = int(headers[1])
        p = int(headers[2])

        # create the edge cost matrix
        edge_costs = np.zeros((num_vertices, num_vertices), dtype=np.int32)
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                # using arbitrarily large numbers to represent infinity.
                # Not using np.inf because I want to stick with integer values.
                edge_costs[i, j] = 999999999
                edge_costs[j, i] = 999999999

        # read in the actual edge values
        for _ in range(num_edges):
            # edge lines are written as: edge_start edge_end edge_value
            edge_data = f.readline().split()
            # Note that edges aren't zero indexed. We have to subtract 1 from the index
            edge_start = int(edge_data[0]) - 1
            edge_end = int(edge_data[1]) - 1
            edge_value = int(edge_data[2])

            # store the data
            edge_costs[edge_start, edge_end] = edge_value
            edge_costs[edge_end, edge_start] = edge_value

        # run Floyd's algorithm in order to get the shortest path to all nodes and therefore
        # fill out the edge cost matrix.
        # using this implementation: https://www.baeldung.com/cs/floyd-warshall-shortest-path .
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if edge_costs[i, j] > edge_costs[i, k] + edge_costs[k, j]:
                        edge_costs[i, j] = edge_costs[i, k] + edge_costs[k, j]

        # convert into .json testing format
        json_dict = {
            'distances': edge_costs.tolist(),
            'is_optimal': True,
            'distance': optimal_distances[filename],
            'n': num_vertices,
            'k': p
        }

        # save the test to disk
        filepath = f'tests/{filename}.json'
        with open(filepath, 'w') as test_file:
            test_file.write(json.dumps(json_dict))


