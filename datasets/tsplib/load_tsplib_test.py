"""
This script converts a tsplib *.tsp file into a json test that our code expects.
"""

def load_test(filename):
    with open(f'original_files/{filename}.tsp', 'r') as f:
        # read the top level values.
        f.readline()  # ignore the name
        f.readline()  # ignore the comment
        f.readline()  # ignore the type
        n = int(f.readline().split()[-1])  # n is the last value in the space separated line
        f.readline()  # ignore the edge_weight_type
        f.readline()  # ignore the header for the node values

        x = []
        y = []
        # read each node
        for _ in range(n):
            # values are: node_id, node_x, node_y
            line = f.readline().split()
            x.append(float(line[1]))
            y.append(float(line[2]))

    # Because this is a TSP problem we do not have a specified k value or an optimal solution. In this case we
    # just return the n and coordinate values

    return n, x, y
