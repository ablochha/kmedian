
def load_xy_values():
    with open("datasets/usca312/usca312_xy.txt", "r") as f:
        # Skip the first 7 comment lines.
        for _ in range(7):
            f.readline()
        # Read everything else into as string so that we can close the file
        data = f.read()

    # parse the giant string of data points
    # here the data is a series of integers where the first int is the x value and the second is the y value.
    x_values = []
    y_values = []
    tokens = data.split()
    for _ in range(312):
        x = float(tokens.pop(0))
        x_values.append(x)
        y = float(tokens.pop(0))
        y_values.append(y)

    return x_values, y_values
