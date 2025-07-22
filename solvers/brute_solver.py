import itertools
import random

import torch


def get_optimal_facilities(graph, n, k):
    """
    Brute force search to get the optimal facilities in a graph G
    """
    best_distance = 10000000000
    best_facilities = None
    for facilities in itertools.combinations(list(range(n)), k):
        distance = calculate_distance(graph, facilities, n)

        if distance < best_distance:
            best_distance = distance
            best_facilities = facilities

    return best_facilities

def get_optimal_local_search_swap(graph, facilities, n):
    best_distance = calculate_distance(graph, facilities, n)
    best_pair = None

    for suggested_facility in range(n):
        if suggested_facility in facilities:
            continue

        for i in range(len(facilities)):
            old_facility = facilities.pop(0)
            facilities.append(suggested_facility)
            new_distance = calculate_distance(graph, facilities, n)
            if new_distance < best_distance:
                best_distance = new_distance
                best_pair = {"client": suggested_facility, "facility": old_facility}

            facilities.remove(suggested_facility)
            facilities.append(old_facility)

    return best_pair

def get_good_enough_local_search_swap(graph, facilities, n, max_check, epsilon=0.1):
    original_distance = calculate_distance(graph, facilities, n)
    best_distance = original_distance
    best_pair = None

    for _ in range(max_check):
        potential_facility = random.choice(facilities)
        potential_client = random.randint(0, n - 1)
        # make sure we don't select a facility
        while potential_client in facilities:
            potential_client = random.randint(0, n - 1)

        facilities.remove(potential_facility)
        facilities.append(potential_client)
        new_distance = calculate_distance(graph, facilities, n)

        if new_distance < best_distance:
            best_distance = new_distance
            best_pair = {"client": potential_client, "facility": potential_facility}

        facilities.remove(potential_client)
        facilities.append(potential_facility)

        # early exit condition, we get a change better than epsilon
        percent_change = abs((best_distance - original_distance) / original_distance)
        if percent_change >= epsilon:
            break

    # for right now we just return two facilities
    if best_distance >= original_distance:
        print("fail")
        candidates = random.sample(facilities, k=2)
        best_pair = {"client": candidates[0], "facility": candidates[1]}

    return best_pair


def calculate_distance(graph, facilities, n=None):
    """
    n is depreciated. Remove later.
    """
    # Create a subgraph where each row is an active facility and the columns are the distances to the facilities.
    distance_sub_graph = graph._distances[facilities]

    # get the min distance for each column
    min_values, _ = torch.min(distance_sub_graph, dim=0)

    # sum up the distances to get the total distance
    return float(torch.sum(min_values))



def get_facilities(h, n, k):
    facilities = set()
    for j in range(k):
        for s in range(n):
            index = (n * k) + s + (j * n)
            if h.V[index] == 1:
                facilities.add(s)
    return list(facilities)
