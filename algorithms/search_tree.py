import random


class SearchAgent:
    def __init__(self, n, k, epsilon=0.1, exclude=False, fixed_size=None):
        self._n = n
        self._k = k
        self._epsilon = epsilon
        self._exclude = exclude
        self._fixed_size = fixed_size
        self._Q = [0]
        self._N = [0]

        root_group = (None, 1)
        # This dict maps the Q index to a tuple of (facility_group, search_value)
        self._index_to_group = {0: root_group}
        # This dict maps the reverse of the above
        self._group_to_index = {root_group: 0}
        # The ID cache prevents duplicate Q entries.
        self._id_cache = {root_group}

    def get_facility_group(self):
        """next_facilities, search_size = search_agent.get_action()"""
        if random.random() < (1 - self._epsilon):
            min_value = min(self._Q)
            index = self._Q.index(min_value)
        else:
            index = random.randint(0, len(self._Q) - 1)

        #print(f"Expanding Node {index} for the {self._N[index] + 1} time")

        group, level = self._index_to_group[index]
        group = list(group) if group is not None else None
        if not self._fixed_size:
            addition = random.randint(1, 5)
            search_size = max(int((self._n / self._k) / level) + addition, 2) if level != 0 else 0
            #search_size = max(int(self._n / (level + 1)), 2) if level != 0 else 0
            #print(search_size)
        else:
            search_size = random.randint(3, 16)
        exclude_original = random.choice([True, False]) if self._exclude is True else False
        #exclude_original = self._exclude
        #print("Search Size:", search_size)
        #print("Exclude next?:", exclude_original)
        return group, level, search_size, exclude_original

    def update_facility_group(self, facility_group, level, new_group, new_distance):
        # remove the root node after it's first been processed
        """
        if facility_group is None:
            self._Q.pop(0)
            self._N.pop(0)
            root = (facility_group, level)
            del self._index_to_group[0]
            del self._group_to_index[root]
        else:
        """
        facility_group = tuple(sorted(facility_group)) if facility_group is not None else None
        # update the facility group that started the search
        action = self._group_to_index[(facility_group, level)]
        self._N[action] += 1
        self._Q[action] += (new_distance - self._Q[action]) / self._N[action]

        # add in the new search candidate
        new_group_entry = (tuple(sorted(new_group)), level + 1)
        if new_group_entry not in self._id_cache:
            self._id_cache.add(new_group_entry)
            self._Q.append(new_distance)
            self._N.append(1)
            new_index = len(self._Q) - 1
            self._index_to_group[new_index] = new_group_entry
            self._group_to_index[new_group_entry] = new_index


