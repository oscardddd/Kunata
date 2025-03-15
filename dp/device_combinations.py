import itertools



devices_example = ['a', 'b', 'c', 'd']

def device_combinations(devices):
    combinations_dict = {}

    for r in range(1, len(devices) + 1):
        combinations_dict[r] = list(itertools.combinations(devices, r))

    return combinations_dict
