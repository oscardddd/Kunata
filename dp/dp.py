import numpy as np
'''
1) We add another bandwidth dimension in the dp
2) In the intra-op, our optimization goal includes energy
3) We include checkpointing


'''

'''
1) 如何keep track of the previous "steps" and their execution times

'''



'''
To reduce the
searching complexity for orchestration across heterogeneous
devices, we sort the devices in descending order by their
memory capacity and map stages accordingly
'''

device_matrix = {
    'a':100,
    'b':200,
    'c': 100,
    'd': 500
}

# profiled execution time for each device for each layer
profile_matrix_f = {
}

profile_matrix_b = {
}

layer_weight_matrix = {}

# a fake allocation algorithm 
def batch_allocation(num_devices, B):
    total = sum(v for k, v in device_matrix.items())
    allocation = {}
    for k,v in device_matrix.items():
        allocation[k] = (v/total) * B
    
    return allocation


def profile_f(device_key, b, layer_start, layer_end):
    #Time to perform FP / BP for layer 𝑙 on device 𝑑 with a batch size of 𝛽.
    return profile_matrix_f[device_key][layer]

def profile_b(device_key, b, layer_start, layer_end):
    #Time to perform FP / BP for layer 𝑙 on device 𝑑 with a batch size of 𝛽.
    return profile_matrix_b[device_key][layer]


def layer_weight(layer_id):
    return layer_weight_matrix[layer_id]


def hpp_dynamic_programming(L, N, min_latency_func, microbatch_count):
    
    TOTAL_BDWIDTH = 100
    """
    Dynamic Programming HPP Planning to minimize HPP-Round Latency.

    Parameters:
    L (int): Total number of layers in the DNN model.
    N (int): Total number of edge devices involved.
    min_latency_func (function): Function that computes the latency given l, n, p values.

    Returns:
    np.array: DP table containing the optimal latency values.
    """
    Q = np.full((L + 1, N + 1, min(L, N) + 1, TOTAL_BDWIDTH + 1), np.inf)
    
    # Base case: If no layers are allocated, latency is 0, E_f = 0, E_b = 0
    Q[0, :, :, :] = (0, 0, 0)
     

    for p in range(1, min(L, N) + 1):  # Number of pipeline stages
        for n in range(1, N + 1):  # Number of devices
            for l in range(1, L + 1):  # Number of layers
                for bd in range(10, TOTAL_BDWIDTH): # bandwidth 
                    for n_prime in range(0, n + 1):  # Possible previous number of devices
                        for l_prime in range(0, l + 1):  # Possible previous number of layers
                            for bd_prime in range(10, bd+10):
                                # calculate inside and outside bandwidth
                                inside_bd, outside_bd = separate_bd(l_prime)
                                
                                '''
                                    We need to know: 
                                    the latency of a new single stage (execution step) with layers l -l' over remaining n -n' devices using data parallelism.
                                '''
                                allocation = batch_allocation(n_prime, microbatch_count)
                                # get the forward latency and backward latency of this stage
                                E_forward = float('-inf')
                                for device_key in allocation.keys():
                                    E_forward_execution = max(E_forward, sum([profile_f(device_key, allocation[device_key], l, l_prime)] for l in range(l_prime)))
                                
                                E_backward = float('-inf')
                                for device_key in allocation.keys():
                                    E_backward_execution = max(E_backward, sum([profile_b(device_key, allocation[device_key], l, l_prime)] for l in range(l_prime)))
                                
                                execution_cost = E_forward_execution + E_backward_execution
                                
                                # All reduce 的时间只取决于 bd_prime 和 l
                                # 
                                avg_bd = (n_prime*(n_prime -1))/2
                                # 这个可以化简
                                all_reduce_time = (2*(n_prime - 1) *sum([layer_weight(l) for l in range(l_prime)])) / (n_prime * avg_bd)
                                
                                
                                
                                communication_cost = all_reduce_time + latency_comunication_with_last_stage_
                                
                                print(f"number of stage: {p};", f"devices: {n_prime};", f"taking layers: {l_prime}", )
                                
                                Q_prime = Q[l_prime, n_prime, p-1, bd -bd_prime] + execution_cost + communication_cost
                                Q[l, n, p, bd] = min(Q[l, n, p, bd], Q_prime)
                                
                                
                                
        
    return Q


def test():
    
    for p in range(1, min(L, N) + 1):  # Number of pipeline stages
        for l in range(L,-1,-1):  
            for n in range(1, N + 1):  # Number of devices



def min_latency_func(l, n, l_prime, n_prime, p):
    """
    Computes the latency based on Eq. (10) in the paper.
    
    Parameters:
    l (int): Current layer count.
    n (int): Current device count.
    l_prime (int): Previous layer count.
    n_prime (int): Previous device count.
    p (int): Pipeline stage count.

    Returns:
    float: Computed latency.
    """
    if p == 1:
        return np.random.uniform(1, 10)  # Placeholder latency value
    return np.random.uniform(1, 10) + np.random.uniform(1, 10)  # Simulating latency

# Example usage
L = 2  # Number of layers
N = 3   # Number of devices
dp_result = hpp_dynamic_programming(L, N, min_latency_func)

# Display the DP table for the minimum latency
print(dp_result)
