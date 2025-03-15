import numpy as np
from itertools import combinations
from device_combinations import device_combinations
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, LpContinuous, value, PULP_CBC_CMD
import time

'''
TODO: Add support for steps not stage, and calculate the corresponding critical path
'''



'''
To reduce the
searching complexity for orchestration across heterogeneous
devices, we sort the devices in descending order by their
memory capacity and map stages accordingly
'''





# profiled execution time for each device for each layer
profile_matrix_f = {
}

profile_matrix_b = {
}


# a fake allocation algorithm 
def batch_allocation(num_devices, B):
    total = sum(v for k, v in device_matrix.items())
    allocation = {}
    for k,v in device_matrix.items():
        allocation[k] = (v/total) * B
    
    return allocation




class algorithm():
    def __init__(self, device_matrix, layer_intermediate_size, layer_weight, total_bdwidth, energy_param):
        self.device_matrix = device_matrix
        self.layer_intermediate_size = layer_intermediate_size
        self.layer_weight = layer_weight
        self.profiled_result = {}
        self.microbatch_count = 8
        self.L = len(layer_weight.keys())
        self.N = len(device_matrix.keys())
        self.d_combinations = device_combinations(device_matrix.keys())
        self.l_combinations= device_combinations(layer_weight.keys())
        self.total_bdwidth = total_bdwidth
        self.energy_param = energy_param
        # The matrix to keep track of optimal configurations and the corresponding latencies
        self.Q = {}
    
    
    ## This is used to compute the optimal microbatch allocation with a device group


    def updated_allocation(self, l_start, l_end, gamma=1e-3):
        alpha = self.energy_param
        devices = list(self.device_matrix.keys())
        problem = LpProblem("OptimalLatencyEnergyAllocation", LpMinimize)
        num_devices = len(devices)

        # Decision Variables
        Y = {d: LpVariable(f"Y_{d}", lowBound=1, cat=LpInteger) for d in devices}
        Y_max = LpVariable("Y_max", lowBound=1, cat=LpInteger)

        M = self.microbatch_count
        bandwidth = self.total_bdwidth

        # workload计算
        workload = sum(self.layer_weight[i] for i in range(l_start, l_end+1))
        if workload <= 0:
            raise ValueError("Workload must be positive!")

        # device-specific parameters (需要提供alpha_d)
        alpha = {d: self.energy_param[d] for d in devices}
        base_forward = {d: self.device_matrix[d] / workload for d in devices}
        base_backward = {d: self.device_matrix[d] * 2 / workload for d in devices}

        # 约束条件 (microbatch数量和Y_max定义)
        problem += lpSum(Y[d] for d in devices) == M, "MicrobatchCount"
        for d in devices:
            problem += Y_max >= Y[d], f"Ymax_{d}"
        problem += Y_max <= M, "YmaxBound"

        # forward和backward的max latency（加能耗后）
        E_f_max = LpVariable("E_f_max", lowBound=0)
        E_b_max = LpVariable("E_b_max", lowBound=0)

        for d in devices:
            # forward latency + energy penalty
            forward_latency_with_energy = base_forward[d] * Y[d] + gamma * alpha[d] * base_forward[d] * Y[d]
            problem += E_f_max >= forward_latency_with_energy, f"ForwardLatencyWithEnergy_{d}"

            # backward latency + energy penalty
            backward_latency_with_energy = base_backward[d] * Y[d] + gamma * alpha[d] * base_backward[d] * Y[d]
            problem += E_b_max >= backward_latency_with_energy, f"BackwardLatencyWithEnergy_{d}"

        # 通信时延 (不加能耗)
        sync_weight_size = self.layer_intermediate_size[l_start]
        T_comm = (M - Y_max) * sync_weight_size / bandwidth

        # 新的优化目标
        problem += E_f_max + E_b_max + T_comm, "MaxLatencyWithEnergy"

        # 求解
        problem.solve()

        # 显示求解结果
        if problem.status != 1:
            print(f"Problem not solved optimally. Status: {problem.status}")
            return

        print("Optimal Microbatch Allocation with Energy Consideration:")
        for d in devices:
            allocated = int(Y[d].varValue) if Y[d].varValue else 0
            print(f"{d}: {allocated} micro-batches")

        total_latency_with_energy = value(problem.objective)
        print(f"Total Latency (with energy penalty): {total_latency_with_energy:.4f} sec")



    def allocation(self, device_group, l_start, l_end):
        devices = list(device_group)
        problem = LpProblem("OptimalMicrobatchAllocation", LpMinimize)
        num_devices = len(devices)

        # 变量定义
        Y = {d: LpVariable(f"Y_{d}", lowBound=1, cat=LpInteger) for d in devices}
        Y_max = LpVariable("Y_max", lowBound=1, cat=LpInteger)
        E_f_max = LpVariable("E_f_max", lowBound=0)
        E_b_max = LpVariable("E_b_max", lowBound=0)

        M = self.microbatch_count
        bandwidth = self.total_bdwidth

        # 计算workload，确保不是0
        workload = sum(self.layer_weight[i] for i in range(l_start, l_end + 1))
        if workload <= 0:
            raise ValueError("Workload must be positive!")

        # Forward/backward base times (per microbatch)
        base_forward = {d: workload / self.device_matrix[d] for d in devices}
        base_backward = {d:  workload / self.device_matrix[d] * 2 for d in devices}

        problem += lpSum(Y[d] for d in devices) == M, "MicrobatchCount"

        # E_f_max 和 E_b_max的约束
        for d in devices:
            problem += E_f_max >= base_forward[d] * Y[d], f"ForwardTime_{d}"
            problem += E_b_max >= base_backward[d] * Y[d], f"BackwardTime_{d}"
            problem += Y_max >= Y[d], f"Ymax_{d}"

        # 明确约束Y_max
        problem += Y_max <= M, "YmaxBound"

        # 通信数据大小
        sync_weight_size = self.layer_intermediate_size[l_end]

        T_comm = (M - Y_max) * sync_weight_size / bandwidth

        # 目标函数
        problem += E_f_max + E_b_max + T_comm, "TotalLatency"

        # 求解问题
        problem.solve(PULP_CBC_CMD(msg=False))

        if problem.status != 1:
            print(f"Problem not solved optimally. Status: {problem.status}")
            return
        
        allocations = {}
        for d in devices:
            allocations[d] = int(Y[d].varValue)
            # print(f"{d}: {int(Y[d].varValue)} micro-batches")
        return value(problem.objective), allocations
        # print("Optimal Microbatch Allocation:")
    

        # print(f"Total Latency: {value(problem.objective):.4f} sec")

    
    def generate_combinations(self, devices):
        all_combinations = []
        n = len(devices)
        for r in range(1, n + 1):  # Generate combinations of all lengths from 1 to n
            all_combinations.extend(combinations(devices, r))
        return all_combinations
    
        
    def profiler(self):
        print(self.d_combinations)
        start_time = time.time()
        for k in self.d_combinations.keys():
            for device_group in self.d_combinations[k]:
                for l_start in range(0, self.L):
                    for l_end in range(l_start, self.L):
                        for bd in range(1, self.total_bdwidth + 1):
                            print(device_group, l_start, l_end, bd)
                            latency, allocation  = self.allocation(device_group, l_start, l_end)
                            self.profiled_result[frozenset(device_group)][(l_start,l_end, bd)] = [latency, allocation]
        print(f"spend {start_time-time.time()}s")
                           
    def hpp_dynamic_programming(self):
        
        TOTAL_BDWIDTH = self.total_bdwidth
        """
        Dynamic Programming HPP Planning to minimize HPP-Round Latency.

        Parameters:
        L (int): Total number of layers in the DNN model.
        N (int): Total number of edge devices involved.
        min_latency_func (function): Function that computes the latency given l, n, p values.

        Returns:
        np.array: DP table containing the optimal latency values.
        """
        for key, comb in self.d_combinations.items():
            for device_group in comb:
                self.Q[frozenset(device_group)] = {}
            # set up the base case
            for l_start in range(1, self.L + 1):
                for l_end in range(l_start, self.L + 1):
                    for bd in range(1, self.total_bdwidth + 1):
                        # 1 here because stage=1 is the base case
                        # TODO: Change this to a name tuple
                        a = self.profiled_result[(frozenset(device_group), l_start, l_end, bd)]
                        self.Q[frozenset(device_group)][(1, l_start, l_end, bd)] = [a[0], a[1]]

    
            
        for n in range(1, self.N + 1):  # Number of devices
            for p in range(1, min(self.L, self.N) + 1):  # Number of pipeline stages
                for device_comb in self.d_combinations[n]: # possible combinations of devicecs with size n
                    for l in range(1, self.L + 1):  # Number of layers
                        for bd in range(1, TOTAL_BDWIDTH): # bandwidth
                            updated_Q = self.intra_op(p, l, bd, device_comb)
                self.Q[frozenset(device_comb)][(p, l, bd)] = updated_Q
        
        
        return self.Q[(self.d_combinations[max(self.d_combinations.keys())])]

    def intra_op(self, p, l, bd, device_comb):
        
        # all possible combinations of devices out of device_comb
        inside_d_combinations = self.generate_combinations(device_comb)
        n = len(device_comb)
        print(inside_d_combinations)
        min_Q = float('inf')
        for n_prime in range(1, n):  # Possible previous number of devices
            for cur_combination in inside_d_combinations[n_prime]:
                left_devices = [device for device in device_comb if device not in cur_combination]
                for l_prime in range(0, l + 1):  # Possible previous number of layers
                    for bd_prime in range(10, bd+10):
                        bd_pta = 0.9*(bd-bd_prime)
                        
                        # 用来两个stage之间通信的bandwidth
                        bd_ptb = 0.1*(bd-bd_prime)
                        
                        latency_ptb, configuration_b =  self.profiled_result[(cur_combination, l_prime, l, bd_pta)]
                        
                        # data size between l_prime and l_prime+1
                        intermediate_size = self.layer_intermediate_size[l_prime]
                        trasnmission_time = intermediate_size/bd_ptb
                        mininum_prev = float('inf')
                        for stage in range(p):
                            if self.Q[frozenset(left_devices)][(stage, 0, l_prime, bd_pta)][0] < mininum_prev:
                                mininum_prev = self.Q[frozenset(left_devices)][(stage, 0, l_prime, bd_prime)][0]
                                mininum_prev_conf = self.Q[frozenset(left_devices)][(stage, 0, l_prime, bd_prime)][1]
                                    
                        total_latency = mininum_prev + trasnmission_time + latency_ptb
                        
        cur_value = self.Q(frozenset(left_devices), (stage, 0, l, bd_prime))
        if cur_value[0] > total_latency:
            self.Q(frozenset(left_devices), (stage, 0, l, bd_prime)) = [total_latency, configuration_b]
        
        
        
                        # 用来两个stage之间通信的bandwidth 
                        # get the forward latency and backward latency of this stage
                        # E_forward = float('-inf')
                        # for device_key in allocation.keys():
                        #     E_forward_execution = max(E_forward, sum([profile_f(device_key, allocation[device_key], l, l_prime)] for l in range(l_prime)))
                        
                        # E_backward = float('-inf')
                        # for device_key in allocation.keys():
                        #     E_backward_execution = max(E_backward, sum([profile_b(device_key, allocation[device_key], l, l_prime)] for l in range(l_prime)))
                        
                        # execution_cost = E_forward_execution + E_backward_execution
                        
                        # print(f"number of stage: {p};", f"devices: {n_prime};", f"taking layers: {l_prime}", )
                        # cur_Q = execution_cost + communication_cost
                        # min_Q = min(cur_Q, min_Q)
        return min_Q 
                    


if __name__ == "__main__":

    device_matrix = {
        'a':100,
        'b':200,
        'c': 100,
        'd': 500
    }

    layer_intermediate_size = {
        0:20,
        1:30,
        2:10,
        3:20,
        4:20,
        5:20,
        6:20
    }

    layer_weight = {
        0:10,
        1:20,
        2:10,
        3:50,
        4:20,
        5:100,
        6: 1
    }
    
    energy_param = {
        'a': 0.2,
        'b': 0.3,
        'c': 0.01,
        'd': 0.3,
        
    }
    
    algo = algorithm(device_matrix, layer_intermediate_size, layer_weight, 400, energy_param)
    # algo.updated_allocation(0, 5)
    algo.profiler()