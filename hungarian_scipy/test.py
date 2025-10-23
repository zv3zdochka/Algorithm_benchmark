import numpy as np
from ant_colony import AntColonyAssignment, ACOParams
from ant_colony.utils import brute_force_opt

C = np.array([[0.01, 1, 1, 1],
              [1, 0.02, 1, 1],
              [1, 1, 0.03, 1],
              [1, 1, 1, 0.04]], dtype=float)

aco = AntColonyAssignment(ACOParams(num_ants=20, max_iters=150, beta=4.0, q0=0.3, local_search="2swap", seed=7))
perm, cost, info = aco.solve(C)

p_opt, c_opt = brute_force_opt(C)
print("ACO:", cost, perm)
print("OPT:", c_opt, p_opt)
