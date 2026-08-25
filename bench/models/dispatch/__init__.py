"""`dispatch`, in every dialect that has one."""

from bench.models.dispatch import gurobipy_loop, gurobipy_matrix

FORMULATIONS = {'gurobipy-loop': gurobipy_loop, 'gurobipy-matrix': gurobipy_matrix}
