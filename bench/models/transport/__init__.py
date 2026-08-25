"""`transport`, in every dialect that has one."""

from bench.models.transport import gurobipy_loop, gurobipy_matrix

FORMULATIONS = {'gurobipy-loop': gurobipy_loop, 'gurobipy-matrix': gurobipy_matrix}
