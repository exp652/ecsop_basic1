import numpy as np
import pyoptinterface as poi
from pyoptinterface import gurobi

model = gurobi.Model()

N = 8

x = np.empty((N, N), dtype=object)
for i in range(N):
    for j in range(N):
        x[i, j] = model.add_variable(domain=poi.VariableDomain.Binary)

for i in range(N):
    # 行列约束
    model.add_linear_constraint(poi.quicksum(x[i, :]), poi.Eq, 1.0)
    model.add_linear_constraint(poi.quicksum(x[:, i]), poi.Eq, 1.0)
for i in range(-N + 1, N):
    # 对角线约束
    model.add_linear_constraint(poi.quicksum(x.diagonal(i)), poi.Leq, 1.0)
model.add_linear_constraint(poi.quicksum(np.fliplr(x).diagonal(i)), poi.Leq, 1.0)

model.optimize()

get_v = np.vectorize(lambda x: model.get_value(x))
x_value = get_v(x)

print(x_value.astype(int))