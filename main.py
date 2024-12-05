import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pyoptinterface as poi
from pyoptinterface import gurobi, ObjectiveSense

from Find_Cr_Cab_new import Find_Cr_Cab_new
from incidence import incidence
from Find_Cr_Cab import Find_Cr_Cab
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

## 读取风机和OSS坐标数据
# 可根据算例修改的参数
data_Coord = pd.read_excel('HKY_WT.xlsx', header=None)
start_index_WT = 2
stop_index_WT = 43
start_index_OS = 44
stop_index_OS = 44
x_col_WT = 2
y_col_WT = 3

# 不可修改
index_WT = np.arange(start_index_WT - 1, stop_index_WT, 1)
Coord_WT = data_Coord.loc[index_WT, [x_col_WT - 1, y_col_WT - 1]].values
Coord_WT = np.array(Coord_WT)
index_OS = np.arange(start_index_OS - 1, stop_index_OS, 1)
Coord_OS = data_Coord.loc[index_OS, [x_col_WT - 1, y_col_WT - 1]].values
Coord_OS = np.array(Coord_OS)
WTs = stop_index_WT - start_index_WT + 1
OSs = stop_index_OS - start_index_OS + 1

# 绘制风机分布图
G = nx.Graph()
node_num = WTs + OSs
nodes = list(range(node_num))
node_Coord = np.append(Coord_WT, Coord_OS, axis=0)
node_withCoord = dict(zip(nodes, node_Coord))
G.add_nodes_from(nodes)
nx.draw(G, pos=node_withCoord, node_size=2, node_color='b')  # 绘制节点
x_max, y_max = node_Coord.max(axis=0)  # 获取每一列最大值
x_min, y_min = node_Coord.min(axis=0)  # 获取每一列最小值
x_num = (x_max - x_min) / 10
y_num = (y_max - y_min) / 10
plt.xlim(x_min - x_num, x_max + x_num)
plt.ylim(y_min - y_num, y_max + y_num)
plt.show()  # 显示图像1,2并且阻塞程序

# 通过设定节点之间最大距离筛除掉一部分边
I = np.array([[]], 'int')
J = np.array([[]], 'int')
len_l = np.array([])
for i in range(1, WTs, 1):
    for j in range(i):
        dist = np.linalg.norm(Coord_WT[i, :] - Coord_WT[j, :])
        if dist <= 5:
            G.add_edge(i, j)
            I = np.append(I, i)
            J = np.append(J, j)
            len_l = np.append(len_l, dist)

for i in range(WTs):
    for j in range(OSs):
        dist = np.linalg.norm(Coord_WT[i, :] - Coord_OS[j, :])
        if dist <= 5.0292:
            G.add_edge(i, WTs + j)
            I = np.append(I, i)
            J = np.append(J, WTs + j)
            len_l = np.append(len_l, dist)

In = incidence(I, J)
L = len(I)

##绘制所有可能的边
nx.draw(G, pos=node_withCoord, node_size=2, node_color='b')  # 绘制节点
x_max, y_max = node_Coord.max(axis=0)  # 获取每一列最大值
x_min, y_min = node_Coord.min(axis=0)  # 获取每一列最小值
x_num = (x_max - x_min) / 10
y_num = (y_max - y_min) / 10
plt.xlim(x_min - x_num, x_max + x_num)
plt.ylim(y_min - y_num, y_max + y_num)
plt.show()  # 显示图像1,2并且阻塞程序

## 数据汇总
ConsInf = np.array([[range(len(I)), I, J, len_l]]).T
# ConsInf=np.transpose(ConsInf)
Pw = np.ones([WTs, 1]) * 12  # 每台风机额定功率（这里都设置为满发）
Sbase = 1e6
Ubase = 66e3
Ibase = Sbase / Ubase / 1.732
Zbase = Ubase / Ibase / 1.732
LineCap = np.array([[2, 3, 4, 5, 6]]) * 12
C_lines = np.array([[185.4, 229.2, 280.3, 355.5, 482.7]])  # 五种电缆数据
z_b = np.array([[0.246 + 0.135j, 0.1328 + 0.122j, 0.0819 + 0.113j, 0.0491 + 0.105j, 0.03 + 0.094j]], complex)
z = np.tile(ConsInf[:, 3], (1, C_lines.shape[1])) * np.tile(z_b, (I.shape[0], 1)) / Zbase
P_Max = np.ones([len(ConsInf[:, 3]), 1]) * LineCap * 1e6 / Sbase
r = np.real(z)
y_line = 1 / z
minFeeders = 7  # OSS 最小出线数
maxFeeders = 12  # OSS 最大出线数
g_Sub_Max = 600  # OSS 最大承载功率
M = 1e9
Cost = (1.05 * ConsInf[:, 3] + 0.13 * 2) @ C_lines
N = WTs + OSs
n_cab = LineCap.shape[1]
# 创建模型
model = gurobi.Model()
# 决策变量
x = np.empty((L, n_cab), dtype=object)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i, j] = model.add_variable(domain=poi.VariableDomain.Binary)

Pij = np.empty((L, 1), dtype=object)
for i in range(Pij.shape[0]):
    for j in range(Pij.shape[1]):
        Pij[i, j] = model.add_variable(domain=poi.VariableDomain.Continuous)

Pij_all = np.empty((L, n_cab), dtype=object)
for i in range(Pij_all.shape[0]):
    for j in range(Pij_all.shape[1]):
        Pij_all[i, j] = model.add_variable(domain=poi.VariableDomain.Continuous)

Pw_shed = np.empty((WTs, 1), dtype=object)
for i in range(Pw_shed.shape[0]):
    for j in range(Pw_shed.shape[1]):
        Pw_shed[i, j] = model.add_variable(lb=0, domain=poi.VariableDomain.Continuous)

g_Sub_P = np.empty((OSs, 1), dtype=object)
for i in range(g_Sub_P.shape[0]):
    for j in range(g_Sub_P.shape[1]):
        g_Sub_P[i, j] = model.add_variable(lb=0, domain=poi.VariableDomain.Continuous)

# 目标函数
Obj_inv = poi.quicksum(Cost[i][j] * x[i][j] for i in range(Cost.shape[0]) for j in range(Cost.shape[1])) * 1e4
Ur_Hours = 2000 / 8760  # 一般年利用小时数在3000小时左右
Pr_ele = 200  # 入网电价，这里是0.2元/度，一版设为500（0.5元/度）
Obj_loss = Ur_Hours * 8760 * 20 * poi.quicksum(
    r[i][j] * Pij_all[i][j] * Pij_all[i][j] for i in range(r.shape[0]) for j in range(r.shape[1])) * Pr_ele
Obj_WindCurt = M * poi.quicksum(Pw_shed[i][0] for i in range(Pw_shed.shape[0]))
# Obj=Obj_inv+Obj_WindCurt+Obj_loss #含网损项目标函数
Obj = Obj_inv + Obj_WindCurt  # 不含网损项目标函数（设计院）
model.set_objective(Obj)

# 约束条件
# 线缆选型约束
cons_op = model.add_linear_constraint(poi.quicksum(x[i][j] for i in range(x.shape[0]) for j in range(x.shape[1])),
                                      poi.ConstraintSense.Equal, WTs)
for j in range(x.shape[0]):
    cons_op = model.add_linear_constraint(poi.quicksum(x[j][i] for i in range(x.shape[1])),
                                          poi.ConstraintSense.LessEqual, 1.0)
print('****** Cons. on Construction Logic Completed! ** cons num: %d ******' % (cons_op.index + 1))
# 线缆不交叉约束
# Cr_Cab, Q = Find_Cr_Cab_new(In, L, node_num, I, J, Coord_WT, Coord_OS)
# for i in range(Q.shape[0]):
#     Cr1_index = Q[i][1] * np.ones([Q[i][0]], dtype='int')
#     Cr2_index = Q[i, 2:(Q[i][0] + 2)]
#     u = np.array(Q[i][0])
#     cons_cac = model.add_linear_constraint(poi.quicksum(x[Cr1_index[k]][j] + x[Cr2_index[k]][j] for k in range(u) for j in range(x.shape[1])),
#                                            poi.ConstraintSense.LessEqual, u)
Cr_Cab = Find_Cr_Cab(In, L, node_num, I, J, Coord_WT, Coord_OS)
for i in range(Cr_Cab.shape[0]):
    cons_cac = model.add_linear_constraint(poi.quicksum(x[Cr_Cab[i][0], :] + x[Cr_Cab[i][1], :]), poi.ConstraintSense.LessEqual, 1)
print('****** Crossing-avoidance Cons.(CAC) Completed! ** cons num: %d ******' % (cons_cac.index + 1))
# 功率平衡约束
for i in range(In.shape[0] - 1):
    cons_s = model.add_linear_constraint(
        poi.quicksum(In[i][j] * Pij[j][0] for j in range(In.shape[1])) + Pw[i][0] - Pw_shed[i][0],
        poi.ConstraintSense.Equal, 0)
cons_s = model.add_linear_constraint(
    poi.quicksum(In[In.shape[0] - 1][j] * Pij[j][0] for j in range(In.shape[1])) - g_Sub_P[0][0],
    poi.ConstraintSense.Equal,
    0)
for i in range(Pij_all.shape[0]):
    cons_s = model.add_linear_constraint(Pij[i][0] - poi.quicksum(Pij_all[i][j] for j in range(Pij_all.shape[1])),
                                         poi.ConstraintSense.Equal, 0)
for i in range(Pw.shape[0]):
    cons_s = model.add_linear_constraint(Pw[i][0] - Pw_shed[i][0], poi.ConstraintSense.GreaterEqual, 0)
print('******Cons. on Power Balance Completed! ** cons num: %d ******' % (cons_s.index + 1))
# 线缆功率上限约束
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        cons_line = model.add_linear_constraint(Pij_all[i][j] + x[i][j] * LineCap[0][j],
                                                poi.ConstraintSense.GreaterEqual, 0)
        cons_line = model.add_linear_constraint(Pij_all[i][j] - x[i][j] * LineCap[0][j], poi.ConstraintSense.LessEqual,
                                                0)
print('******Cons. on Power Limits of Lines Completed! ** cons num: %d ******' % (
        cons_line.index + 1))
# 升压站功率馈线约束
cons_sub = model.add_linear_constraint(g_Sub_P[0][0], poi.ConstraintSense.LessEqual, g_Sub_Max)
Cab_sub = np.array([[]], 'int')
for i in range(WTs, node_num, 1):
    Cab_sub = np.append(Cab_sub, np.argwhere(ConsInf[:, 2] == i)[:, 0])
cons_sub = model.add_linear_constraint(poi.quicksum(x[i][j] for i in Cab_sub for j in range(x.shape[1])),
                                       poi.ConstraintSense.LessEqual, maxFeeders)
cons_sub = model.add_linear_constraint(poi.quicksum(x[i][j] for i in Cab_sub for j in range(x.shape[1])),
                                       poi.ConstraintSense.GreaterEqual, minFeeders)
print('******Cons. on Power Limits of Subs Completed! ** cons num: %d ******' % (
        cons_sub.index + 1))

# 求解
model.optimize()

##获取求解结果
get_x = np.vectorize(lambda x: model.get_value(x))
x_value = get_x(x)
get_Pij = np.vectorize(lambda Pij: model.get_value(Pij))
Pij_value = get_Pij(Pij)
get_Pij_all = np.vectorize(lambda Pij_all: model.get_value(Pij_all))
Pij_all_value = get_Pij_all(Pij_all)
get_Pw_shed = np.vectorize(lambda Pw_shed: model.get_value(Pw_shed))
Pw_shed_value = get_Pw_shed(Pw_shed)
get_g_Sub_P = np.vectorize(lambda g_Sub_P: model.get_value(g_Sub_P))
g_sub_value = get_g_Sub_P(g_Sub_P)
get_Obj = np.vectorize(lambda Obj: model.get_value(Obj))
Obj_value = get_Obj(Obj)
get_Obj_inv = np.vectorize(lambda Obj_inv: model.get_value(Obj_inv))
Obj_inv_value = get_Obj_inv(Obj_inv)
get_Obj_WindCurt = np.vectorize(lambda Obj_WindCurt: model.get_value(Obj_WindCurt))
Obj_WindCurt_value = get_Obj_WindCurt(Obj_WindCurt)
get_Obj_loss = np.vectorize(lambda Obj_loss: model.get_value(Obj_loss))
Obj_loss_value = get_Obj_loss(Obj_loss)
##展示结果
print('********* 风电场集电系统规划结束！')
print('********* 最优拓扑结构如图所示！')
print('********* 1 线路规划建设成本：￥ %d' % Obj_inv_value)
print('********* 2 弃风成本：￥ %d' % Obj_WindCurt_value)
print('********* 3 集电系统网损成本：￥ %d' % Obj_loss_value)
print('********* 总成本：￥ %d' % Obj_value)

G_result = nx.Graph()
G_result.add_nodes_from(nodes)
edge_index = np.sum(x_value, axis=1)
for i in range(x.shape[0]):
    if edge_index[i] == 1:
        G_result.add_edge(I[i], J[i])
nx.draw(G_result, pos=node_withCoord, node_size=2, node_color='b')  # 绘制节点
x_max, y_max = node_Coord.max(axis=0)  # 获取每一列最大值
x_min, y_min = node_Coord.min(axis=0)  # 获取每一列最小值
x_num = (x_max - x_min) / 10
y_num = (y_max - y_min) / 10
plt.xlim(x_min - x_num, x_max + x_num)
plt.ylim(y_min - y_num, y_max + y_num)
plt.show()  # 显示图像1,2并且阻塞程序
