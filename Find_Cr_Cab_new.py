import numpy as np


def Find_Cr_Cab_new(In, L, node_num, s, t, Coord_WT, Coord_OS):
    Coord = np.array(Coord_WT)
    Coord = np.append(Coord, Coord_OS, axis=0)
    V_l = np.zeros([L, 2])  # 两节点的向量
    for i in range(L):
        V_l[i, :] = Coord[s[i], :] - Coord[t[i], :]
    Cr_Cab = np.array([[0, 0]])
    for i in range(node_num):
        for j in range(L - 1):
            for k in range(j + 1, L, 1):
                if s[j] == i or t[j] == i or s[k] == i or t[k] == i:
                    AC = np.array(Coord[s[k], :] - Coord[s[j], :])
                    AB = np.array(Coord[t[j], :] - Coord[s[j], :])
                    AD = np.array(Coord[t[k], :] - Coord[s[j], :])
                    CB = np.array(Coord[t[j], :] - Coord[s[k], :])
                    CD = np.array(Coord[t[k], :] - Coord[s[k], :])
                    if np.cross(AC, AB) * np.cross(AD, AB) < -1e-6 and np.cross(-AC, CD) * np.cross(CB, CD) < -1e-6:
                        Cr_Cab = np.append(Cr_Cab, np.array([[j, k]]), axis=0)
    Cr_Cab = np.unique(Cr_Cab[1:, :], axis=0)
    Cr_Cab_new = Cr_Cab
    row_index = np.array([Cr_Cab[:, 0]])
    col_index = np.array([Cr_Cab[:, 1]])
    Cab_Cr1 = np.unique(row_index)
    Num_Cr = np.array([[]], dtype='int')

    for i in range(len(Cab_Cr1)):
        Cr2_where = np.argwhere(row_index == Cab_Cr1[i])
        Num_Cr = np.append(Num_Cr, Cr2_where.shape[0])
    max_cols = 2 + int(max(Num_Cr))

    Q = np.zeros([len(Cab_Cr1), max_cols], dtype='int')
    for i in range(len(Cab_Cr1)):
        Cr2_where = np.argwhere(row_index == Cab_Cr1[i])
        Q_row_part1 = np.append(Cab_Cr1[i], col_index[0, Cr2_where[:, 1]])
        Q[i][0:int(Num_Cr[i]) + 2] = np.append(Num_Cr[i], Q_row_part1)

    return Cr_Cab, Q
