import numpy as np


def incidence(s, t):
    maxNode = max(max(s), max(t)) + 1
    I = np.zeros([maxNode, len(s)])
    for j in range(len(s)):
        I[s[j], j] = 1
        I[t[j], j] = -1
    return I
