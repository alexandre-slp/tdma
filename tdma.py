# TriDiagonal Matrix Algorithm (TDMA) or Thomas Algorithm
# Caso particular do algoritmo de eleiminacao Gaussiana
# Algoritmo para resolver sistema de equacoes de primeira ordem
# [ B.  C.  0.  0.  0.]
# [ A.  B.  C.  0.  0.]
# [ 0.  A.  B.  C.  0.]
# [ 0.  0.  A.  B.  C.]
# [ 0.  0.  0.  A.  B.]


import numpy as np
# import matplotlib.pyplot as plt


def tdma_solver(tdm):
    solved_tdm = np.zeros((5, 5))
    return solved_tdm

tdm = np.zeros((5, 5))


def load_vector(tdm):
    A = []
    B = []
    C = []

    for coluna in range(0, len(tdm)):
        linha = coluna
        B.append(tdm[linha, coluna])
        if coluna == 0:
            A.append(tdm[linha + 1, coluna])
            C.append(0)
        elif coluna == (len(tdm) - 1):
            A.append(0)
            C.append(tdm[linha - 1, coluna])
        elif coluna != 0 and coluna != (len(tdm) - 1):
            A.append(tdm[linha + 1, coluna])
            C.append(tdm[linha - 1, coluna])

    return A, B, C
