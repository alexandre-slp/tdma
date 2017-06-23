# TriDiagonal Matrix Algorithm (TDMA) or Thomas Algorithm.
# Caso particular do algoritmo de eleiminacao Gaussiana.
# Algoritmo para resolver sistema de equacoes de primeira ordem
# que resultam em matrizes tridiagonais.
#
# Exemplo para um sistema de 5 equacoes e 5 incognitas(xN):
#
# Sistema de equacoes:
#
# B(x1) + C(x2) + 0(x3) + 0(x4) + 0(x5) = d1
# A(x1) + B(x2) + C(x3) + 0(x4) + 0(x5) = d2
# 0(x1) + A(x2) + B(x3) + C(x4) + 0(x5) = d3
# 0(x1) + 0(x2) + A(x3) + B(x4) + C(x5) = d4
# 0(x1) + 0(x2) + 0(x3) + A(x4) + B(x5) = d5
#
# Sistema na forma matricial: A . X = D
#
# [ B.  C.  0.  0.  0.]   [x1]    [d1]
# [ A.  B.  C.  0.  0.]   [x2]    [d2]
# [ 0.  A.  B.  C.  0.] x [x3] =  [d3]
# [ 0.  0.  A.  B.  C.]   [x4]    [d4]
# [ 0.  0.  0.  A.  B.]   [x5]    [d5]
#

import numpy as np
import matplotlib.pyplot as plt

BI_COLOR = ['#4286F4', '#6BC924']


def load_abc_vectors(tdm):
    'Retorna os vetores a, b, c da matriz tridiagonal'
    a = []                                              # inicial o vetor a
    b = []                                              # inicial o vetor b
    c = []                                              # inicial o vetor c
    for coluna in range(0, len(tdm)):                   # carrega os vetores a, b, c percorrendo a matriz
        linha = coluna                                  # percore pela diagonal principal
        b.append(tdm[linha, coluna])                    # carrega o vetor b
        if coluna == 0:                                 # primeiro elto
            a.append(0)                                 # primeiro elto do vetor a eh 0
            a.append(tdm[linha + 1, coluna])            # pega o elto abaixo da diagonal principal
        if coluna == (len(tdm) - 1):                    # ultimo elto
            c.append(tdm[linha - 1, coluna])            # pega o elto acima da diagonal principal
            c.append(0)                                 # ultimo elto do vetor c eh 0
        if coluna != 0 and coluna != (len(tdm) - 1):    # demais eltos
            a.append(tdm[linha + 1, coluna])            # pega elto abaixo da diagonal principal
            c.append(tdm[linha - 1, coluna])            # pega o elto acima da diagonal principal
    return a, b, c


def load_cl(a, b, c):
    'Retorna o vetor c linha a partir dos vetores a, b, c'
    cl = [0] * (len(c) - 1)                           # inicializa o vetor c linha
    for i in range(0, len(cl)):                       # percorre c linha
        if i == 0:                                    # primeiro elemento
            cl[i] = float(c[i])/b[i]                  # converte para float e calcula o primeiro c linha
        else:                                         # demais eltos
            cl[i] = c[i]/(b[i] - (a[i] * cl[i - 1]))  # calcula os demais eltos
    return cl


def load_dl(a, b, cl, d):
    'Retorna o vetor d linha a partir dos vetores a, b, cl, d'
    dl = [0] * len(d)                                                        # inicializa o vetor d linha
    for i in range(0, len(dl)):                                              # percorre d linha
        if i == 0:                                                           # primeiro elemento
            dl[i] = float(d[i])/b[i]                                         # converte para float e calcula
                                                                             # o primeiro d linha
        else:                                                                # demais eltos
            dl[i] = (d[i] - (a[i] * dl[i - 1]))/(b[i] - (a[i] * cl[i - 1]))  # calcula os demais eltos
    return dl


def tdma_solver(tdm, d):
    'Retorna o vetor x com as incognitas'
    x = [0] * len(d)                                     # inicialixa o vetor x
    a, b, c = load_abc_vectors(tdm)                      # pega os valores de a, b, c, da matriz tdm de entrada
    cl = load_cl(a, b, c)                                # pega o vetor c linha calculado
    dl = load_dl(a, b, cl, d)                            # pega o vetor d linha calculado
    for i in range(len(x) - 1, -1, -1):                  # percorre o vetor x de tras para frente
        if i == len(x) - 1:                              # ultimo elto
            x[i] = dl[i]                                 # carrega o ultimo elto
        else:                                            # demais eltos
            x[i] = dl[i] - (cl[i] * x[i + 1])            # carrega demais eltos
    return x


def label_gen(x):
    'Retorna os labels do eixo X intercalados verticalmente'
    labels = []
    for i in range(1, len(x) + 1):
        if i % 2 == 0:
            labels.append('|\nX' + str(i))
        else:
            labels.append('X' + str(i))
    return labels


# Dados do trabalho:
tdm_input = np.matrix(np.zeros((100, 100)))
for coluna in range(0, len(tdm_input)):
    linha = coluna
    tdm_input[linha, coluna] = coluna + 1 + (coluna + 1) * 0.001          # diagonal principal
    if coluna == 0:                                                       # primeiro elto
        tdm_input[linha + 1, coluna] = coluna + 2 + (coluna + 1) * 0.001  # elto abaixo da diagonal principal
    if coluna == (len(tdm_input) - 1):                                    # ultimo elto
        tdm_input[linha - 1, coluna] = coluna + 0 + (coluna + 1) * 0.001  # elto acima da diagonal principal
    if coluna != 0 and coluna != (len(tdm_input) - 1):                    # demais eltos
        tdm_input[linha + 1, coluna] = coluna + 2 + (coluna + 1) * 0.001  # elto abaixo da diagonal principal
        tdm_input[linha - 1, coluna] = coluna + 0 + (coluna + 1) * 0.001  # elto acima da diagonal principal

vec_d = np.matrix(np.arange(1, len(tdm_input) + 1).reshape(len(tdm_input), 1))
vec_x = tdma_solver(tdm_input, vec_d)

# Plot:
plt.style.use('ggplot')

n_bars = len(vec_x)
x_loc = np.arange(n_bars)
bar_width = len(x_loc)/n_bars

fig, ax = plt.subplots(figsize=(20, 10))
bars_rects = ax.bar(x_loc, vec_x, bar_width, color=BI_COLOR)
x_labels = label_gen(vec_x)

y_min = min(vec_x) * 1.3
y_max = max(vec_x) * 1.3
max_abs = max(abs(y_min), abs(y_max))

ax.set_ylim((y_min, y_max))
ax.set_title('Solucao do sistema', fontsize=30)
ax.set_ylabel('Valores', fontsize=20)
ax.set_xticks(x_loc)
ax.set_xticklabels(x_labels, fontsize=8)

plt.axhline(color='k')
# plt.show()
plt.savefig('vec_x.png')

