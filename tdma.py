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
    cl = [0] * (len(c) - 1)                         # inicializa o vetor c linha
    for i in range(0, len(cl)):                     # percorre c linha
        if i == 0:                                  # primeiro elemento
            cl[i] = float(c[i])/b[i]                # converte para float e calcula o primeiro c linha
        else:                                       # demais eltos
            cl[i] = c[i]/(b[i] - a[i] * cl[i - 1])  # calcula os demais eltos
    return cl


def load_dl(a, b, cl, d):
    'Retorna o vetor d linha a partir dos vetores a, b, cl, d'
    dl = [0] * len(d)                                                       # inicializa o vetor d linha
    for i in range(0, len(dl)):                                             # percorre d linha
        if i == 0:                                                          # primeiro elemento
            dl[i] = float(d[i])/b[i]                                        # converte para float e calcula
                                                                            # o primeiro d linha
        else:                                                               # demais eltos
            dl[i] = (d[i] - a[i] * dl[i - 1])/(b[i] - a[i] * cl[i - 1])     # calcula os demais eltos
    return dl


def tdma_solver(tdm, d):
    'Retorna o vetor x com as incognitas'
    x = [0] * len(d)                                    # inicialixa o vetor x
    a, b, c = load_abc_vectors(tdm)                     # pega os valores de a, b, c, da matriz tdm de entrada
    cl = load_cl(a, b, c)                               # pega o vetor c linha calculado
    dl = load_dl(a, b, cl, d)                           # pega o vetor d linha calculado
    for i in range(len(x) - 1, -1, -1):                 # percorre o vetor x de tras para frente
        if i == len(x) - 1:                             # ultimo elto
            x[i] = round(dl[i], 4)                      # carrega o ultimo elto
        else:                                           # demais eltos
            x[i] = round(dl[i] - cl[i] * x[i + 1], 4)   # carrega demais eltos
    return x


def label_gen(x):
    labels = []
    for i in range(1, len(x) + 1):
        labels.append('X' + str(i))
    return labels


def auto_label(bars, max):
    """
    Attach a text label above each bar displaying its height
    """
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01 * max,
                    '{:.4f}'.format(height),
                    ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height - 0.045 * max,
                    '{:.4f}'.format(height),
                    ha='center', va='bottom')

tdm_input = np.array([[2, 1, 0, 0, 0],
                      [1, 2, 1, 0, 0],
                      [0, 1, 2, 1, 0],
                      [0, 0, 1, 2, 1],
                      [0, 0, 0, 1, 2]])

vec_d = [4, 4, 0, 0, 2]

vec_x = tdma_solver(tdm_input, vec_d)

plt.style.use('ggplot')

n_bars = len(vec_x)
x_loc = np.arange(n_bars)
bar_width = 5/n_bars

fig, ax = plt.subplots()
bars_rects = ax.bar(x_loc, vec_x, bar_width, color=BI_COLOR)
x_labels = label_gen(vec_x)

y_min = min(vec_x) * 1.5
y_max = max(vec_x) * 1.5
max_abs = max(abs(y_min), abs(y_max))

ax.set_ylim((y_min, y_max))
ax.set_title('Solucao do sistema')
ax.set_ylabel('Valores')
ax.set_xticks(x_loc)
ax.set_xticklabels(x_labels)
auto_label(bars_rects, max_abs)

plt.axhline(color='k')
plt.show()

