import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def runge_kutta(f, punto_inicial, hasta, n, precision=5):

    h=(hasta-punto_inicial[0])/n
    x = np.zeros(n+1)
    y = np.zeros(n+1)

    x[0] = punto_inicial[0]
    y[0] = punto_inicial[1]

    tabla = []
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + k1*h/2)
        k3 = f(x[i] + h/2, y[i] + k2*h/2)
        k4 = f(x[i] + h, y[i] + k3*h)

        x[i+1] = x[i] + h
        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6

        tabla.append([x[i], y[i], k1, k2, k3, k4, x[i+1], y[i+1]])
        df_summary = pd.DataFrame(tabla, columns=['x_n', 'y_n', 'k1', 'k2', 'k3', 'k4', 'x_n+1', 'y_n+1'])

    print(df_summary)
    graficar(x,y)

def graficar(x, y):
    plt.plot(x, y, 'o-', label='Runge Kutta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Método de Runge Kutta')
    plt.show()