import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def euler(f, punto_inicial, hasta, n, precision=5):

    h=(hasta-punto_inicial[0])/n
    x = np.zeros(n+1)
    y = np.zeros(n+1)

    x[0] = punto_inicial[0]
    y[0] = punto_inicial[1]

    for i in range(1, n+1):
        x[i] = x[i-1] + h
        y[i] = y[i-1] + h * f(x[i-1], y[i-1])

    data = {'x': x, 'y': y}
    df = pd.DataFrame(data)
    print(df)
    graficar(x, y)

def graficar(x, y):
    plt.plot(x, y, label='Aproximación de Euler')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Método de Euler')
    plt.show()