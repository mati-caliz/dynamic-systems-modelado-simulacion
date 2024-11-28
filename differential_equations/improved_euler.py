import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def euler_mejorado(f, solucion_analitica, punto_inicial, hasta, n, precision=5):

    h=(hasta-punto_inicial[0])/n
    x = np.zeros(n+1)
    y = np.zeros(n+1)

    x[0] = punto_inicial[0]
    y[0] = punto_inicial[1]

    error = np.zeros(n+1)
    yreal = np.zeros(n+1)
    yreal[0] = solucion_analitica(x[0])
    error[0] = abs(y[0] - yreal[0])
    tabla = []
    tabla.append([0, x[0], "-", y[0], yreal[0], error[0]])

    for i in range(n):
        y_pred = y[i] + h * f(x[i], y[i])  # Predicción con Euler simple
        y[i+1] = y[i] + (h / 2) * (f(x[i], y[i]) + f(x[i] + h, y_pred))  # Corrección
        x[i+1] = x[i] + h
        yreal[i+1] = solucion_analitica(x[i+1])
        error[i+1] = abs(y[i+1] - yreal[i+1])  # Calcular error respecto a la solución exacta

        # Agregar fila a la tabla con x, y_pred, y_n, y_real, error
        tabla.append([i+1, x[i+1], y_pred, y[i+1], yreal[i+1], error[i+1]])
    
    # Imprimir la tabla usando tabulate
    headers = ["Iteración", "x", "y* (Predicción)", "y_n (Euler Mejorado)", "y_real (Exacta)", "Error"]
    print(tabulate(tabla, headers=headers, floatfmt=".6f"))
    graficar(x,y,solucion_analitica)

def graficar(x, y,solucion_analitica):
    plt.plot(x, y, 'o-', label='Euler Mejorado')
    plt.plot(x, solucion_analitica(x), 'r--', label='Solución Exacta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Método de Euler Mejorado')
    plt.show()