import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tabulate import tabulate

 
def punto_fijo(g, valor_inicial, max_iteraciones=100, tolerancia=1e-6, precision=5):
    x = valor_inicial
    if(abs(derivative(g, x, dx=tolerancia)) >= 1):
        raise ValueError("No cumple la condición de convergencia")
    
    iteraciones = []
    for i in range(max_iteraciones):
        x_nuevo = g(x)
        error_abs = abs(x_nuevo - x)
        error_rel = abs((x_nuevo - x)*100/x_nuevo)
        iteraciones.append([i+1, round(x, precision), round(x_nuevo, precision), round(error_abs, precision), f"{round(error_rel, 2)} %)"])
        if error_abs < tolerancia:
            print(tabulate(iteraciones, headers=["Iteración", "x", "g(x)", "Error abs", "Error rel"], tablefmt="grid"))
            return x_nuevo
        x = x_nuevo
    raise ValueError("El método no convergió o faltan iteraciones.")


def graficar(f, raiz):
    x_vals = np.linspace(0, 2, 400)
    y_vals = f(x_vals)
    
    plt.plot(x_vals, y_vals, label='$f(x)')
    plt.plot(raiz, f(raiz), 'ro', label=f'Raíz aproximada: x = {raiz:.5f}')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title('Método del Punto Fijo')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()