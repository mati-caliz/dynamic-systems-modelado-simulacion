import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def steffensen_aitken(f, g, valor_inicial, max_iteraciones=100, precision=8):
    x = valor_inicial
    tolerancia = 10**-precision
    calculos = []
    
    for i in range(max_iteraciones):
        x1 = g(x)
        x2 = g(x1)
        x_nuevo = x - (x1 - x)**2 / (x2 - 2 * x1 + x)
        
        error_abs = abs(x_nuevo - x)
        error_rel = abs((x_nuevo - x2) * 100 / x_nuevo) if x_nuevo != 0 else np.inf

        if (i + 1 <= 2):
            print(f"\nIteración {i+1}:")
            print(f"x0 = {round(x, precision)}")
            print(f"x1 = g({round(x, precision)}) = {round(x1, precision)}")
            print(f"x2 = g({round(x1, precision)}) = {round(x2, precision)}")
            print(f"x0_nuevo = {round(x, precision)} - (({round(x1, precision)} - {round(x, precision)})^2 / ({round(x2, precision)} - 2 * {round(x1, precision)} + {round(x, precision)})) = {round(x_nuevo, precision)}")
        
        calculos.append([i+1, round(x, precision), round(x1, precision), round(x2, precision), round(x_nuevo, precision), f"{round(error_rel, 2)} %"])
        
        if error_abs < tolerancia:
            print(tabulate(calculos, headers=["Iteración", "x0", "x1", "x2", "Resultado", "Error rel"], floatfmt=f"{precision}f", tablefmt="grid"))
            print(f"Raiz: {round(x_nuevo, precision)}")
            graficar(f, x_nuevo, precision)
            return x_nuevo
        
        x = x_nuevo
    
    raise ValueError("El método no convergió o faltan iteraciones.")

# Función para graficar el resultado del método Steffensen-Aitken
def graficar(fx, raiz, precision):
    x_vals = np.linspace(0, 2, 400)
    y_vals = fx(x_vals)
    
    plt.plot(x_vals, y_vals, label=f'$g(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(raiz, fx(raiz), 'ro', label=f'Raíz aproximada: x = {round(raiz, precision)}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.title('Gráfica del método Steffensen-Aitken')
    plt.show()
