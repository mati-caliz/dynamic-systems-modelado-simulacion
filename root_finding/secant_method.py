import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
 
def secante(f, x_inicial, x_final, tolerancia=1e-6, max_iteraciones=100, precision=5):
    iteraciones = []
    for i in range(max_iteraciones):
        x_nuevo = x_final - f(x_final) * (x_final - x_inicial) / (f(x_final) - f(x_inicial))
        error_abs = abs(x_nuevo - x_inicial)
        error_rel = abs((x_nuevo - x_inicial)*100/x_nuevo)
        iteraciones.append([i+1, round(x_inicial, precision), round(f(x_inicial), precision), round(x_nuevo, precision), round(error_abs, precision), f"{round(error_rel, 2)} %)"])
        if abs(x_nuevo - x_final) < tolerancia:
            print(tabulate(iteraciones, headers=["Iteración", "x", "f(x)", "Resultado", "Error abs", "Error rel"], tablefmt="grid"))
            return x_nuevo
        x_inicial, x_final = x_final, x_nuevo
    raise ValueError("El método no convergió o faltan iteraciones.")
  

def graficar(f, raiz):
    x = np.linspace(-3, 3, 400)
    y = f(x)
    
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(raiz, color='red', linestyle='--', label=f'Raíz aproximada: {raiz:.5f}')
    plt.scatter(raiz, 0, color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Método de la Secante')
    plt.legend()
    plt.grid(True)
    plt.show()