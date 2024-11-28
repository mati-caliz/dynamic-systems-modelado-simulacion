import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tabulate import tabulate

def newton_raphson(f, valor_inicial, max_iteraciones=100, precision=8):
    tolerancia = 10**-precision
    x = valor_inicial
    calculos = []

    for i in range(max_iteraciones):
        f_x = f(x)
        df_x = derivative(f, x, dx=tolerancia)
        x_nuevo = x - (f_x / df_x)
        
        error_abs = abs(x_nuevo - x)
        error_rel = abs((x_nuevo - x) * 100 / x_nuevo)

        if(i+1 <= 2):
            print(f"\nIteración {i+1}:")
            print(f"f({i+1}) = {round(f_x, precision)}")
            print(f"f'({i+1}) = {round(df_x, precision)}")
            print(f"x_{i+2} = {round(x, precision)} - ({round(f_x, precision)} / {round(df_x, precision)}) = {round(x_nuevo, precision)}")
        
        calculos.append([ i+1, x, x_nuevo, f"{round(error_rel, 2)} %"])
        
        if error_abs < tolerancia:
            print(tabulate(calculos, headers=["Iteración", "x", "Resultado", "Error rel"], floatfmt=f"{precision}f", tablefmt="grid"))
            print(f"Raiz: {round(x_nuevo, precision)}")
            graficar(f, x_nuevo, precision)                        
            return x_nuevo
        
        x = x_nuevo
    
    raise ValueError("El método no convergió o faltan iteraciones.")

def graficar(fx, raiz, precision):
    x_vals = np.linspace(0, 3, 400)
    y_vals = fx(x_vals)
    
    plt.plot(x_vals, y_vals, label=f'$f(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(raiz, fx(raiz), 'ro', label=f'Raíz aproximada: x = {round(raiz, precision)}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica de la función y su raíz')
    plt.show()
