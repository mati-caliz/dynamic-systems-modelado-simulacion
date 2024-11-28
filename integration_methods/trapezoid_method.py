import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def area_trapecios(f, desde, hasta, cantidad_trapecios, precision=5):
    a = desde
    b = hasta
    n = cantidad_trapecios
    h = (b - a) / n
    area_acumulada = 0
    resultados = []
    
    x_trap = np.linspace(a, b, n+1)
    y_trap = f(x_trap)
    
    for i in range(n):
        area_i = (h / 2) * (y_trap[i] + y_trap[i+1])
        area_acumulada += area_i
        resultados.append([i+1, round(x_trap[i], precision), round(x_trap[i+1], precision), round(y_trap[i], precision), round(y_trap[i+1], precision), round(area_i, precision)])
    
    print(tabulate(resultados, headers=["i", "xi", "xi+1", "f(xi)", "f(xi+1)", "Área trapecio"], tablefmt="grid"))
    area_total = round(area_acumulada, precision)
    print(f"\nÁrea aproximada bajo la curva desde {a} hasta {b}: {area_total}")
    
    graficar_trapecios(f, x_trap, y_trap, area_total)

    return area_total

def graficar_trapecios(f, x_trap, y_trap, area):
    x_vals = np.linspace(x_trap[0], x_trap[-1], 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    for i in range(len(x_trap) - 1):
        plt.fill([x_trap[i], x_trap[i+1], x_trap[i+1], x_trap[i]],
                 [0, 0, y_trap[i+1], y_trap[i]], 'orange', edgecolor='black', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Área bajo la curva usando trapecios: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()
