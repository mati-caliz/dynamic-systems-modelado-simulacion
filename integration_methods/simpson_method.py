import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def area_simpson(f, desde, hasta, cantidad_intervalos, precision=5):
    if cantidad_intervalos % 2 != 0:
        raise ValueError("El número de intervalos debe ser par para el método de Simpson.")
    
    a = desde
    b = hasta
    n = cantidad_intervalos
    h = (b - a) / n
    area_acumulada = 0
    resultados = []
    suma_impares = 0
    for i in range(2, n, 2):
        x_i = a + i * h
        suma_impares += f(x_i)

    suma_pares = 0
    for i in range(1, n, 2):
        x_i = a + i * h
        suma_pares += f(x_i)

    area_acumulada = (h / 3) * (f(a) + 4 * suma_pares + 2 * suma_impares + f(b))
    for i in range(n + 1):
        x_i = a + i * h
        resultados.append([i, round(x_i, precision), round(f(x_i), precision)])

    print(tabulate(resultados, headers=["i", "x_i", "f(x_i)"], tablefmt="grid"))
    area = round(area_acumulada, precision)
    print(f"\nÁrea aproximada bajo la curva desde {a} hasta {b}: {area}")
    graficar_simpson(f, desde, hasta, area, n)
    return area


def graficar_simpson(f, desde, hasta, area, n):
    x_vals = np.linspace(desde, hasta, 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    x_points = np.linspace(desde, hasta, n+1)
    y_points = f(x_points)
    plt.plot(x_points, y_points, 'ro')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Área bajo la curva usando Simpson: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()