import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def simpson_area(x, f_x, start, end, num_intervals, precision=5):
    if num_intervals % 2 != 0:
        raise ValueError("El número de intervalos debe ser par para el método de Simpson.")

    f_num = sp.lambdify(x, f_x, 'numpy')

    a = start
    b = end
    n = num_intervals
    h = (b - a) / n
    results = []

    odd_sum = 0
    for i in range(1, n, 2):
        x_i = a + i * h
        odd_sum += f_num(x_i)

    even_sum = 0
    for i in range(2, n, 2):
        x_i = a + i * h
        even_sum += f_num(x_i)

    accumulated_area = (h / 3) * (f_num(a) + 4 * odd_sum + 2 * even_sum + f_num(b))

    for i in range(n + 1):
        x_i = a + i * h
        results.append([i, round(x_i, precision), round(f_num(x_i), precision)])

    print(tabulate(results, headers=["i", "x_i", "f(x_i)"], tablefmt="grid"))

    area = round(accumulated_area, precision)
    print(f"\nÁrea aproximada bajo la curva desde {a} hasta {b}: {area}")

    plot_simpson(f_num, start, end, area, n)
    return area


def plot_simpson(f_num, start, end, area, n):
    x_vals = np.linspace(start, end, 400)
    y_vals = f_num(x_vals)
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')

    x_points = np.linspace(start, end, n + 1)
    y_points = f_num(x_points)
    plt.plot(x_points, y_points, 'ro', label='Puntos de Simpson')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Área bajo la curva usando Simpson: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x ** 3 - 2 * x - 5
    start = 0
    end = 3
    num_intervals = 100 # Debe ser par
    precision = 5

    simpson_area(x, f_x, start, end, num_intervals, precision)


if __name__ == "__main__":
    main()
