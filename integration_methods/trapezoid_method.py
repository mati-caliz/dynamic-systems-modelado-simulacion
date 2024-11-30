import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def trapezoidal_area(x, f_x, start, end, num_trapezoids, precision=5):
    f_num = sp.lambdify(x, f_x, 'numpy')
    a = start
    b = end
    n = num_trapezoids
    h = (b - a) / n
    accumulated_area = 0
    results = []

    x_trap = np.linspace(a, b, n + 1)
    y_trap = f_num(x_trap)

    for i in range(n):
        area_i = (h / 2) * (y_trap[i] + y_trap[i + 1])
        accumulated_area += area_i
        results.append([
            i + 1,
            round(x_trap[i], precision),
            round(x_trap[i + 1], precision),
            round(y_trap[i], precision),
            round(y_trap[i + 1], precision),
            round(area_i, precision)
        ])

    print(tabulate(
        results,
        headers=["i", "xi", "xi+1", "f(xi)", "f(xi+1)", "Área Trapecio"],
        tablefmt="grid"
    ))

    total_area = round(accumulated_area, precision)
    print(f"\nÁrea aproximada bajo la curva desde {a} hasta {b}: {total_area}")

    plot_trapezoidal(f_num, x_trap, y_trap, total_area)

    return total_area


def plot_trapezoidal(f, x_trap, y_trap, area):
    x_vals = np.linspace(x_trap[0], x_trap[-1], 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')

    for i in range(len(x_trap) - 1):
        plt.fill([
            x_trap[i],
            x_trap[i + 1],
            x_trap[i + 1],
            x_trap[i]
        ], [
            0,
            0,
            y_trap[i + 1],
            y_trap[i]
        ], 'orange', edgecolor='black', alpha=0.5)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Área bajo la curva usando trapecios: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x ** 3 - 2 * x - 5
    start = 0
    end = 3
    num_trapezoids = 100  # Debe ser par
    precision = 5

    trapezoidal_area(x, f_x, start, end, num_trapezoids, precision)


if __name__ == "__main__":
    main()
