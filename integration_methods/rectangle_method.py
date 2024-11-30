import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def rectangle_area(x, f_x, start, end, num_rectangles, point="medio", precision=5):
    f_num = sp.lambdify(x, f_x, 'numpy')

    a = start
    b = end
    n = num_rectangles
    h = (b - a) / n
    accumulated_area = 0
    results = []
    x_rect = []
    y_rect = []
    for i in range(n):
        if point == "izquierdo":
            x_i = a + i * h
        elif point == "derecho":
            x_i = a + i * h + h
        else:
            x_i = a + i * h + h / 2

        f_xi = f_num(x_i)
        area_i = f_xi * h
        accumulated_area += area_i

        if point == "izquierdo":
            x_plot = a + i * h
        elif point == "derecho":
            x_plot = a + (i + 1) * h
        else:
            x_plot = a + i * h + h / 2

        x_rect.append(x_plot)
        y_rect.append(f_num(x_plot))
        results.append([
            round(i + 1, precision),
            round(a + i * h + h, precision),
            round(x_i, precision),
            round(f_xi, precision),
            round(area_i, precision)
        ])

    area = round(accumulated_area, precision)
    print(f"Área aproximada bajo la curva desde {a} hasta {b}: {area}")
    print(f"Se calculó utilizando punto {point}.")

    results.append(["", "", "", "", round(accumulated_area, precision)])
    print(tabulate(results, headers=["i", "x", "xi", "f(xi)", "Área [f(xi)*h]"], tablefmt="grid"))

    plot_area(x, f_num, area, a, b, x_rect, y_rect, h, point)
    return area


def plot_area(x, f_num, area, start, end, x_rect, y_rect, rect_width, point):
    x_vals = np.linspace(start, end, 400)
    y_vals = f_num(x_vals)

    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')

    for x_val, y_val in zip(x_rect, y_rect):
        if point == "izquierdo":
            plt.bar(x_val, y_val, width=rect_width, align='edge', edgecolor='red', color='red', alpha=0.3)
        elif point == "derecho":
            plt.bar(x_val - rect_width, y_val, width=rect_width, align='edge', edgecolor='green', color='green',
                    alpha=0.3)
        else:
            plt.bar(x_val - rect_width / 2, y_val, width=rect_width, align='center', edgecolor='orange', color='orange',
                    alpha=0.3)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Área bajo la curva usando punto {point}: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()

    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x ** 3 - 2 * x - 5
    start = 0
    end = 3
    num_rectangles = 100
    point = "medio"  # Opciones: "izquierdo", "derecho", "medio"
    precision = 5

    rectangle_area(x, f_x, start, end, num_rectangles, point, precision)


if __name__ == "__main__":
    main()
