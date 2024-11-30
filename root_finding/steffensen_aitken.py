import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def steffensen_aitken(x, f_x, g_x, initial_value, max_iterations=100, precision=8):
    f_num = sp.lambdify(x, f_x, 'numpy')
    g_num = sp.lambdify(x, g_x, 'numpy')

    x_current = initial_value
    tolerance = 10 ** -precision
    accum = []

    for iteration in range(max_iterations):
        x1 = g_num(x_current)
        x2 = g_num(x1)
        denominator = x2 - 2 * x1 + x_current

        if denominator == 0:
            raise ZeroDivisionError("División por cero durante las iteraciones.")

        x_new = x_current - ((x1 - x_current) ** 2) / denominator

        error_abs = abs(x_new - x_current)
        error_rel = abs((x_new - x2) * 100 / x_new) if x_new != 0 else np.inf

        if iteration < 3:
            print(f"\nIteración {iteration + 1}:")
            print(f"x0 = {round(x_current, precision)}")
            print(f"x1 = g({round(x_current, precision)}) = {round(x1, precision)}")
            print(f"x2 = g({round(x1, precision)}) = {round(x2, precision)}")
            print(
                f"x_nuevo = {round(x_current, precision)} - (({round(x1, precision)} - {round(x_current, precision)})^2 / ({round(x2, precision)} - 2 * {round(x1, precision)} + {round(x_current, precision)})) = {round(x_new, precision)}")

        accum.append([
            iteration + 1,
            round(x_current, precision),
            round(x1, precision),
            round(x2, precision),
            round(x_new, precision),
            f"{round(error_abs, precision)}",
            f"{round(error_rel, 2)} %"
        ])

        if error_abs < tolerance:
            print(tabulate(
                accum,
                headers=["Iteración", "x0", "x1", "x2", "Resultado", "Error Absoluto", "Error Relativo"],
                floatfmt=f".{precision}f",
                tablefmt="grid"
            ))
            print(f"\nRaíz aproximada: {round(x_new, precision)}")
            plot_function(f_num, x_new, precision)
            return x_new

        x_current = x_new

    raise ValueError("El método no convergió dentro del número máximo de iteraciones.")


def plot_function(f_num, root, precision):
    margin = 1
    x_vals = np.linspace(root - margin, root + margin, 400)
    y_vals = f_num(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(root, color='red', linestyle='--', label=f'Raíz aproximada: x = {round(root, precision)}')
    plt.scatter(root, 0, color='red', zorder=5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica del método Steffensen-Aitken')
    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x ** 3 - 2 * x - 5
    g_x = (2 * x + 5) ** (1 / 3)

    initial_value = 2
    precision = 8
    max_iterations = 100

    steffensen_aitken(x, f_x, g_x, initial_value, max_iterations, precision)


if __name__ == "__main__":
    main()
