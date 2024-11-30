import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def secant(x, f_expr, x_initial, x_final, tolerance=1e-6, max_iterations=100, precision=5):
    f = sp.lambdify(x, f_expr, 'numpy')
    iterations = []
    secant_lines = []
    x_values = [x_initial, x_final]

    for i in range(max_iterations):
        f_x_initial = f(x_initial)
        f_x_final = f(x_final)
        denominator = f_x_final - f_x_initial

        if denominator == 0:
            raise ZeroDivisionError("Denominador cero durante la iteración.")

        x_new = x_final - f_x_final * (x_final - x_initial) / denominator
        abs_error = abs(x_new - x_final)
        rel_error = abs((x_new - x_final) * 100 / x_new)

        iterations.append([
            i + 1,
            round(x_initial, precision),
            round(f_x_initial, precision),
            round(x_new, precision),
            round(abs_error, precision),
            f"{round(rel_error, 2)} %"
        ])
        secant_lines.append(((x_initial, f_x_initial), (x_final, f_x_final)))
        x_values.append(x_new)

        if i < 3:
            print(f"\nIteración {i + 1}:")
            print(f"x_{i + 1} = {x_final}")
            print(f"f(x_{i + 1}) = {f_x_final}")
            print(f"x_nuevo = {x_final} - ({f_x_final} * ({x_final} - {x_initial}) / ({f_x_final} - {f_x_initial})) = {x_new}")

        if abs(x_new - x_final) < tolerance:
            print("\nResumen de Iteraciones:")
            print(tabulate(iterations, headers=["Iteración", "x", "f(x)", "Resultado", "Error abs", "Error rel"], tablefmt="grid"))
            plot_function(f, root=x_new, precision=precision, secant_lines=secant_lines, x_values=x_values)
            print(f"\nRaíz encontrada: {x_new:.{precision}f}")
            return x_new

        x_initial, x_final = x_final, x_new

    raise ValueError("El método no convergió o faltan iteraciones.")


def plot_function(fx, root, precision, secant_lines, x_values):
    min_x = min(x_values)
    max_x = max(x_values)
    padding = (max_x - min_x) * 0.2 if max_x != min_x else 1
    range_min = min_x - padding
    range_max = max_x + padding

    x_vals = np.linspace(range_min, range_max, 400)
    y_vals = fx(x_vals)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(root, color='red', linestyle='--', label=f'Raíz aproximada: {root:.{precision}f}')
    plt.scatter(root, 0, color='red', zorder=5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(secant_lines)))

    for idx, ((x1, y1), (x2, y2)) in enumerate(secant_lines):
        plt.plot([x1, x2], [y1, y2], linestyle='--', color=colors[idx], label=f'Secante {idx + 1}')
        plt.plot(x1, y1, 'o', color=colors[idx])
        plt.plot(x2, y2, 'o', color=colors[idx])

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Método de la Secante')
    plt.grid(True)
    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x**3 - 2*x - 5
    x_initial = 2
    x_final = 3
    tolerance = 1e-6
    max_iterations = 100
    precision = 5

    secant(x, f_x, x_initial, x_final, tolerance, max_iterations, precision)


if __name__ == "__main__":
    main()
