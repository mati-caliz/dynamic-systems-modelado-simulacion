import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp

def fixed_point(initial_value, f_sym, g_sym, max_iterations=100, tolerance=1e-6, precision=5):
    x_sym = sp.symbols('x')
    g = sp.lambdify(x_sym, g_sym, 'numpy')
    f = sp.lambdify(x_sym, f_sym, 'numpy')
    h = 1e-5
    deriv = (g(initial_value + h) - g(initial_value - h)) / (2 * h)
    if abs(deriv) >= 1:
        raise ValueError("No cumple la condición de convergencia (|g'(x)| < 1)")
    x = initial_value
    iterations = []
    for i in range(max_iterations):
        x_new = g(x)
        abs_error = abs(x_new - x)
        rel_error = abs((x_new - x) * 100 / x_new) if x_new != 0 else float('inf')
        iterations.append([
            i + 1,
            round(x, precision),
            round(x_new, precision),
            round(abs_error, precision),
            f"{round(rel_error, 2)} %"
        ])
        if abs_error < tolerance:
            print(tabulate(
                iterations,
                headers=["Iteración", "x", "g(x)", "Error Absoluto", "Error Relativo"],
                tablefmt="grid"
            ))
            x_vals = np.linspace(0, 2, 400)
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label='$f(x)$')
            plt.plot(x_new, f(x_new), 'ro', label=f'Raíz aproximada: x = {x_new:.5f}')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.grid(color='gray', linestyle='--', linewidth=0.5)
            plt.title('Método del Punto Fijo')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            plt.legend()
            plt.show()
            return x_new
        x = x_new
    raise ValueError("El método no convergió dentro del número máximo de iteraciones.")

def main():
    x = sp.symbols('x')
    f_sym = x**2 - 2
    g_sym = (x + 2 / x) / 2
    initial_value = 1.0
    max_iterations = 100
    tolerance = 1e-6
    precision = 5
    fixed_point(initial_value, f_sym, g_sym, max_iterations, tolerance, precision)

if __name__ == "__main__":
    main()
