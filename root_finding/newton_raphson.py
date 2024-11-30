import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def newton_raphson(x, f_expr, initial_value, max_iterations=100, precision=8):
    f_prime_expr = sp.diff(f_expr, x)
    f = sp.lambdify(x, f_expr, 'numpy')
    f_prime = sp.lambdify(x, f_prime_expr, 'numpy')

    tolerance = 10 ** -precision
    x_current = initial_value
    counts = []
    iterations = []

    for iteration in range(max_iterations):
        f_current = f(x_current)
        f_prime_current = f_prime(x_current)

        if f_prime_current == 0:
            raise ZeroDivisionError(f"Derivada cero en x = {x_current}. No se puede continuar.")

        x_new = x_current - f_current / f_prime_current
        abs_error = abs(x_new - x_current)
        rel_error = abs((x_new - x_current) * 100 / x_new)

        # Impresión detallada de las primeras tres iteraciones
        if iteration < 3:
            print(f"\nIteración {iteration + 1}:")
            print(f"f({x_current}) = {round(f_current, precision)}")
            print(f"f'({x_current}) = {round(f_prime_current, precision)}")
            print(f"x_{iteration + 1} = {round(x_current, precision)} - ({round(f_current, precision)} / {round(f_prime_current, precision)}) = {round(x_new, precision)}")
            print(f"Error abs = {round(abs_error, precision)}")
            print(f"Error rel = {round(rel_error, 2)} %")

        counts.append([
            iteration + 1,
            round(x_current, precision),
            round(x_new, precision),
            round(abs_error, precision),
            f"{round(rel_error, 2)} %"
        ])
        iterations.append((x_current, f_current, f_prime_current, x_new))

        if abs_error < tolerance:
            print("\nResumen de Iteraciones:")
            print(tabulate(
                counts,
                headers=["Iteración", "x", "Resultado", "Error abs", "Error rel"],
                floatfmt=f".{precision}f",
                tablefmt="grid"
            ))
            graficar(f, root=x_new, precision=precision, iterations=iterations)
            print(f"\nRaíz encontrada: {x_new:.{precision}f}")
            return x_new

        x_current = x_new

    raise ValueError("El método no convergió dentro del número máximo de iteraciones.")


def graficar(fx, root, precision, iterations):
    # Renombrar 'range' a 'rango' para evitar conflictos con la función incorporada
    rango = 5
    x_vals = np.linspace(root - rango, root + rango, 400)
    y_vals = fx(x_vals)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(root, color='red', linestyle='--', label=f'Raíz aproximada: {round(root, precision)}')
    plt.scatter(root, 0, color='red', zorder=5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))

    for idx, (x_current, f_current, f_prime_current, x_new) in enumerate(iterations):
        plt.plot(x_current, f_current, 'o', color=colors[idx])
        plt.plot(x_new, 0, 'x', color=colors[idx])

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica de la función y la raíz encontrada')
    plt.grid(True)
    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x ** 3 - 2 * x - 5
    initial_value = 2
    precision = 8
    max_iterations = 100
    newton_raphson(x, f_x, initial_value, max_iterations, precision)


if __name__ == "__main__":
    main()
