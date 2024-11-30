import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def newton_raphson(f_expr, valor_inicial, max_iteraciones=100, precision=8):
    x = sp.symbols('x')
    f_prime_expr = sp.diff(f_expr, x)
    f = sp.lambdify(x, f_expr, 'numpy')
    f_prime = sp.lambdify(x, f_prime_expr, 'numpy')

    tolerancia = 10 ** -precision
    x_current = valor_inicial
    calculos = []

    for i in range(max_iteraciones):
        f_current = f(x_current)
        f_prime_current = f_prime(x_current)

        if f_prime_current == 0:
            raise ZeroDivisionError(f"Derivada cero en x = {x_current}. No se puede continuar.")

        x_new = x_current - f_current / f_prime_current
        error_abs = abs(x_new - x_current)
        error_rel = abs((x_new - x_current) * 100 / x_new)

        if i < 3:
            print(f"\nIteración {i + 1}:")
            print(f"f({x_current}) = {round(f_current, precision)}")
            print(f"f'({x_current}) = {round(f_prime_current, precision)}")
            print(
                f"x_{i + 1} = {round(x_current, precision)} - ({round(f_current, precision)} / {round(f_prime_current, precision)}) = {round(x_new, precision)}")

        calculos.append([i + 1, round(x_current, precision), round(x_new, precision), f"{round(error_rel, 2)} %"])

        if error_abs < tolerancia:
            print("\nResumen de Iteraciones:")
            print(tabulate(calculos, headers=["Iteración", "x", "Resultado", "Error rel"], floatfmt=f".{precision}f",
                           tablefmt="grid"))
            print(f"\nRaíz aproximada: x = {round(x_new, precision)}")
            graficar(f, x_new, precision)
            return x_new

        x_current = x_new

    raise ValueError("El método no convergió dentro del número máximo de iteraciones.")


def graficar(fx, raiz, precision):
    rango = 5
    x_vals = np.linspace(raiz - rango, raiz + rango, 400)
    y_vals = fx(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='$f(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(raiz, fx(raiz), 'ro', label=f'Raíz aproximada: x = {round(raiz, precision)}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica de la función y su raíz')
    plt.show()


def main():
    x = sp.symbols('x')
    f_x = x ** 3 - 2 * x - 5
    valor_inicial = 2
    precision = 8
    max_iteraciones = 100
    newton_raphson(f_x, valor_inicial, max_iteraciones, precision)


if __name__ == "__main__":
    main()
