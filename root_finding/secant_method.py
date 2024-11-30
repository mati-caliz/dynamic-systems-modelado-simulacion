import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def secante(x, f_expr, x_inicial, x_final, tolerancia=1e-6, max_iteraciones=100, precision=5):
    f = sp.lambdify(x, f_expr, 'numpy')
    iteraciones = []
    secant_lines = []
    x_values = [x_inicial, x_final]

    for i in range(max_iteraciones):
        f_x_inicial = f(x_inicial)
        f_x_final = f(x_final)
        denominator = f_x_final - f_x_inicial

        if denominator == 0:
            raise ZeroDivisionError("Denominador cero durante la iteración.")

        x_nuevo = x_final - f_x_final * (x_final - x_inicial) / denominator
        error_abs = abs(x_nuevo - x_final)
        error_rel = abs((x_nuevo - x_final) * 100 / x_nuevo)

        iteraciones.append([
            i + 1,
            round(x_inicial, precision),
            round(f_x_inicial, precision),
            round(x_nuevo, precision),
            round(error_abs, precision),
            f"{round(error_rel, 2)} %"
        ])
        secant_lines.append(((x_inicial, f_x_inicial), (x_final, f_x_final)))
        x_values.append(x_nuevo)

        # Impresión detallada de las primeras 3 iteraciones
        if i < 3:
            print(f"\nIteración {i + 1}:")
            print(f"x_{i + 1} = {x_final}")
            print(f"f(x_{i + 1}) = {f_x_final}")
            print(f"x_nuevo = {x_final} - ({f_x_final} * ({x_final} - {x_inicial}) / ({f_x_final} - {f_x_inicial})) = {x_nuevo}")

        if abs(x_nuevo - x_final) < tolerancia:
            print("\nResumen de Iteraciones:")
            print(tabulate(iteraciones, headers=["Iteración", "x", "f(x)", "Resultado", "Error abs", "Error rel"], tablefmt="grid"))
            graficar(f, raiz=x_nuevo, precision=precision, secant_lines=secant_lines, x_values=x_values)
            print(f"\nRaíz encontrada: {x_nuevo:.{precision}f}")
            return x_nuevo

        x_inicial, x_final = x_final, x_nuevo

    raise ValueError("El método no convergió o faltan iteraciones.")


def graficar(fx, raiz, precision, secant_lines, x_values):
    # Determinar el rango de la gráfica basado en los valores de x utilizados
    min_x = min(x_values)
    max_x = max(x_values)
    padding = (max_x - min_x) * 0.2 if max_x != min_x else 1
    rango_min = min_x - padding
    rango_max = max_x + padding

    x_vals = np.linspace(rango_min, rango_max, 400)
    y_vals = fx(x_vals)

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(raiz, color='red', linestyle='--', label=f'Raíz aproximada: {raiz:.{precision}f}')
    plt.scatter(raiz, 0, color='red', zorder=5)

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
    x_inicial = 2
    x_final = 3
    tolerancia = 1e-6
    max_iteraciones = 100
    precision = 5

    secante(x, f_x, x_inicial, x_final, tolerancia, max_iteraciones, precision)


if __name__ == "__main__":
    main()
