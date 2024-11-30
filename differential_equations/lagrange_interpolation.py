import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp


def lagrange_interpolation(points, num_points=100, precision=5):
    x = sp.symbols('x')
    polynomial = 0
    print("Construcción del polinomio de Lagrange:")
    for i in range(len(points)):
        li = 1
        numerators = []
        denominators = []
        for j in range(len(points)):
            if i != j:
                numerator = f"(x - {round(points[j][0], precision)})"
                denominator = f"({round(points[i][0], precision)} - {round(points[j][0], precision)})"
                numerators.append(numerator)
                denominators.append(denominator)
                li *= (x - points[j][0]) / (points[i][0] - points[j][0])
        numerator_str = ' * '.join(numerators)
        denominator_str = ' * '.join(denominators)

        print(f"\nTérmino l_{i}(x):")
        print(f"   {numerator_str}")
        print(f"  {'-' * max(len(numerator_str), len(denominator_str))}")
        print(f"   {denominator_str}")
        print(f"\n\n{sp.pretty(sp.simplify(li), use_unicode=True)}")
        polynomial += sp.Rational(points[i][1]) * li
    expanded_polynomial = sp.expand(polynomial)
    print(f"\nPolinomio de Lagrange: \n{sp.pretty(expanded_polynomial, use_unicode=True)}")
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    x_eval = np.linspace(x_min, x_max, num_points)
    polynomial_func = np.vectorize(lambda x_val: expanded_polynomial.evalf(subs={x: x_val}))
    y_vals = polynomial_func(x_eval)
    plot_lagrange(points, x_eval, y_vals)

def plot_lagrange(points, x_eval, y_vals):
    x_nodes = [p[0] for p in points]
    y_nodes = [p[1] for p in points]
    plt.plot(x_nodes, y_nodes, 'o', label='Nodos', color='red')
    plt.plot(x_eval, y_vals, '-', label='Interpolación de Lagrange', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Interpolación de Lagrange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    points = [(0, 1), (1, 3), (2, 7)]
    num_points = 100
    precision = 5
    lagrange_interpolation(points, num_points, precision)

if __name__ == "__main__":
    main()
