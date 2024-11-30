import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp

def euler_improved(f_sym, exact_solution_sym, initial_point, end, steps, precision=5):
    x_sym, y_sym = sp.symbols('x y')
    f = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
    exact_solution = sp.lambdify(x_sym, exact_solution_sym, 'numpy')
    h = (end - initial_point[0]) / steps
    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    x[0] = initial_point[0]
    y[0] = initial_point[1]
    error = np.zeros(steps + 1)
    y_real = np.zeros(steps + 1)
    y_real[0] = exact_solution(x[0])
    error[0] = abs(y[0] - y_real[0])
    table = []
    table.append([0, round(x[0], precision), "-", round(y[0], precision), round(y_real[0], precision), round(error[0], precision)])
    for i in range(steps):
        y_pred = y[i] + h * f(x[i], y[i])
        y[i+1] = y[i] + (h / 2) * (f(x[i], y[i]) + f(x[i] + h, y_pred))
        x[i+1] = x[i] + h
        y_real[i+1] = exact_solution(x[i+1])
        error[i+1] = abs(y[i+1] - y_real[i+1])
        table.append([i+1, round(x[i+1], precision), round(y_pred, precision), round(y[i+1], precision), round(y_real[i+1], precision), round(error[i+1], precision)])
    headers = ["Iteración", "x", "y* (Predicción)", "y_n (Euler Mejorado)", "y_real (Exacta)", "Error"]
    float_format = [None, f".{precision}f", None, f".{precision}f", f".{precision}f", f".{precision}f"]
    print(tabulate(table, headers=headers, floatfmt=float_format, tablefmt="grid"))
    plot_euler(x, y, exact_solution)

def plot_euler(x, y, exact_solution):
    plt.plot(x, y, 'o-', label='Euler Mejorado')
    plt.plot(x, exact_solution(x), 'r--', label='Solución Exacta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Método de Euler Mejorado')
    plt.show()

def main():
    x, y = sp.symbols('x y')
    f = x + y
    exact_solution = 2 * sp.exp(x) - x - 1
    initial_point = (0, 1)
    end = 2
    steps = 10
    precision = 5
    euler_improved(f, exact_solution, initial_point, end, steps, precision)

if __name__ == "__main__":
    main()
