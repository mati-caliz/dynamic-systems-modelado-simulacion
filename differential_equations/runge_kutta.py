import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp

def runge_kutta(f_sym, initial_point, end, steps, precision=5):
    x_sym, y_sym = sp.symbols('x y')
    f = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
    h = (end - initial_point[0]) / steps
    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    x[0] = initial_point[0]
    y[0] = initial_point[1]
    table = [["Iteración", "x_n", "y_n", "k1", "k2", "k3", "k4", "x_n+1", "y_n+1"]]
    for i in range(steps):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + k1*h/2)
        k3 = f(x[i] + h/2, y[i] + k2*h/2)
        k4 = f(x[i] + h, y[i] + k3*h)
        x_next = x[i] + h
        y_next = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        table.append([
            i+1,
            round(x[i], precision),
            round(y[i], precision),
            round(k1, precision),
            round(k2, precision),
            round(k3, precision),
            round(k4, precision),
            round(x_next, precision),
            round(y_next, precision)
        ])
        x[i+1] = x_next
        y[i+1] = y_next
    print(tabulate(table, headers="firstrow", tablefmt="grid", floatfmt=f".{precision}f"))
    plot_runge_kutta(x, y)

def plot_runge_kutta(x, y):
    plt.plot(x, y, 'o-', label='Runge Kutta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Método de Runge Kutta')
    plt.show()

def main():
    x, y = sp.symbols('x y')
    f = x + y
    initial_point = (0, 1)
    end = 2
    steps = 10
    precision = 5
    runge_kutta(f, initial_point, end, steps, precision)

if __name__ == "__main__":
    main()
