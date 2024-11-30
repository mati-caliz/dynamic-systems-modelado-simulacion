import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp

def euler(f_sym, initial_point, end, steps, precision=5):
    x_sym, y_sym = sp.symbols('x y')
    f = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
    h = (end - initial_point[0]) / steps
    x = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    x[0] = initial_point[0]
    y[0] = initial_point[1]
    data = [["t", "y"]]
    data.append([round(x[0], precision), round(y[0], precision)])
    for i in range(1, steps + 1):
        x[i] = x[i - 1] + h
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        data.append([round(x[i], precision), round(y[i], precision)])
    print(tabulate(data, headers="firstrow", tablefmt="grid"))
    plot_euler(x, y)

def plot_euler(x, y):
    plt.plot(x, y, label='Aproximación de Euler')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Método de Euler')
    plt.show()

def main():
    x, y = sp.symbols('x y')
    f = x + y
    initial_point = (0, 1)
    end = 2
    steps = 10
    precision = 5
    euler(f, initial_point, end, steps, precision)

if __name__ == "__main__":
    main()
