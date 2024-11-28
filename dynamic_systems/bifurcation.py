# bifurcation.py

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, solve

def generate_bifurcation_diagram(f_sym, g_sym, parameter, param_range, variables=(sp.Symbol('x'), sp.Symbol('y'))):
    """
    Genera un diagrama de bifurcación para el sistema dado.

    Args:
        f_sym (sympy expression): Expresión simbólica de f(x, y).
        g_sym (sympy expression): Expresión simbólica de g(x, y).
        parameter (sympy Symbol): Parámetro para el cual se genera el diagrama.
        param_range (tuple): Rango de valores del parámetro.
        variables (tuple): Variables simbólicas, típicamente (x, y).
    """
    equilibria_values = []
    param_values = np.linspace(param_range[0], param_range[1], 200)
    for param_val in param_values:
        f_sub = f_sym.subs({parameter: param_val})
        g_sub = g_sym.subs({parameter: param_val})
        solutions = solve([f_sub, g_sub], variables, dict=True)
        for sol in solutions:
            try:
                eq_x = sol.get(variables[0], variables[0])
                eq_y = sol.get(variables[1], variables[1])
                eq_x_num = float(eq_x.evalf())
                eq_y_num = float(eq_y.evalf())
                equilibria_values.append((param_val, eq_x_num, eq_y_num))
            except (TypeError, ValueError):
                continue
    if not equilibria_values:
        print("No se encontraron puntos de equilibrio para el rango de parámetros dado.")
        return
    param_vals = [val[0] for val in equilibria_values]
    x_vals = [val[1] for val in equilibria_values]
    y_vals = [val[2] for val in equilibria_values]
    plt.figure(figsize=(10, 6))
    plt.plot(param_vals, x_vals, 'b.', label='x equilibria')
    plt.plot(param_vals, y_vals, 'r.', label='y equilibria')
    plt.xlabel(f'Valor del parámetro {parameter}')
    plt.ylabel('Puntos de equilibrio')
    plt.title('Diagrama de Bifurcación')
    plt.legend()
    plt.grid(True)
    plt.show()
