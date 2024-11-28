# plotting.py

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import nsimplify

def display_nullclines(nullclines):
    for nullcline in nullclines:
        variable = nullcline['variable']
        solutions = nullcline['solutions']
        label = nullcline['label']
        print(f"{label}:")
        for expr in solutions:
            print(f"  {variable} = {expr}")

def plot_phase_portrait(f_vectorized, g_vectorized, equilibria, parameters, x_range=(-3, 3), y_range=(-3, 3), density=1.5, results=None, nullclines=None):
    if parameters:
        param_values = {p.name: 1 for p in parameters}
    else:
        param_values = {}

    x_vals = np.linspace(float(x_range[0]), float(x_range[1]), 200)
    y_vals = np.linspace(float(y_range[0]), float(y_range[1]), 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y, **param_values)
    V = g_vectorized(X, Y, **param_values)
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)
    for eq in equilibria:
        if eq[0].free_symbols or eq[1].free_symbols:
            continue
        else:
            try:
                eq_numeric = (float(eq[0].evalf(subs=param_values)), float(eq[1].evalf(subs=param_values)))
                plt.scatter(eq_numeric[0], eq_numeric[1], color="red", s=100, label="Punto de equilibrio")
                plt.text(eq_numeric[0], eq_numeric[1], f'({eq_numeric[0]:.2f}, {eq_numeric[1]:.2f})', color='black',
                         fontsize=9)
            except (TypeError, ValueError):
                continue

    if nullclines is not None:
        x_vals = np.linspace(float(x_range[0]), float(x_range[1]), 400)
        y_vals = np.linspace(float(y_range[0]), float(y_range[1]), 400)
        labels_plotted = set()
        for nullcline in nullclines:
            variable = nullcline['variable']
            solutions = nullcline['solutions']
            base_label = nullcline['label']
            for expr in solutions:
                try:
                    if variable == sp.Symbol('y'):
                        y_nullcline = sp.lambdify(sp.Symbol('x'), expr, modules='numpy')
                        y_plot = y_nullcline(x_vals)
                        y_plot = np.real_if_close(y_plot)
                        label_full = f"{base_label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_vals, y_plot, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_vals, y_plot, '--')
                    else:
                        x_nullcline = sp.lambdify(sp.Symbol('y'), expr, modules='numpy')
                        x_plot = x_nullcline(y_vals)
                        x_plot = np.real_if_close(x_plot)
                        label_full = f"{base_label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_plot, y_vals, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_plot, y_vals, '--')
                except Exception as e:
                    continue

    if results is not None:
        plotted_vectors = set()
        for result in results:
            eq = result["equilibrium"]
            if eq[0].free_symbols or eq[1].free_symbols:
                continue
            else:
                try:
                    if parameters:
                        param_values = {p.name: 1 for p in parameters}
                        eq_numeric = (float(eq[0].evalf(subs=param_values)), float(eq[1].evalf(subs=param_values)))
                    else:
                        eq_numeric = (float(eq[0]), float(eq[1]))
                    eigenvects = result["eigenvects"]
                    for ev, mult, vects in eigenvects:
                        for vect in vects:
                            vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                            vect_numeric = np.array([float(comp.evalf()) for comp in vect_simplified])
                            eigvec = vect_numeric / np.linalg.norm(vect_numeric)
                            scale = (x_range[1] - x_range[0]) / 2
                            x_line = [eq_numeric[0] - eigvec[0]*scale, eq_numeric[0] + eigvec[0]*scale]
                            y_line = [eq_numeric[1] - eigvec[1]*scale, eq_numeric[1] + eigvec[1]*scale]
                            label = f"Autovector asociado a λ = {ev}"
                            if label not in plotted_vectors:
                                plt.plot(x_line, y_line, '-', linewidth=2, label=label)
                                plotted_vectors.add(label)
                            else:
                                plt.plot(x_line, y_line, '-', linewidth=2)
                except (TypeError, ValueError):
                    continue

    plt.title("Diagrama de Fase del Sistema No Lineal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()
