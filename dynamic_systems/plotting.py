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


def plot_phase_portrait(f_vectorized, g_vectorized, equilibria, x_range=(-3, 3), y_range=(-3, 3),
                        density=1.5, results=None, nullclines=None):
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    y_vals = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y)
    V = g_vectorized(X, Y)
    U = np.where(np.iscomplex(U), np.nan, U.real)
    V = np.where(np.iscomplex(V), np.nan, V.real)

    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)

    equilibrium_plotted = False
    for eq in equilibria:
        try:
            eq_numeric = (float(eq[0]), float(eq[1]))
            plt.scatter(eq_numeric[0], eq_numeric[1], color="red", s=100, label="Punto de equilibrio" if not equilibrium_plotted else "")
            equilibrium_plotted = True
            plt.text(eq_numeric[0], eq_numeric[1], f'({eq_numeric[0]:.2f}, {eq_numeric[1]:.2f})', color='black',
                     fontsize=9)
        except (TypeError, ValueError):
            continue

    if nullclines:
        x_vals_plot = np.linspace(x_range[0], x_range[1], 400)
        y_vals_plot = np.linspace(y_range[0], y_range[1], 400)
        labels_plotted = set()
        for nullcline in nullclines:
            variable = nullcline['variable']
            solutions = nullcline['solutions']
            label = nullcline['label']
            for expr in solutions:
                if expr.has(sp.I):
                    continue
                try:
                    if variable == sp.Symbol('y'):
                        independent_var = sp.Symbol('x')
                        plot_x = x_vals_plot
                        plot_y = sp.lambdify(independent_var, expr, modules='numpy')(plot_x)
                    else:
                        independent_var = sp.Symbol('y')
                        plot_y = y_vals_plot
                        plot_x = sp.lambdify(independent_var, expr, modules='numpy')(plot_y)

                    if np.isrealobj(plot_x if variable == sp.Symbol('x') else plot_y):
                        plot_x = plot_x.real
                        plot_y = plot_y.real
                    else:
                        plot_x = np.nan * np.ones_like(plot_x)
                        plot_y = np.nan * np.ones_like(plot_y)

                    label_full = f"{label}: {variable} = {expr}"
                    if label_full not in labels_plotted:
                        plt.plot(plot_x, plot_y, '--', label=label_full)
                        labels_plotted.add(label_full)
                    else:
                        plt.plot(plot_x, plot_y, '--')
                except Exception:
                    continue

    if results:
        for result in results:
            eq = result["equilibrium"]
            try:
                eq_numeric = (float(eq[0]), float(eq[1]))
                eigenvects = result["eigenvects"]
                for ev, _, vects in eigenvects:
                    for vect in vects:
                        if not ev.is_real:
                            continue
                        vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                        vect_numeric = np.array([float(comp.evalf()) for comp in vect_simplified])
                        if np.linalg.norm(vect_numeric) == 0:
                            continue
                        eigvec = vect_numeric
                        scale = (x_range[1] - x_range[0]) / 4
                        end_point = np.array(eq_numeric) + eigvec * scale
                        plt.arrow(eq_numeric[0], eq_numeric[1],
                                  eigvec[0] * scale, eigvec[1] * scale,
                                  head_width=0.05 * scale, head_length=0.1 * scale, fc='green', ec='green',
                                  length_includes_head=True)
                        plt.text(end_point[0], end_point[1], f'λ={ev}', color='green', fontsize=9)
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
