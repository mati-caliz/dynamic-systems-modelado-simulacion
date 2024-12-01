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

    for eq in equilibria:
        if hasattr(eq[0], 'free_symbols') or hasattr(eq[1], 'free_symbols'):
            continue
        try:
            eq_numeric = (float(eq[0].evalf()), float(eq[1].evalf()))
            plt.scatter(eq_numeric[0], eq_numeric[1], color="red", s=100, label="Punto de equilibrio")
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
                        y_nullcline = sp.lambdify(sp.Symbol('x'), expr, modules='numpy')
                        y_plot = y_nullcline(x_vals_plot)
                        y_plot = y_plot.real if np.isrealobj(y_plot) else np.nan * np.ones_like(y_plot)
                        label_full = f"{label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_vals_plot, y_plot, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_vals_plot, y_plot, '--')
                    else:
                        x_nullcline = sp.lambdify(sp.Symbol('y'), expr, modules='numpy')
                        x_plot = x_nullcline(y_vals_plot)
                        x_plot = x_plot.real if np.isrealobj(x_plot) else np.nan * np.ones_like(x_plot)
                        label_full = f"{label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_plot, y_vals_plot, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_plot, y_vals_plot, '--')
                except Exception:
                    continue

    if results:
        for result in results:
            eq = result["equilibrium"]
            if hasattr(eq[0], 'free_symbols') or hasattr(eq[1], 'free_symbols'):
                continue
            try:
                eq_numeric = (float(eq[0].evalf()), float(eq[1].evalf()))
                eigenvals = result["eigenvals"]
                eigenvects = result["eigenvects"]
                for ev, _, vects in eigenvects:
                    for vect in vects:
                        if not ev.is_real:
                            continue
                        vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                        vect_numeric = np.array([float(comp.evalf()) for comp in vect_simplified])
                        eigvec = vect_numeric / np.linalg.norm(vect_numeric)
                        scale = (x_range[1] - x_range[0]) / 4
                        start_point = np.array(eq_numeric)
                        end_point = start_point + eigvec * scale
                        plt.arrow(start_point[0], start_point[1],
                                  eigvec[0]*scale, eigvec[1]*scale,
                                  head_width=0.05*scale, head_length=0.1*scale, fc='green', ec='green',
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
