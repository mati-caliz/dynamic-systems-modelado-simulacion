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

def plot_phase_portrait(f_vectorized, g_vectorized, equilibria, parameters, x_range=(-3, 3), y_range=(-3, 3),
                        density=1.5, results=None, nullclines=None):
    if parameters:
        param_values = {p.name: 1 for p in parameters}
    else:
        param_values = {}

    x_vals = np.linspace(float(x_range[0]), float(x_range[1]), 200)
    y_vals = np.linspace(float(y_range[0]), float(y_range[1]), 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y, **param_values)
    V = g_vectorized(X, Y, **param_values)
    U = np.where(np.iscomplex(U), np.nan, U.real)
    V = np.where(np.iscomplex(V), np.nan, V.real)

    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)

    # Graficar puntos de equilibrio
    for eq in equilibria:
        if hasattr(eq[0], 'free_symbols') or hasattr(eq[1], 'free_symbols'):
            continue
        else:
            try:
                eq_numeric = (float(eq[0].evalf(subs=param_values)), float(eq[1].evalf(subs=param_values)))
                plt.scatter(eq_numeric[0], eq_numeric[1], color="red", s=100, label="Punto de equilibrio")
                plt.text(eq_numeric[0], eq_numeric[1], f'({eq_numeric[0]:.2f}, {eq_numeric[1]:.2f})', color='black',
                         fontsize=9)
            except (TypeError, ValueError):
                continue

    # Graficar nullclines
    if nullclines is not None:
        x_vals_plot = np.linspace(float(x_range[0]), float(x_range[1]), 400)
        y_vals_plot = np.linspace(float(y_range[0]), float(y_range[1]), 400)
        labels_plotted = set()
        for nullcline in nullclines:
            variable = nullcline['variable']
            solutions = nullcline['solutions']
            label = nullcline['label']
            for expr in solutions:
                try:
                    # Verificar si la expresión es completamente real
                    if expr.has(sp.I):
                        print(f"Saltando nullcline '{label}: {variable} = {expr}' porque contiene componentes complejas.")
                        continue

                    if variable == sp.Symbol('y'):
                        y_nullcline = sp.lambdify(sp.Symbol('x'), expr, modules='numpy')
                        y_plot = y_nullcline(x_vals_plot)
                        # Verificar si y_plot tiene componentes complejas
                        if np.iscomplexobj(y_plot):
                            if np.all(np.abs(y_plot.imag) < 1e-5):
                                y_plot = y_plot.real
                            else:
                                print(f"Advertencia: y_plot para nullcline '{label}: {variable} = {expr}' contiene componentes complejas.")
                                y_plot = np.nan * np.ones_like(y_plot)  # Reemplazar con NaN
                        label_full = f"{label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_vals_plot, y_plot, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_vals_plot, y_plot, '--')
                    else:
                        x_nullcline = sp.lambdify(sp.Symbol('y'), expr, modules='numpy')
                        x_plot = x_nullcline(y_vals_plot)
                        # Verificar si x_plot tiene componentes complejas
                        if np.iscomplexobj(x_plot):
                            if np.all(np.abs(x_plot.imag) < 1e-5):
                                x_plot = x_plot.real
                            else:
                                print(f"Advertencia: x_plot para nullcline '{label}: {variable} = {expr}' contiene componentes complejas.")
                                x_plot = np.nan * np.ones_like(x_plot)  # Reemplazar con NaN
                        label_full = f"{label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_plot, y_vals_plot, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_plot, y_vals_plot, '--')
                except Exception as e:
                    print(f"Error al graficar nullcline '{label}: {expr}': {e}")
                    continue

    if results is not None:
        for result in results:
            eq = result["equilibrium"]
            if hasattr(eq[0], 'free_symbols') or hasattr(eq[1], 'free_symbols'):
                continue
            else:
                try:
                    if parameters:
                        param_values = {p.name: 1 for p in parameters}
                        eq_numeric = (float(eq[0].evalf(subs=param_values)), float(eq[1].evalf(subs=param_values)))
                    else:
                        eq_numeric = (float(eq[0]), float(eq[1]))

                    eigenvals = result["eigenvals"]
                    eigenvects = result["eigenvects"]

                    for ev, mult, vects in eigenvects:
                        for vect in vects:
                            # Solo graficar vectores asociados a valores propios reales
                            if not ev.is_real:
                                print(f"Saltando autovector para λ = {ev} porque es complejo.")
                                continue

                            vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                            vect_numeric = np.array([float(comp.evalf()) for comp in vect_simplified])
                            eigvec = vect_numeric / np.linalg.norm(vect_numeric)
                            scale = (x_range[1] - x_range[0]) / 4  # Ajustar escala según el rango
                            # Definir puntos para el vector
                            start_point = np.array(eq_numeric)
                            end_point = start_point + eigvec * scale
                            # Graficar el vector como una flecha
                            plt.arrow(start_point[0], start_point[1],
                                      eigvec[0]*scale, eigvec[1]*scale,
                                      head_width=0.05*scale, head_length=0.1*scale, fc='green', ec='green',
                                      length_includes_head=True)
                            # Añadir etiqueta cerca del final de la flecha
                            plt.text(end_point[0], end_point[1], f'λ={ev}', color='green', fontsize=9)
                except (TypeError, ValueError) as e:
                    print(f"Error al procesar resultados: {e}")
                    continue

    plt.title("Diagrama de Fase del Sistema No Lineal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()
