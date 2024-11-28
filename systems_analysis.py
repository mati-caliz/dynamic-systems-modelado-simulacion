import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, lambdify, solve, simplify, sympify, pprint, nsimplify, S, Rational, Expr, exp

x, y, a = symbols('x y a')

def process_system(f_sym, g_sym, parameters=None):
    if parameters is None:
        parameters = set()
    else:
        parameters = set(parameters)

    def ensure_sympy_expression(func):
        expr = sympify(func, evaluate=False)
        expr = nsimplify(expr, rational=True)
        return expr

    f_sym = ensure_sympy_expression(f_sym)
    g_sym = ensure_sympy_expression(g_sym)
    compute_jacobian_symbolic(f_sym, g_sym, (x, y))
    equilibria = find_equilibria_symbolic(f_sym, g_sym, parameters)
    nullclines = compute_nullclines(f_sym, g_sym)
    if equilibria:
        print("Puntos de equilibrio encontrados:")
        for eq in equilibria:
            pprint(eq)
        results = analyze_equilibria(equilibria, f_sym, g_sym, parameters)
        display_results(results)
    else:
        print("No se encontraron puntos de equilibrio.")
        results = []
    general_solution = compute_general_solution(f_sym, g_sym)
    print("\nSolución general del sistema:")
    print(f"x(t) = {general_solution[0]}")
    print(f"y(t) = {general_solution[1]}")
    print("\nNuclinas del sistema:")
    display_nullclines(nullclines)
    f_num, g_num = lambdify_functions(x, y, f_sym, g_sym, parameters)
    run_system(f_num, g_num, equilibria, parameters, results, nullclines)

    if parameters:
        param = next(iter(parameters))
        generate_bifurcation_diagram(f_sym, g_sym, param, param_range=(-2, 2))

def compute_jacobian_symbolic(f_sym, g_sym, variables):
    jacobian_matrix = Matrix([[simplify(f_sym.diff(var)) for var in variables],
                              [simplify(g_sym.diff(var)) for var in variables]])
    print("Matriz Jacobiana del Sistema:")
    pprint(jacobian_matrix)
    print()

def lambdify_functions(x_sym, y_sym, f_sym, g_sym, parameters):
    vars = (x_sym, y_sym) + tuple(parameters)
    f = lambdify(vars, f_sym, modules='numpy')
    g = lambdify(vars, g_sym, modules='numpy')
    return f, g

def vectorize_functions(f, g):
    f_vectorized = np.vectorize(f)
    g_vectorized = np.vectorize(g)
    return f_vectorized, g_vectorized

def find_equilibria_symbolic(f_sym, g_sym, parameters):
    variables = (x, y)
    solutions = solve([f_sym, g_sym], variables, dict=True, rational=True)
    equilibria = []
    if solutions:
        for sol in solutions:
            eq_x = sol.get(x, x)
            eq_y = sol.get(y, y)
            equilibria.append((simplify(eq_x), simplify(eq_y)))
    return equilibria

def compute_nullclines(f_sym, g_sym):
    nullclines = []
    nullcline_f_y = solve(f_sym, y)
    if nullcline_f_y:
        nullclines.append({'variable': y, 'solutions': nullcline_f_y, 'label': "Nuclina x'"})
    else:
        nullcline_f_x = solve(f_sym, x)
        if nullcline_f_x:
            nullclines.append({'variable': x, 'solutions': nullcline_f_x, 'label': "Nuclina x'"})
    nullcline_g_y = solve(g_sym, y)
    if nullcline_g_y:
        nullclines.append({'variable': y, 'solutions': nullcline_g_y, 'label': "Nuclina y'"})
    else:
        nullcline_g_x = solve(g_sym, x)
        if nullcline_g_x:
            nullclines.append({'variable': x, 'solutions': nullcline_g_x, 'label': "Nuclina y'"})
    return nullclines

def display_nullclines(nullclines):
    for nullcline in nullclines:
        variable = nullcline['variable']
        solutions = nullcline['solutions']
        label = nullcline['label']
        print(f"{label}:")
        for expr in solutions:
            print(f"  {variable} = {expr}")

def analyze_equilibria(equilibria, f_sym, g_sym, parameters):
    results = []
    for eq in equilibria:
        if not isinstance(eq[0], Expr) or not isinstance(eq[1], Expr):
            print(f"\nEl equilibrio {eq} no es una expresión simbólica válida y se omitirá.")
            continue
        if eq[0].free_symbols or eq[1].free_symbols:
            print(f"\nEl equilibrio {eq} es simbólico y no se analizará individualmente.")
            continue
        J = compute_jacobian_at_equilibrium(f_sym, g_sym, eq)
        eigenvals = J.eigenvals()
        eigenvects = J.eigenvects()
        # Calcular valores y vectores propios numéricos
        J_numeric = np.array(J.evalf(), dtype=float)
        eigvals_num, eigvecs_num = np.linalg.eig(J_numeric)
        results.append({
            "equilibrium": eq,
            "jacobian": J,
            "eigenvals": eigenvals,
            "eigenvects": eigenvects,
            "jacobian_numeric": J_numeric,
            "eigenvalues_numeric": eigvals_num,
            "eigenvectors_numeric": eigvecs_num
        })
    return results

def compute_jacobian_at_equilibrium(f_sym, g_sym, eq):
    jacobian_matrix = Matrix([[simplify(f_sym.diff(var)) for var in (x, y)],
                              [simplify(g_sym.diff(var)) for var in (x, y)]])
    J_at_eq = jacobian_matrix.subs({x: eq[0], y: eq[1]})
    return J_at_eq

def classify_equilibrium(eigenvalues):
    real_parts = [ev.as_real_imag()[0] for ev in eigenvalues]
    imag_parts = [ev.as_real_imag()[1] for ev in eigenvalues]
    real_positive = [re.is_positive for re in real_parts]
    real_negative = [re.is_negative for re in real_parts]
    imag_nonzero = [im.is_zero == False for im in imag_parts]
    if None in real_positive or None in real_negative or None in imag_nonzero:
        return "No se puede determinar la clasificación del equilibrio debido a valores propios simbólicos."
    if all(im for im in imag_nonzero):
        if all(re == True for re in real_positive):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
        elif all(re == True for re in real_negative):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif all(re == False for re in real_positive + real_negative):
            return "Centro (valores propios puramente imaginarios)"
        else:
            return "Espiral Silla (valores propios complejos con partes reales de signos opuestos)"
    else:
        if all(re == True for re in real_positive):
            return "Nodo Inestable (valores propios reales y positivos)"
        elif all(re == True for re in real_negative):
            return "Nodo Estable (valores propios reales y negativos)"
        elif any(re == True for re in real_positive) and any(re == True for re in real_negative):
            return "Punto Silla (valores propios reales de signos opuestos)"
        elif all(re == False for re in real_positive + real_negative):
            return "Centro o Nodo Degenerado (valores propios reales nulos o repetidos)"
        else:
            return "Otro tipo de equilibrio"

def display_results(results):
    for result in results:
        eq = result["equilibrium"]
        J = result["jacobian"]
        eigenvals = result["eigenvals"]
        eigenvects = result["eigenvects"]
        print("\nPunto de equilibrio:")
        pprint(eq)
        print("Matriz Jacobiana en el equilibrio:")
        pprint(J)
        print("Valores propios:")
        for ev in eigenvals:
            ev_simplified = nsimplify(ev, rational=True)
            ev_str = str(ev_simplified).replace('I', 'i')
            print(f"λ = {ev_str}")
        print("Vectores propios:")
        for ev, mult, vects in eigenvects:
            ev_simplified = nsimplify(ev, rational=True)
            ev_str = str(ev_simplified).replace('I', 'i')
            for vect in vects:
                vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                vect_components = [str(comp).replace('I', 'i') for comp in vect_simplified]
                vect_str = f"({', '.join(vect_components)})"
                print(f"Vector propio asociado a λ = {ev_str}: {vect_str}")
        eigenvalues = list(eigenvals.keys())
        classification = classify_equilibrium(eigenvalues)
        print("\nClasificación del sistema:", classification)

def compute_general_solution(f_sym, g_sym):
    variables = (x, y)
    A = Matrix([[f_sym.coeff(var) for var in variables],
                [g_sym.coeff(var) for var in variables]])
    eigenvals = A.eigenvals()
    eigenvects = A.eigenvects()
    c = symbols('c1:%d' % (len(eigenvects)*A.shape[0]+1))
    t = symbols('t')
    sol_x = 0
    sol_y = 0
    idx = 0
    for ev, mult, vects in eigenvects:
        for vect in vects:
            term = c[idx]*exp(ev*t)*vect
            sol_x += term[0]
            sol_y += term[1]
            idx += 1
    return (simplify(sol_x), simplify(sol_y))

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
                    if variable == y:
                        y_nullcline = lambdify(x, expr, modules='numpy')
                        y_plot = y_nullcline(x_vals)
                        y_plot = np.real_if_close(y_plot)
                        label_full = f"{base_label}: {variable} = {expr}"
                        if label_full not in labels_plotted:
                            plt.plot(x_vals, y_plot, '--', label=label_full)
                            labels_plotted.add(label_full)
                        else:
                            plt.plot(x_vals, y_plot, '--')
                    else:
                        x_nullcline = lambdify(y, expr, modules='numpy')
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

    # Graficar los autovectores en los puntos de equilibrio
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
    plt.legend()
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()

def run_system(f, g, equilibria, parameters, results, nullclines):
    f_vectorized, g_vectorized = vectorize_functions(f, g)
    plot_phase_portrait(f_vectorized, g_vectorized, equilibria, parameters, results=results, nullclines=nullclines)

def generate_bifurcation_diagram(f_sym, g_sym, parameter, param_range, variables=(x, y)):
    equilibria_values = []
    param_values = np.linspace(param_range[0], param_range[1], 200)
    for param_val in param_values:
        f_sub = f_sym.subs({parameter: param_val})
        g_sub = g_sym.subs({parameter: param_val})
        solutions = solve([f_sub, g_sub], variables, dict=True)
        for sol in solutions:
            try:
                eq_x = sol.get(x, x)
                eq_y = sol.get(y, y)
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

def main():
    x_function = 2 * x - 2 * y
    y_function = 4 * x - 2 * y
    process_system(x_function, y_function)

if __name__ == "__main__":
    main()
