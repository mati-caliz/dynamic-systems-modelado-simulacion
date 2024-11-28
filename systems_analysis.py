import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Symbol, Expr, Matrix, lambdify, solve, simplify, S, sympify

x, y = symbols('x y')


def process_system(f_sym, g_sym):
    def ensure_sympy_expression(func):
        return S(func) if isinstance(func, (int, float)) else sympify(func)

    f_sym = ensure_sympy_expression(f_sym)
    g_sym = ensure_sympy_expression(g_sym)
    compute_jacobian_symbolic(f_sym, g_sym, (x, y))
    equilibria = find_equilibria_symbolic(f_sym, g_sym)
    if equilibria:
        print("Puntos de equilibrio encontrados:")
        for eq in equilibria:
            print(eq)
        results = analyze_equilibria(equilibria, f_sym, g_sym)
        display_results(results)
    else:
        print("No se encontraron puntos de equilibrio específicos.")
        results = []
    f_num, g_num = lambdify_functions(x, y, f_sym, g_sym)
    run_system(f_num, g_num, equilibria)


def compute_jacobian_symbolic(f_sym, g_sym, variables):
    import re
    def remove_spaces_around_operators(expr_str):
        expr_str = re.sub(r'\s*([\+\-\*/\^])\s*', r'\1', expr_str)
        return expr_str

    jacobian_matrix = Matrix([[simplify(f_sym.diff(var)) for var in variables],
                              [simplify(g_sym.diff(var)) for var in variables]])
    print("Matriz Jacobiana del Sistema:")
    matrix_list = jacobian_matrix.tolist()
    processed_matrix_list = []
    for row in matrix_list:
        processed_row = []
        for expr in row:
            expr_str = str(expr).replace('**', '^')
            expr_str = remove_spaces_around_operators(expr_str)
            processed_row.append(expr_str)
        processed_matrix_list.append(processed_row)
    col_widths = [max(len(row[i]) for row in processed_matrix_list) for i in range(len(variables))]
    for row in processed_matrix_list:
        row_str = ['{0:>{1}}'.format(row[i], col_widths[i]) for i in range(len(variables))]
        print('[ ' + '  '.join(row_str) + ' ]')
    print()


def lambdify_functions(x_sym, y_sym, f_sym, g_sym):
    f = lambdify((x_sym, y_sym), f_sym, modules='numpy')
    g = lambdify((x_sym, y_sym), g_sym, modules='numpy')
    return f, g


def vectorize_functions(f, g):
    f_vectorized = np.vectorize(f)
    g_vectorized = np.vectorize(g)
    return f_vectorized, g_vectorized


def find_equilibria_symbolic(f_sym, g_sym):
    if f_sym == 0 and g_sym == 0:
        print("El sistema es trivial; cualquier punto es un punto de equilibrio.")
        return []
    elif f_sym == 0:
        solutions = solve(g_sym, (x, y), dict=True)
        if not solutions:
            return [("x", "y")]
    elif g_sym == 0:
        solutions = solve(f_sym, (x, y), dict=True)
        if not solutions:
            return [("x", "y")]
    else:
        solutions = solve([f_sym, g_sym], (x, y), dict=True)
    equilibria = []
    if solutions:
        for sol in solutions:
            eq_x = sol.get(x, x)
            eq_y = sol.get(y, y)
            eq = (simplify(eq_x), simplify(eq_y))
            equilibria.append(eq)
    else:
        equilibria.append(("x", "y"))
    return equilibria


def analyze_equilibria(equilibria, f_sym, g_sym):
    results = []
    for eq in equilibria:
        if eq[0].free_symbols or eq[1].free_symbols:
            print(f"\nEl equilibrio {eq} es continuo y no se analizará individualmente.")
            continue
        J = compute_jacobian_at_equilibrium(f_sym, g_sym, eq)
        eigenvects = J.eigenvects()
        results.append({
            "equilibrium": eq,
            "jacobian": J,
            "eigenvects": eigenvects
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
    import re
    def remove_spaces_around_operators(expr_str):
        expr_str = re.sub(r'\s*([\+\-\*/\^])\s*', r'\1', expr_str)
        return expr_str

    for result in results:
        eq = result["equilibrium"]
        J = result["jacobian"]
        eigenvects = result["eigenvects"]
        print("\nPunto de equilibrio:")
        print(eq)
        print("Matriz Jacobiana en el equilibrio:")
        J_list = J.tolist()
        processed_J_list = []
        for row in J_list:
            processed_row = []
            for expr in row:
                expr_str = str(expr).replace('**', '^')
                expr_str = remove_spaces_around_operators(expr_str)
                processed_row.append(expr_str)
            processed_J_list.append(processed_row)
        col_widths = [max(len(row[i]) for row in processed_J_list) for i in range(len(processed_J_list[0]))]
        for row in processed_J_list:
            row_str = ['{0:>{1}}'.format(row[i], col_widths[i]) for i in range(len(row))]
            print('[ ' + '  '.join(row_str) + ' ]')
        print("Valores propios:")
        for ev, mult, vects in eigenvects:
            ev_simplified = simplify(ev)
            print(f"λ = {ev_simplified}")
        eigenvalues = [ev for ev, mult, vects in eigenvects]
        classification = classify_equilibrium(eigenvalues)
        print("\nClasificación del sistema:", classification)


def plot_phase_portrait(f_vectorized, g_vectorized, equilibria, x_range=(-3, 3), y_range=(-3, 3), density=1.5):
    x_vals = np.linspace(float(x_range[0]), float(x_range[1]), 400)
    y_vals = np.linspace(float(y_range[0]), float(y_range[1]), 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y)
    V = g_vectorized(X, Y)
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)

    added_label = False  # Bandera para controlar etiquetas en la leyenda

    for eq in equilibria:
        if eq[0].free_symbols or eq[1].free_symbols:
            if eq[0] == x and eq[1] != y:
                if not added_label:
                    plt.axhline(y=eq[1], color='red', linestyle='--', label='Línea de equilibrio')
                    added_label = True
                else:
                    plt.axhline(y=eq[1], color='red', linestyle='--')
            elif eq[1] == y and eq[0] != x:
                if not added_label:
                    plt.axvline(x=eq[0], color='red', linestyle='--', label='Línea de equilibrio')
                    added_label = True
                else:
                    plt.axvline(x=eq[0], color='red', linestyle='--')
        else:
            try:
                eq_numeric = (float(eq[0].evalf()), float(eq[1].evalf()))
                plt.scatter(eq_numeric[0], eq_numeric[1], color="red", s=100, label="Punto de equilibrio")
                plt.text(eq_numeric[0], eq_numeric[1], f'({eq_numeric[0]:.2f}, {eq_numeric[1]:.2f})', color='black',
                         fontsize=9)
            except (TypeError, ValueError):
                continue

    plt.title("Diagrama de Fase del Sistema No Lineal")
    plt.xlabel("x")
    plt.ylabel("y")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()



def run_system(f, g, equilibria):
    f_vectorized, g_vectorized = vectorize_functions(f, g)
    plot_phase_portrait(f_vectorized, g_vectorized, equilibria)


def main():
    x_function = 0
    y_function = y
    process_system(x_function, y_function)


if __name__ == "__main__":
    main()
