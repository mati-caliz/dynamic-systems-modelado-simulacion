import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, lambdify, solve, simplify

x, y = symbols('x y')

def process_system(f_sym, g_sym):
    compute_jacobian_symbolic(f_sym, g_sym, (x, y))
    equilibria = find_equilibria_symbolic(f_sym, g_sym)
    print("Puntos de equilibrio encontrados:")
    for eq in equilibria:
        print(eq)
    results = analyze_equilibria(equilibria, f_sym, g_sym)
    display_results(results)
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
    solutions = solve([f_sym, g_sym], (x, y), dict=True)
    equilibria = []
    for sol in solutions:
        eq = (simplify(sol[x]), simplify(sol[y]))
        equilibria.append(eq)
    return equilibria

def analyze_equilibria(equilibria, f_sym, g_sym):
    results = []
    for eq in equilibria:
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
    if all(im != 0 for im in imag_parts):
        if all(re == 0 for re in real_parts):
            return "Centro (valores propios puramente imaginarios)"
        elif all(re < 0 for re in real_parts):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif all(re > 0 for re in real_parts):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
        else:
            return "Espiral Silla (valores propios complejos con partes reales de signos opuestos)"
    else:
        if all(re > 0 for re in real_parts):
            return "Nodo Inestable (valores propios reales y positivos)"
        elif all(re < 0 for re in real_parts):
            return "Nodo Estable (valores propios reales y negativos)"
        elif any(re > 0 for re in real_parts) and any(re < 0 for re in real_parts):
            return "Punto Silla (valores propios reales de signos opuestos)"
        elif all(re == 0 for re in real_parts):
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

def plot_phase_portrait(f_vectorized, g_vectorized, results, x_range=(-3, 3), y_range=(-3, 3), density=1.5):
    x_vals = np.linspace(float(x_range[0]), float(x_range[1]), 400)
    y_vals = np.linspace(float(y_range[0]), float(y_range[1]), 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y)
    V = g_vectorized(X, Y)
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)
    if results:
        eq_points = np.array([[float(eq[0]), float(eq[1])] for eq in [res["equilibrium"] for res in results]])
        plt.scatter(eq_points[:, 0], eq_points[:, 1], color="red", s=100, label="Puntos de equilibrio")
        for eq in eq_points:
            plt.text(eq[0], eq[1], f'({eq[0]:.2f}, {eq[1]:.2f})', color='black', fontsize=9)
    plt.title("Diagrama de Fase del Sistema No Lineal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()

def run_system(f, g, equilibria):
    f_vectorized, g_vectorized = vectorize_functions(f, g)
    results = []
    for eq in equilibria:
        eq_numeric = (float(eq[0]), float(eq[1]))
        results.append({"equilibrium": eq_numeric})
    plot_phase_portrait(f_vectorized, g_vectorized, results)

def main():
    x_function = x ** 2 - 1
    y_function = x
    process_system(x_function, y_function)
 # TODO: Implementar que funcione cuando alguna de las dos es 0 y = 0 por ejemplo.
if __name__ == "__main__":
    main()
