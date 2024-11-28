import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Symbol, Expr, Matrix, lambdify, solve, simplify, S, sympify, pprint

x, y, a = symbols('x y a')

def process_system(f_sym, g_sym, parameters=None):
    if parameters is None:
        parameters = set()
    else:
        parameters = set(parameters)

    def ensure_sympy_expression(func):
        return S(func) if isinstance(func, (int, float)) else sympify(func)

    f_sym = ensure_sympy_expression(f_sym)
    g_sym = ensure_sympy_expression(g_sym)
    compute_jacobian_symbolic(f_sym, g_sym, (x, y))
    equilibria = find_equilibria_symbolic(f_sym, g_sym, parameters)
    if equilibria:
        print("Puntos de equilibrio encontrados:")
        for eq in equilibria:
            print(eq)
        results = analyze_equilibria(equilibria, f_sym, g_sym, parameters)
        display_results(results)
    else:
        print("No se encontraron puntos de equilibrio.")
        results = []
    f_num, g_num = lambdify_functions(x, y, f_sym, g_sym, parameters)
    run_system(f_num, g_num, equilibria, parameters)

    # Generar el diagrama de bifurcación si hay parámetros
    if parameters:
        # Podemos tomar el primer parámetro del conjunto
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
    solutions = solve([f_sym, g_sym], variables, dict=True)
    equilibria = []
    if solutions:
        for sol in solutions:
            eq_x = sol.get(x, x)
            eq_y = sol.get(y, y)
            equilibria.append((simplify(eq_x), simplify(eq_y)))
    return equilibria

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
    for result in results:
        eq = result["equilibrium"]
        J = result["jacobian"]
        eigenvects = result["eigenvects"]
        print("\nPunto de equilibrio:")
        print(eq)
        print("Matriz Jacobiana en el equilibrio:")
        pprint(J)
        print("Valores propios:")
        for ev, mult, vects in eigenvects:
            ev_simplified = simplify(ev)
            print(f"λ = {ev_simplified}")
        eigenvalues = [ev for ev, mult, vects in eigenvects]
        classification = classify_equilibrium(eigenvalues)
        print("\nClasificación del sistema:", classification)

def plot_phase_portrait(f_vectorized, g_vectorized, equilibria, parameters, x_range=(-3, 3), y_range=(-3, 3), density=1.5):
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
    added_label = False
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

def run_system(f, g, equilibria, parameters):
    f_vectorized, g_vectorized = vectorize_functions(f, g)
    plot_phase_portrait(f_vectorized, g_vectorized, equilibria, parameters)

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
    x_function = 3/2 * x + 1/2 * y
    y_function = 1/2 * x + 3/2 * y
    process_system(x_function, y_function)

if __name__ == "__main__":
    main()
