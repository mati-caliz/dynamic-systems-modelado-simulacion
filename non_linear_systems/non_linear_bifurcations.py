import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import fsolve
from sympy import symbols, Matrix, lambdify

def define_symbols():
    return symbols('x y')

def define_equations(x, y, mu):
    x_function = y - x**2 - mu
    y_function = x**2 - y**2
    return x_function, y_function

def compute_jacobian_symbolic(f_sym, g_sym, variables):
    jacobian_matrix = Matrix([[f_sym.diff(var) for var in variables],
                              [g_sym.diff(var) for var in variables]])
    print("Matriz Jacobiana simbólica del sistema:")
    for row in jacobian_matrix.tolist():
        print("\t".join([str(elem) for elem in row]))
    print()
    return jacobian_matrix

def lambdify_functions(x_sym, y_sym, f_sym, g_sym):
    f = lambdify((x_sym, y_sym, 'mu'), f_sym, modules='numpy')
    g = lambdify((x_sym, y_sym, 'mu'), g_sym, modules='numpy')
    return f, g

def compute_jacobian_numeric(f, g, x, y, mu, h=1e-5):
    df_dx = (f(x + h, y, mu) - f(x - h, y, mu)) / (2 * h)
    df_dy = (f(x, y + h, mu) - f(x, y - h, mu)) / (2 * h)
    dg_dx = (g(x + h, y, mu) - g(x - h, y, mu)) / (2 * h)
    dg_dy = (g(x, y + h, mu) - g(x, y - h, mu)) / (2 * h)
    return np.array([[df_dx, df_dy],
                     [dg_dx, dg_dy]])

def find_equilibria(f, g, mu, initial_guesses, tolerance=1e-5):
    equilibria = []
    for guess in initial_guesses:
        try:
            equilibrium, info, ier, mesg = fsolve(
                lambda vars: [f(vars[0], vars[1], mu), g(vars[0], vars[1], mu)],
                guess,
                full_output=True
            )
            if ier == 1:
                if not any(np.allclose(equilibrium, eq, atol=tolerance) for eq in equilibria):
                    equilibria.append(equilibrium)
            else:
                print(f"fsolve no convergió para la suposición inicial {guess}: {mesg}")
        except Exception as e:
            print(f"Error al resolver para la suposición inicial {guess}: {e}")
    return equilibria

def analyze_equilibria(equilibria, f, g, mu):
    results = []
    for eq in equilibria:
        J = compute_jacobian_numeric(f, g, eq[0], eq[1], mu)
        eigenvalues, eigenvectors = eig(J)
        results.append({
            "equilibrium": eq,
            "jacobian": J,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        })
    return results

def format_eigenvalue(eigenvalue, decimals=4):
    real = np.round(eigenvalue.real, decimals)
    imag = np.round(eigenvalue.imag, decimals)
    if np.abs(imag) < 1e-10:
        return f"{real}"
    elif real == 0:
        return f"{imag}i"
    else:
        sign = '+' if imag > 0 else '-'
        return f"{real} {sign} {np.abs(imag)}i"

def classify_equilibrium(eigenvalues, tol=1e-10):
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    real_parts[np.abs(real_parts) < tol] = 0
    imag_parts[np.abs(imag_parts) < tol] = 0

    if np.all(imag_parts != 0):
        if np.all(real_parts == 0):
            return "Centro (valores propios puramente imaginarios)"
        elif np.all(real_parts < 0):
            return "Foco estable (valores propios complejos con partes reales negativas)"
        elif np.all(real_parts > 0):
            return "Foco inestable (valores propios complejos con partes reales positivas)"
        else:
            return "Espiral de silla (valores propios complejos con partes reales mixtas)"
    else:
        if np.all(real_parts > 0):
            return "Nodo inestable (todos los valores propios reales positivos)"
        elif np.all(real_parts < 0):
            return "Nodo estable (todos los valores propios reales negativos)"
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            return "Punto silla (valores propios reales con signos opuestos)"
        elif np.all(real_parts == 0):
            return "Centro o nodo degenerado (todos los valores propios reales cero)"
        else:
            return "Otro tipo de equilibrio"

def display_results(results, mu):
    for result in results:
        eq = result["equilibrium"]
        J = result["jacobian"]
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]

        print(f"\nPunto de equilibrio: {np.round(eq, 5)} para μ = {mu}")
        print("Matriz Jacobiana:\n", np.round(J, 5))
        print()
        formatted_eigenvalues = [format_eigenvalue(ev) for ev in eigenvalues]
        print("Valores propios:", formatted_eigenvalues)
        print("Vectores propios:\n", np.round(eigenvectors, 5))

        classification = classify_equilibrium(eigenvalues)
        print("\nClasificación del sistema:", classification)

def plot_phase_portrait(f, g, results, mu, x_range=(-5, 5), y_range=(-5, 5), density=1.5):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f(X, Y, mu)
    V = g(X, Y, mu)

    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)

    if results:
        eq_points = np.array([result["equilibrium"] for result in results])
        plt.scatter(eq_points[:, 0], eq_points[:, 1], color="red", s=100, label="Puntos de Equilibrio")
        for eq in eq_points:
            plt.text(eq[0], eq[1], f'({eq[0]:.2f}, {eq[1]:.2f})', color='black', fontsize=9)

    plt.title(f"Retrato de fases para μ = {mu}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()

def plot_bifurcation(mu_range, bifurcation_data, title="Diagrama de Bifurcación"):
    plt.figure(figsize=(12, 8))
    classifications = {}

    for data in bifurcation_data:
        mu = data["mu"]
        for eq_info in data["equilibria"]:
            eq = eq_info["equilibrium"]
            classification = classify_equilibrium(eq_info["eigenvalues"])
            x, y = eq
            if classification not in classifications:
                classifications[classification] = {"x": [], "y": [], "mu": []}
            classifications[classification]["x"].append(x)
            classifications[classification]["y"].append(y)
            classifications[classification]["mu"].append(mu)

    color_map = {
        "Nodo estable (todos los valores propios reales negativos)": "green",
        "Nodo inestable (todos los valores propios reales positivos)": "red",
        "Punto silla (valores propios reales con signos opuestos)": "black",
        "Foco estable (valores propios complejos con partes reales negativas)": "blue",
        "Foco inestable (valores propios complejos con partes reales positivas)": "orange",
        "Centro (valores propios puramente imaginarios)": "purple",
        "Espiral de silla (valores propios complejos con partes reales mixtas)": "brown",
        "Otro tipo de equilibrio": "gray"
    }

    for classification, data in classifications.items():
        plt.scatter(data["mu"], data["x"], color=color_map.get(classification, "gray"), label=classification, alpha=0.6)
        plt.scatter(data["mu"], data["y"], color=color_map.get(classification, "gray"), alpha=0.6)

    plt.xlabel("μ")
    plt.ylabel("Coordenadas de equilibrio (x e y)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def bifurcation_analysis(f, g, mu_range, initial_guesses):
    bifurcation_data = []

    for mu in mu_range:
        print(f"\nAnalizando para μ = {mu}")
        equilibria = find_equilibria(f, g, mu, initial_guesses)
        print(f"Encontrado(s) {len(equilibria)} punto(s) de equilibrio.")
        if not equilibria:
            print(f"No se encontraron puntos de equilibrio para μ = {mu}")
            continue
        results = analyze_equilibria(equilibria, f, g, mu)
        bifurcation_data.append({
            "mu": mu,
            "equilibria": results
        })
        display_results(results, mu)

    return bifurcation_data

def run_system(f, g, mu, initial_guesses):
    equilibria = find_equilibria(f, g, mu, initial_guesses)
    if not equilibria:
        print(f"No se encontraron puntos de equilibrio para μ = {mu}.")
        return

    results = analyze_equilibria(equilibria, f, g, mu)
    display_results(results, mu)
    plot_phase_portrait(f, g, results, mu)

def main():
    print("=== Análisis de sistema no lineal con parámetro μ ===")
    x_sym, y_sym = define_symbols()

    initial_guesses = [
        [0, 0],
        [2, 0],
        [0, 3],
        [2, 3],
        [1, 1.5],
        [1, 1],
        [3, 3],
        [-1, -1],
        [1.5, 1.5],
        [2, 2]
    ]

    choice = input("¿Desea analizar un μ específico o realizar un análisis de bifurcación? (Ingrese 'specific' o 'bifurcation'): ").strip().lower()

    if choice == 'specific':
        try:
            mu = float(input("Ingrese el valor de μ: "))
        except ValueError:
            print("Entrada inválida para μ.")
            return

        f_sym, g_sym = define_equations(x_sym, y_sym, mu)
        compute_jacobian_symbolic(f_sym, g_sym, (x_sym, y_sym))
        f, g = lambdify_functions(x_sym, y_sym, f_sym, g_sym)

        run_system(f, g, mu, initial_guesses)

    elif choice == 'bifurcation':
        try:
            mu_min = float(input("Ingrese el valor mínimo de μ: "))
            mu_max = float(input("Ingrese el valor máximo de μ: "))
            num_mu = int(input("Ingrese el número de valores de μ a analizar: "))
            if num_mu <= 0:
                raise ValueError("El número de valores de μ debe ser un entero positivo.")
        except ValueError as e:
            print(f"Entrada inválida: {e}")
            return

        mu_range = np.linspace(mu_min, mu_max, num_mu)

        f_sym, g_sym = define_equations(x_sym, y_sym, symbols('mu'))
        compute_jacobian_symbolic(f_sym, g_sym, (x_sym, y_sym))
        f, g = lambdify_functions(x_sym, y_sym, f_sym, g_sym)

        bifurcation_data = bifurcation_analysis(f, g, mu_range, initial_guesses)

        plot_bifurcation(mu_range, bifurcation_data)

    else:
        print("Elección inválida. Por favor, ingrese 'specific' o 'bifurcation'.")

if __name__ == "__main__":
    main()
