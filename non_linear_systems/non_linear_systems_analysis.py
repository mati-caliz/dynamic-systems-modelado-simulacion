import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import fsolve
from sympy import symbols, Matrix, lambdify

x, y = symbols('x y')

def process_system(f_sym, g_sym):
    compute_jacobian_symbolic(f_sym, g_sym, (x, y))
    f, g = lambdify_functions(x, y, f_sym, g_sym)
    run_system(f, g)

def compute_jacobian_symbolic(f_sym, g_sym, variables):
    jacobian_matrix = Matrix([[f_sym.diff(var) for var in variables],
                              [g_sym.diff(var) for var in variables]])
    print("Matriz Jacobiana Simbólica del Sistema:")
    for i in range(jacobian_matrix.shape[0]):
        for j in range(jacobian_matrix.shape[1]):
            print(jacobian_matrix[i, j], end="\t")
        print()
    print()

def lambdify_functions(x_sym, y_sym, f_sym, g_sym):
    f = lambdify((x_sym, y_sym), f_sym, modules='numpy')
    g = lambdify((x_sym, y_sym), g_sym, modules='numpy')
    return f, g

def vectorize_functions(f, g):
    f_vectorized = np.vectorize(f)
    g_vectorized = np.vectorize(g)
    return f_vectorized, g_vectorized

def compute_jacobian_numeric(f, g, x_val, y_val, h=1e-5):
    df_dx = (f(x_val + h, y_val) - f(x_val - h, y_val)) / (2 * h)
    df_dy = (f(x_val, y_val + h) - f(x_val, y_val - h)) / (2 * h)
    dg_dx = (g(x_val + h, y_val) - g(x_val - h, y_val)) / (2 * h)
    dg_dy = (g(x_val, y_val + h) - g(x_val, y_val - h)) / (2 * h)
    return np.array([[df_dx, df_dy],
                     [dg_dx, dg_dy]])

def find_equilibria(f, g, initial_guesses, tolerance=1e-5):
    equilibria = []
    for guess in initial_guesses:
        try:
            equilibrium, info, ier, mesg = fsolve(
                lambda vars: [f(vars[0], vars[1]), g(vars[0], vars[1])],
                guess,
                full_output=True
            )
            if ier == 1:
                if not any(np.allclose(equilibrium, eq, atol=tolerance) for eq in equilibria):
                    equilibria.append(equilibrium)
            else:
                pass  # No se encontró un equilibrio
        except Exception as e:
            print(f"Error al resolver con aproximación inicial {guess}: {e}")
    return equilibria

def analyze_equilibria(equilibria, f, g):
    results = []
    for eq in equilibria:
        J = compute_jacobian_numeric(f, g, eq[0], eq[1])
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

    # Tratar partes reales e imaginarias pequeñas como cero
    real_parts[np.abs(real_parts) < tol] = 0
    imag_parts[np.abs(imag_parts) < tol] = 0

    if np.all(imag_parts != 0):
        if np.all(real_parts == 0):
            return "Centro (valores propios puramente imaginarios)"
        elif np.all(real_parts < 0):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif np.all(real_parts > 0):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
        else:
            return "Espiral Silla (valores propios complejos con partes reales de signos opuestos)"
    else:
        if np.all(real_parts > 0):
            return "Nodo Inestable (valores propios reales y positivos)"
        elif np.all(real_parts < 0):
            return "Nodo Estable (valores propios reales y negativos)"
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            return "Punto Silla (valores propios reales de signos opuestos)"
        elif np.all(real_parts == 0):
            return "Centro o Nodo Degenerado (valores propios reales nulos o repetidos)"
        else:
            return "Otro tipo de equilibrio"

def display_results(results):
    for result in results:
        eq = result["equilibrium"]
        J = result["jacobian"]
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]

        print(f"\nPunto de equilibrio: {np.round(eq, 5)}")
        print("Matriz Jacobiana:\n", np.round(J, 5))
        print()
        formatted_eigenvalues = [format_eigenvalue(ev) for ev in eigenvalues]
        print("Valores propios:", formatted_eigenvalues)
        print("Vectores propios:\n", np.round(eigenvectors, 5))

        classification = classify_equilibrium(eigenvalues)
        print("\nClasificación del sistema:", classification)

def plot_phase_portrait(f_vectorized, g_vectorized, results, x_range=(-2, 2), y_range=(-2, 2), density=1.5):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y)
    V = g_vectorized(X, Y)

    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=density, linewidth=1, arrowsize=1)

    if results:
        eq_points = np.array([result["equilibrium"] for result in results])
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

def run_system(f, g):
    f_vectorized, g_vectorized = vectorize_functions(f, g)

    initial_guesses = [
        [0, 0],
        [1, 1],
        [-1, -1],
        [1, -1],
        [-1, 1]
    ]

    equilibria = find_equilibria(f, g, initial_guesses)
    print("Puntos de equilibrio encontrados:")
    for eq in equilibria:
        print(np.round(eq, 5))
    results = analyze_equilibria(equilibria, f, g)
    display_results(results)
    plot_phase_portrait(f_vectorized, g_vectorized, results)

def main():
    x_function = x**2 + y**2 - 2
    y_function = x**2 - y**2
    process_system(x_function, y_function)

if __name__ == "__main__":
    main()
