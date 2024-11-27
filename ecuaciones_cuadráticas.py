import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import fsolve
from sympy import symbols, Matrix, lambdify

# Definir símbolos
x_sym, y_sym = symbols('x y')

# Definir las funciones simbólicas (modifica aquí tus ecuaciones)
# Por ejemplo, para x' = x*(2 - x), y' = y*(3 - y)
f_sym = y_sym - x_sym ** 2 - 2
g_sym = x_sym ** 2 - x_sym * y_sym

# Convertir funciones simbólicas en funciones numéricas
f = lambdify((x_sym, y_sym), f_sym, modules='numpy')
g = lambdify((x_sym, y_sym), g_sym, modules='numpy')

# Vectorizar las funciones para manejar arreglos de Numpy
f_vectorized = np.vectorize(f)
g_vectorized = np.vectorize(g)

# Crear la matriz Jacobiana simbólicamente
J_sym = Matrix([f_sym, g_sym]).jacobian([x_sym, y_sym])

# Lambdificar la Jacobiana para convertirla en una función numérica
jacobian_func = lambdify((x_sym, y_sym), J_sym, modules='numpy')

# Función para encontrar múltiples puntos de equilibrio
def find_equilibria(guesses):
    equilibria = []
    for guess in guesses:
        equilibrium, info, ier, mesg = fsolve(
            lambda vars: [f(vars[0], vars[1]), g(vars[0], vars[1])],
            guess,
            full_output=True
        )
        if ier == 1:
            if not any(np.allclose(equilibrium, eq, atol=1e-5) for eq in equilibria):
                equilibria.append(equilibrium)
        else:
            print(f"fsolve no convergió para la aproximación inicial {guess}")
    return equilibria

# Buscar puntos de equilibrio
initial_guesses = [[0, 0], [2, 0], [0, 3], [2, 3], [1, 1.5]]
equilibria = find_equilibria(initial_guesses)
print("Puntos de equilibrio encontrados:")
for eq in equilibria:
    print(np.round(eq, 5))

# Evaluar la Jacobiana, valores propios y vectores propios para cada punto
results = []
for eq in equilibria:
    J = jacobian_func(eq[0], eq[1])
    eigenvalues, eigenvectors = eig(J)
    results.append({
        "equilibrium": eq,
        "jacobian": J,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors
    })

# Mostrar resultados
for result in results:
    eq = result["equilibrium"]
    print(f"\nPunto de equilibrio: {np.round(eq, 5)}")
    print("Matriz Jacobiana:\n", result["jacobian"])
    print("Valores propios:", result["eigenvalues"])
    print("Vectores propios (columnas):\n", result["eigenvectors"])

# Graficar el diagrama de fase con puntos de equilibrio
def plot_phase_portrait():
    x_vals = np.linspace(-1, 3, 400)
    y_vals = np.linspace(-1, 4, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = f_vectorized(X, Y)
    V = g_vectorized(X, Y)  # Asegúrate de que las funciones dependan de X e Y según corresponda

    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, color="blue", density=1.5, linewidth=1, arrowsize=1)

    # Resaltar puntos de equilibrio
    eq_points = np.array([result["equilibrium"] for result in results])
    plt.scatter(eq_points[:, 0], eq_points[:, 1], color="red", s=100, label="Puntos de equilibrio")

    plt.title("Diagrama de Fase del Sistema No Lineal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_phase_portrait()
