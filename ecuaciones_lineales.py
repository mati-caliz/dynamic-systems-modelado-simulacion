import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig


def calculate_equilibrium_point(A, B=None):
    """
    Calcula el punto de equilibrio del sistema lineal.

    Parámetros:
    - A: Matriz de coeficientes del sistema.
    - B: Vector constante (opcional).

    Retorna:
    - equilibrium_point: Vector con el punto de equilibrio.
    """
    if B is None:
        return np.zeros(A.shape[0])
    try:
        return np.linalg.solve(-A, B)
    except np.linalg.LinAlgError:
        return np.dot(np.linalg.pinv(-A), B)


def simplify_vector(vector, tol=1e-5, max_scale=100):
    """
    Simplifica un vector propio intentando escalarlo a números enteros.

    Parámetros:
    - vector: Vector propio a simplificar.
    - tol: Tolerancia para considerar si un número es entero.
    - max_scale: Máximo factor de escala a intentar.

    Retorna:
    - vector simplificado con componentes enteros o normalizado.
    """
    # Intentar encontrar un factor de escala que convierta los componentes en enteros
    for scale in range(1, max_scale + 1):
        scaled = vector * scale
        if np.all(np.abs(scaled - np.round(scaled)) < tol):
            return np.round(scaled).astype(int)

    # Si no se encuentra una escala entera, normalizar por el componente de mayor magnitud
    max_index = np.argmax(np.abs(vector))
    if vector[max_index] != 0:
        vector = vector / vector[max_index]
    vector = np.round(vector, decimals=5)
    return vector


def classify_system(eigenvalues):
    """
    Clasifica el tipo de sistema según los valores propios.

    Parámetros:
    - eigenvalues: Array de valores propios.

    Retorna:
    - Tipo de sistema (str).
    """
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    if np.all(imag_parts == 0):
        if np.all(real_parts > 0):
            return "Nodo Inestable (valores propios reales y positivos)"
        elif np.all(real_parts < 0):
            return "Nodo Estable (valores propios reales y negativos)"
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            return "Punto Silla (valores propios reales de signos opuestos)"
        else:
            return "Valores propios reales nulos o repetidos"
    else:
        if np.all(real_parts == 0):
            return "Centro (valores propios puramente imaginarios)"
        elif np.all(real_parts < 0):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif np.all(real_parts > 0):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
        else:
            return "Caso Indeterminado (valores propios complejos con partes reales de signos opuestos)"


def print_general_equation(eigenvalues, eigenvectors):
    """
    Genera y muestra una única ecuación general para x(t) y y(t) combinando los valores propios y vectores propios.

    Parámetros:
    - eigenvalues: Array de valores propios.
    - eigenvectors: Matriz de vectores propios.
    """
    print("\nEcuación general combinada del sistema:")

    # Inicializar ecuaciones combinadas
    x_eq_terms = []
    y_eq_terms = []

    # Recorrer los valores y vectores propios
    for i, eigenvalue in enumerate(eigenvalues):
        vector = simplify_vector(eigenvectors[:, i].real)  # Vector propio simplificado
        coef = f"C{i + 1}"  # Constante arbitraria
        term_x = f"{coef} * {vector[0]} * e^({eigenvalue.real:.2f}t)"
        term_y = f"{coef} * {vector[1]} * e^({eigenvalue.real:.2f}t)"
        x_eq_terms.append(term_x)
        y_eq_terms.append(term_y)

    # Construir ecuación final combinada
    x_eq = " + ".join(x_eq_terms)
    y_eq = " + ".join(y_eq_terms)

    print(f"x'(t) = {x_eq}")
    print(f"y'(t) = {y_eq}")



def plot_extended_vectors(ax, equilibrium_point, V1, V2, axis_limit):
    """
    Grafica los vectores propios como líneas extendidas en el diagrama de fase.

    Parámetros:
    - ax: Axes de Matplotlib donde se graficará.
    - equilibrium_point: Punto de equilibrio.
    - V1, V2: Vectores propios simplificados.
    - axis_limit: Límite de los ejes para el gráfico.
    """
    t = np.linspace(-axis_limit, axis_limit, 100)
    V1_line = equilibrium_point[:, None] + V1[:, None] * t
    V2_line = equilibrium_point[:, None] + V2[:, None] * t

    ax.plot(V1_line[0], V1_line[1], 'r-', label=f"V1 {V1}")
    ax.plot(V2_line[0], V2_line[1], 'g-', label=f"V2 {V2}")


def plot_phase_portrait(A, V1, V2, equilibrium_point, B=None):
    """
    Grafica el diagrama de fase del sistema lineal.

    Parámetros:
    - A: Matriz de coeficientes del sistema.
    - V1, V2: Vectores propios simplificados.
    - equilibrium_point: Punto de equilibrio.
    - B: Vector constante (opcional).
    """
    eigenvalues, _ = eig(A)
    max_eigenvalue = max(np.abs(eigenvalues.real))
    axis_limit = max(3, max_eigenvalue * 2)  # Aumentar el rango para mejor visualización

    x_vals = np.linspace(-axis_limit, axis_limit, 20) + equilibrium_point[0]
    y_vals = np.linspace(-axis_limit, axis_limit, 20) + equilibrium_point[1]
    X, Y = np.meshgrid(x_vals, y_vals)

    # Ajustar el campo de direcciones
    U = A[0, 0] * (X - equilibrium_point[0]) + A[0, 1] * (Y - equilibrium_point[1])
    V = A[1, 0] * (X - equilibrium_point[0]) + A[1, 1] * (Y - equilibrium_point[1])

    fig, ax = plt.subplots()
    ax.streamplot(X, Y, U, V, color='b', density=1.5, linewidth=0.8)

    # Graficar los vectores propios como líneas extendidas
    plot_extended_vectors(ax, equilibrium_point, V1, V2, axis_limit)

    ax.set_title("Diagrama de Fase del Sistema Lineal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_xlim(equilibrium_point[0] - axis_limit, equilibrium_point[0] + axis_limit)
    ax.set_ylim(equilibrium_point[1] - axis_limit, equilibrium_point[1] + axis_limit)
    plt.grid()
    plt.show()


def main():
    """
    Función principal que ejecuta el análisis del sistema lineal.
    """
    # Definir la matriz A
    A = np.array([[-4/3, 1/3],
                  [2/3, -5/3]])

    # Definir el vector B (si no existe dejar None)
    B = None
    # B = np.array([-5, -7])

    # Validaciones
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada.")
    if B is not None and B.shape[0] != A.shape[0]:
        raise ValueError("El vector B debe tener la misma cantidad de filas que A.")

    # Calcular el punto de equilibrio
    equilibrium_point = calculate_equilibrium_point(A, B)

    # Calcular valores propios y vectores propios
    eigenvalues, eigenvectors = eig(A)

    # Simplificar los vectores propios
    V1 = simplify_vector(eigenvectors[:, 0].real)
    V2 = simplify_vector(eigenvectors[:, 1].real)

    # Mostrar valores propios y vectores propios simplificados
    print("Valores propios:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"\nλ{i + 1} = {eigenvalue.real}")
        vector = simplify_vector(eigenvectors[:, i].real)
        print(f"Vector propio simplificado V{i + 1}: {vector}")

    # Clasificar el sistema
    system_type = classify_system(eigenvalues)
    print("\nTipo de sistema:", system_type)
    print("Punto de equilibrio:", equilibrium_point)

    # Generar la ecuación general del sistema
    print_general_equation(eigenvalues, eigenvectors)

    # Graficar el diagrama de fase
    plot_phase_portrait(A, V1, V2, equilibrium_point, B)


if __name__ == "__main__":
    main()
