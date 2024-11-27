import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig


def calculate_equilibrium_point(A, B=None):
    if B is None:
        return np.zeros(A.shape[0])
    try:
        return np.linalg.solve(-A, B)
    except np.linalg.LinAlgError:
        return np.dot(np.linalg.pinv(-A), B)


def classify_system(eigenvalues, tol=1e-10):
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    # Tratar partes reales muy pequeñas como cero
    real_parts[np.abs(real_parts) < tol] = 0

    if np.all(imag_parts != 0):
        if np.all(real_parts == 0):
            return "Centro (valores propios puramente imaginarios)"
        elif np.all(real_parts < 0):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif np.all(real_parts > 0):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
        else:
            return "Caso Indeterminado (valores propios complejos con partes reales de signos opuestos)"
    else:
        if np.all(real_parts > 0):
            return "Nodo Inestable (valores propios reales y positivos)"
        elif np.all(real_parts < 0):
            return "Nodo Estable (valores propios reales y negativos)"
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            return "Punto Silla (valores propios reales de signos opuestos)"
        else:
            return "Valores propios reales nulos o repetidos"


def format_complex(complex_number, decimals=2, tolerance=1e-10):
    real = 0 if abs(complex_number.real) < tolerance else round(complex_number.real, decimals)
    imaginary = 0 if abs(complex_number.imag) < tolerance else round(complex_number.imag, decimals)
    if imaginary == 0:
        return f"{real}"
    elif real == 0:
        return f"{imaginary}i"
    elif imaginary > 0:
        return f"{real}+{imaginary}i"
    else:
        return f"{real}{imaginary}i"


def scale_eigenvector(vector):
    if vector[0] != 0:
        return vector / vector[0]
    else:
        return vector


def print_general_equation(eigenvalues, eigenvectors, tol=1e-10, decimals=2):
    print("\nEcuación general combinada del sistema:")
    if np.all(np.iscomplex(eigenvalues)):
        lambda_real = np.real(eigenvalues[0])
        lambda_imag = np.imag(eigenvalues[0])
        vector = eigenvectors[:, 0]
        c1, c2 = 'C1', 'C2'
        x_eq = f"({format_complex(vector[0].real)} * {c1} - {format_complex(vector[0].imag)} * {c2}) * cos({round(lambda_imag, decimals)}t) " \
               f"- ({format_complex(vector[0].imag)} * {c1} + {format_complex(vector[0].real)} * {c2}) * sin({round(lambda_imag, decimals)}t)"
        y_eq = f"({format_complex(vector[1].real)} * {c1} - {format_complex(vector[1].imag)} * {c2}) * cos({round(lambda_imag, decimals)}t) " \
               f"- ({format_complex(vector[1].imag)} * {c1} + {format_complex(vector[1].real)} * {c2}) * sin({round(lambda_imag, decimals)}t)"
        print(f"x(t) = {x_eq}")
        print(f"y(t) = {y_eq}")
    else:
        # Manejo para valores propios reales
        x_eq_terms = []
        y_eq_terms = []
        for i, eigenvalue in enumerate(eigenvalues):
            vector = eigenvectors[:, i].real
            coef = f"C{i + 1}"
            term_x = f"{coef} * {round(vector[0], decimals)} * e^({round(eigenvalue.real, decimals)}t)"
            term_y = f"{coef} * {round(vector[1], decimals)} * e^({round(eigenvalue.real, decimals)}t)"
            x_eq_terms.append(term_x)
            y_eq_terms.append(term_y)
        x_eq = " + ".join(x_eq_terms)
        y_eq = " + ".join(y_eq_terms)
        print(f"x(t) = {x_eq}")
        print(f"y(t) = {y_eq}")


def plot_extended_vectors(ax, equilibrium_point, V1, V2, axis_limit):
    equilibrium_point = np.array(equilibrium_point)
    t = np.linspace(-axis_limit, axis_limit, 100)
    V1_real = V1.real
    V1_imag = V1.imag
    V1_real_line = equilibrium_point[:, None] + V1_real[:, None] * t
    V1_imag_line = equilibrium_point[:, None] + V1_imag[:, None] * t
    ax.plot(V1_real_line[0], V1_real_line[1], 'r-',
            label=f"V1 real {format_complex(V1_real[0])}, {format_complex(V1_real[1])}")
    ax.plot(V1_imag_line[0], V1_imag_line[1], 'r--',
            label=f"V1 imag {format_complex(V1_imag[0])}, {format_complex(V1_imag[1])}")
    V2_real = V2.real
    V2_imag = V2.imag
    V2_real_line = equilibrium_point[:, None] + V2_real[:, None] * t
    V2_imag_line = equilibrium_point[:, None] + V2_imag[:, None] * t
    ax.plot(V2_real_line[0], V2_real_line[1], 'g-',
            label=f"V2 real {format_complex(V2_real[0])}, {format_complex(V2_real[1])}")
    ax.plot(V2_imag_line[0], V2_imag_line[1], 'g--',
            label=f"V2 imag {format_complex(V2_imag[0])}, {format_complex(V2_imag[1])}")


def plot_phase_portrait(A, V1, V2, equilibrium_point, B=None):
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
    plot_extended_vectors(ax, equilibrium_point, V1, V2, axis_limit)

    ax.set_title("Diagrama de Fase del Sistema Lineal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_xlim(equilibrium_point[0] - axis_limit, equilibrium_point[0] + axis_limit)
    ax.set_ylim(equilibrium_point[1] - axis_limit, equilibrium_point[1] + axis_limit)
    plt.grid()
    plt.show()


def validate_inputs(A, B):
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada.")
    if B is not None and B.shape[0] != A.shape[0]:
        raise ValueError("El vector B debe tener la misma cantidad de filas que A.")


def display_eigen_info(eigenvalues, eigenvectors):
    print("Valores propios:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"\nλ{i + 1} = {format_complex(eigenvalue)}")
        vector = eigenvectors[:, i]
        scaled_vector = scale_eigenvector(vector)
        formatted_vector = [format_complex(component) for component in scaled_vector]
        print(f"Vector propio V{i + 1}: {formatted_vector}")


def process_system(A, B):
    validate_inputs(A, B)
    equilibrium_point = calculate_equilibrium_point(A, B)
    eigenvalues, eigenvectors = eig(A)
    V1 = scale_eigenvector(eigenvectors[:, 0])
    V2 = scale_eigenvector(eigenvectors[:, 1])
    display_eigen_info(eigenvalues, eigenvectors)
    system_type = classify_system(eigenvalues)
    print("\nTipo de sistema:", system_type)
    print("Punto de equilibrio:", equilibrium_point)
    print_general_equation(eigenvalues, eigenvectors)
    plot_phase_portrait(A, V1, V2, equilibrium_point, B)


def main():
    # Definir la matriz A
    A = np.array([[3, -2],
                  [4, -1]])
    # Definir el vector B (si no existe dejar None)
    B = None
    # B = np.array([-5, -7]) o B = None
    process_system(A, B)


if __name__ == "__main__":
    main()
