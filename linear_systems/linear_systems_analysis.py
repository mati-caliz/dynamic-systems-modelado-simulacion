import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, simplify, exp, re, im, I, cos, sin, N
from sympy.abc import t

def calculate_equilibrium_point(A, B=None):
    if B is None:
        n = A.shape[0]
        return Matrix([0]*n)
    else:
        equilibrium_point = -A.inv() * B
        return equilibrium_point

def classify_system(eigenvalues):
    real_parts = [re(ev) for ev in eigenvalues]
    imag_parts = [im(ev) for ev in eigenvalues]
    if all(imag != 0 for imag in imag_parts):
        if all(real == 0 for real in real_parts):
            return "Centro (valores propios puramente imaginarios)"
        elif all(real < 0 for real in real_parts):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif all(real > 0 for real in real_parts):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
    else:
        if all(real > 0 for real in real_parts):
            return "Nodo Inestable (valores propios reales positivos)"
        elif all(real < 0 for real in real_parts):
            return "Nodo Estable (valores propios reales negativos)"
        elif any(real > 0 for real in real_parts) and any(real < 0 for real in real_parts):
            return "Punto Silla (valores propios reales de signos opuestos)"
        elif all(real == 0 for real in real_parts):
            return "Nodo Degenerado (valores propios reales repetidos)"
    return "Caso no clasificado"

def format_complex_sympy(complex_number):
    s = str(simplify(complex_number))
    s = s.replace('I', 'i')
    return s

def scale_eigenvector(vector):
    if vector[0] != 0:
        return vector / vector[0]
    else:
        return vector

def print_general_equation(eigenvalues, eigenvectors):
    print("\nEcuación general combinada del sistema:")
    C = symbols('C1:%d' % (len(eigenvalues)+1))
    x_expr = 0
    y_expr = 0
    i = 0
    while i < len(eigenvalues):
        ev = eigenvalues[i]
        if ev.is_real:
            vect = eigenvectors[i]
            x_expr += C[i] * vect[0] * exp(ev * t)
            y_expr += C[i] * vect[1] * exp(ev * t)
            i += 1
        else:
            alpha = re(ev)
            beta = im(ev)
            vect = eigenvectors[i]
            p = Matrix([re(vect[0]), re(vect[1])])
            q = Matrix([im(vect[0]), im(vect[1])])
            x_expr += exp(alpha * t) * ((C[i]*p[0] + C[i+1]*q[0]) * cos(beta*t) - (C[i]*q[0] - C[i+1]*p[0]) * sin(beta*t))
            y_expr += exp(alpha * t) * ((C[i]*p[1] + C[i+1]*q[1]) * cos(beta*t) - (C[i]*q[1] - C[i+1]*p[1]) * sin(beta*t))
            i += 2
    print(f"x(t) = {simplify(x_expr)}")
    print(f"y(t) = {simplify(y_expr)}")

def plot_phase_portrait(A, eigenvectors, equilibrium_point):
    A_num = np.array(A.evalf(), dtype=np.float64)
    equilibrium_point_num = np.array(equilibrium_point.evalf(), dtype=np.float64).flatten()
    x_min, x_max = equilibrium_point_num[0] - 5, equilibrium_point_num[0] + 5
    y_min, y_max = equilibrium_point_num[1] - 5, equilibrium_point_num[1] + 5
    x_vals = np.linspace(x_min, x_max, 20)
    y_vals = np.linspace(y_min, y_max, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = A_num[0, 0] * (X - equilibrium_point_num[0]) + A_num[0, 1] * (Y - equilibrium_point_num[1])
    V_dir = A_num[1, 0] * (X - equilibrium_point_num[0]) + A_num[1, 1] * (Y - equilibrium_point_num[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.streamplot(X, Y, U, V_dir, color='b', density=1.5, linewidth=0.8)

    t_line = np.linspace(-10, 10, 100)
    for i, vect in enumerate(eigenvectors):
        vect = scale_eigenvector(vect)
        vect_re = np.array([re(v).evalf() for v in vect], dtype=np.float64)
        vect_im = np.array([im(v).evalf() for v in vect], dtype=np.float64)
        if not np.allclose(vect_re, 0):
            line_re = equilibrium_point_num[:, None] + vect_re[:, None] * t_line
            label_re = f"Re(V{i+1}) ({vect_re[0]:.2f}, {vect_re[1]:.2f})"
            ax.plot(line_re[0], line_re[1], label=label_re)
        if not np.allclose(vect_im, 0):
            line_im = equilibrium_point_num[:, None] + vect_im[:, None] * t_line
            label_im = f"Im(V{i+1}) ({vect_im[0]:.2f}, {vect_im[1]:.2f})"
            ax.plot(line_im[0], line_im[1], label=label_im, linestyle='--')

    ax.set_title("Diagrama de Fase del Sistema Lineal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
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
        eigenvalue_str = format_complex_sympy(eigenvalue)
        print(f"\nλ{i + 1} = {eigenvalue_str}")
        vector = eigenvectors[i]
        scaled_vector = scale_eigenvector(vector)
        formatted_vector = [format_complex_sympy(component) for component in scaled_vector]
        vector_str = '(' + ', '.join(formatted_vector) + ')'
        print(f"Vector propio V{i + 1}: {vector_str}")

def process_system(A, B):
    validate_inputs(A, B)
    equilibrium_point = calculate_equilibrium_point(A, B)
    eigenvects = A.eigenvects()
    eigenvalues = []
    eigenvectors = []
    for ev, mult, vects in eigenvects:
        for vect in vects:
            eigenvalues.append(ev)
            eigenvectors.append(vect)
    display_eigen_info(eigenvalues, eigenvectors)
    system_type = classify_system(eigenvalues)
    print("\nTipo de sistema:", system_type)
    equilibrium_point_list = [format_complex_sympy(coord) for coord in equilibrium_point]
    equilibrium_point_str = '(' + ', '.join(equilibrium_point_list) + ')'
    print("Punto de equilibrio:", equilibrium_point_str)
    print_general_equation(eigenvalues, eigenvectors)
    plot_phase_portrait(A, eigenvectors, equilibrium_point)

def main():
    A = Matrix([[1, -2],
                [-2, 1]])
    B = None
    process_system(A, B)

if __name__ == "__main__":
    main()
