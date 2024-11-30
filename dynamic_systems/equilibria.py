from sympy import symbols, solve, simplify, nsimplify, Expr
import numpy as np  # Importar numpy
from jacobian import compute_jacobian_at_equilibrium  # Importar la función necesaria

def find_equilibria_symbolic(f_sym, g_sym, parameters):
    x, y = symbols('x y')
    variables = (x, y)
    solutions = solve([f_sym, g_sym], variables, dict=True, rational=True)
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
        eigenvals = J.eigenvals()
        eigenvects = J.eigenvects()
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
