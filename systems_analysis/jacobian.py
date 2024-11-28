import sympy as sp
from sympy import Matrix, simplify

def compute_jacobian_symbolic(f_sym, g_sym, variables):
    jacobian_matrix = Matrix([
        [simplify(f_sym.diff(var)) for var in variables],
        [simplify(g_sym.diff(var)) for var in variables]
    ])
    return jacobian_matrix

def compute_jacobian_at_equilibrium(f_sym, g_sym, eq):
    jacobian_matrix = compute_jacobian_symbolic(f_sym, g_sym, (sp.Symbol('x'), sp.Symbol('y')))
    J_at_eq = jacobian_matrix.subs({sp.Symbol('x'): eq[0], sp.Symbol('y'): eq[1]})
    return J_at_eq
