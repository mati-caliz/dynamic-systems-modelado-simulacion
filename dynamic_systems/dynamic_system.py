# dynamic_system.py
import sympy as sp
import numpy as np
from sympy import symbols, Matrix, lambdify, solve, expand, pprint, nsimplify, exp
from plotting import plot_phase_portrait, display_nullclines
from bifurcation import generate_bifurcation_diagram
from utils import ensure_sympy_expression, classify_equilibrium, format_eigenvalue
from jacobian import compute_jacobian_symbolic
from equilibria import find_equilibria_symbolic, analyze_equilibria


class DynamicSystem:
    def __init__(self, f_sym, g_sym, parameters=None):
        self.x, self.y = sp.symbols('x y')
        self.parameters = parameters if parameters else set()
        self.f_sym = ensure_sympy_expression(f_sym)
        self.g_sym = ensure_sympy_expression(g_sym)
        self.variables = (self.x, self.y)
        self.jacobian_matrix = None
        self.equilibria = []
        self.nullclines = []
        self.results = []
        self.general_solution = (None, None)
        self.f_num = None
        self.g_num = None

    def run_full_analysis(self):
        self.compute_jacobian()
        self.find_equilibria()
        self.compute_nullclines()
        if self.equilibria:
            print("Puntos de equilibrio encontrados:")
            for eq in self.equilibria:
                pprint(eq)
            self.analyze_equilibria()
            self.display_results()
        else:
            print("No se encontraron puntos de equilibrio.")
        self.compute_general_solution()
        self.display_general_solution()
        print("\nNulclinas del sistema:")
        display_nullclines(self.nullclines)
        self.lambdify_functions()
        if self.parameters:
            param = next(iter(self.parameters))
            generate_bifurcation_diagram(self.f_sym, self.g_sym, param, param_range=(-2, 2))
        else:
            self.plot_phase_portrait()

    def compute_jacobian(self):
        self.jacobian_matrix = compute_jacobian_symbolic(self.f_sym, self.g_sym, self.variables)
        print("Matriz Jacobiana del Sistema:")
        pprint(self.jacobian_matrix)
        print()

    def find_equilibria(self):
        self.equilibria = find_equilibria_symbolic(self.f_sym, self.g_sym, self.parameters)

    def compute_nullclines(self):
        nullclines = []
        f_nullcline = solve(self.f_sym, self.y)
        if f_nullcline:
            nullclines.append({'variable': self.y, 'solutions': f_nullcline, 'label': "Nulclina x'"})
        else:
            f_nullcline = solve(self.f_sym, self.x)
            if f_nullcline:
                nullclines.append({'variable': self.x, 'solutions': f_nullcline, 'label': "Nulclina x'"})
        g_nullcline = solve(self.g_sym, self.y)
        if g_nullcline:
            nullclines.append({'variable': self.y, 'solutions': g_nullcline, 'label': "Nulclina y'"})
        else:
            g_nullcline = solve(self.g_sym, self.x)
            if g_nullcline:
                nullclines.append({'variable': self.x, 'solutions': g_nullcline, 'label': "Nulclina y'"})
        self.nullclines = nullclines

    def analyze_equilibria(self):
        self.results = analyze_equilibria(self.equilibria, self.f_sym, self.g_sym, self.parameters)

    def display_results(self):
        for result in self.results:
            eq = result["equilibrium"]
            J = result["jacobian"]
            eigenvals = result["eigenvals"]
            eigenvects = result["eigenvects"]
            print("\nPunto de equilibrio:")
            pprint(eq)
            print("Matriz Jacobiana en el equilibrio:")
            pprint(J)
            print("Valores propios:")
            for ev in eigenvals:
                ev_str = format_eigenvalue(ev)
                print(f"λ = {ev_str}")
            print("Vectores propios:")
            for ev, mult, vects in eigenvects:
                ev_str = format_eigenvalue(ev)
                for vect in vects:
                    vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                    vect_components = [str(comp).replace('I', 'i') for comp in vect_simplified]
                    vect_str = f"({', '.join(vect_components)})"
                    print(f"Vector propio asociado a λ = {ev_str}:\n{vect_str}")
            eigenvalues = list(eigenvals.keys())
            classification = classify_equilibrium(eigenvalues)
            print("\nClasificación del sistema:", classification)

    def compute_general_solution(self):
        A = Matrix([
            [self.f_sym.coeff(var) for var in self.variables],
            [self.g_sym.coeff(var) for var in self.variables]
        ])
        eigenvects = A.eigenvects()
        c = symbols('c1:%d' % (len(eigenvects) * A.shape[0] + 1))
        t = symbols('t')
        sol_x = 0
        sol_y = 0
        idx = 0
        for ev, mult, vects in eigenvects:
            for vect in vects:
                term = c[idx] * exp(ev * t) * vect
                sol_x += term[0]
                sol_y += term[1]
                idx += 1
        self.general_solution = (expand(sol_x), expand(sol_y))

    def display_general_solution(self):
        print("\nSolución general del sistema:")
        print(f"x(t) = {self.general_solution[0]}")
        print(f"y(t) = {self.general_solution[1]}")

    def lambdify_functions(self):
        vars = (self.x, self.y) + tuple(self.parameters)
        self.f_num = lambdify(vars, self.f_sym, modules=['numpy', {'I': 0}])
        self.g_num = lambdify(vars, self.g_sym, modules=['numpy', {'I': 0}])

    def vectorize_functions(self):
        f_vectorized = np.vectorize(self.f_num)
        g_vectorized = np.vectorize(self.g_num)
        return f_vectorized, g_vectorized

    def plot_phase_portrait(self):
        f_vectorized, g_vectorized = self.vectorize_functions()
        plot_phase_portrait(
            f_vectorized,
            g_vectorized,
            self.equilibria,
            results=self.results,
            nullclines=self.nullclines
        )
