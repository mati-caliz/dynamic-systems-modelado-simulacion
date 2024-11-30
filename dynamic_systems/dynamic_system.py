import sympy as sp
import numpy as np
from sympy import symbols, Matrix, lambdify, solve, simplify, sympify, pprint, nsimplify, S, Rational, Expr, exp
from jacobian import compute_jacobian_symbolic, compute_jacobian_at_equilibrium
from equilibria import find_equilibria_symbolic, analyze_equilibria
from plotting import plot_phase_portrait, display_nullclines
from bifurcation import generate_bifurcation_diagram

class DynamicSystem:
    def __init__(self, f_sym, g_sym, parameters=None):
        self.x, self.y = sp.symbols('x y')
        self.parameters = parameters if parameters else set()
        self.f_sym = self.ensure_sympy_expression(f_sym)
        self.g_sym = self.ensure_sympy_expression(g_sym)
        self.variables = (self.x, self.y)
        self.jacobian_matrix = None
        self.equilibria = []
        self.nullclines = []
        self.results = []

    def ensure_sympy_expression(self, func):
        expr = sympify(func, evaluate=False)
        expr = nsimplify(expr, rational=True)
        return expr

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
        print("\nNuclinas del sistema:")
        display_nullclines(self.nullclines)
        self.lambdify_functions()
        self.plot_phase_portrait()
        if self.parameters:
            param = next(iter(self.parameters))
            generate_bifurcation_diagram(self.f_sym, self.g_sym, param, param_range=(-2, 2))

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
            nullclines.append({'variable': self.y, 'solutions': f_nullcline, 'label': "Nuclina x'"})
        else:
            f_nullcline = solve(self.f_sym, self.x)
            if f_nullcline:
                nullclines.append({'variable': self.x, 'solutions': f_nullcline, 'label': "Nuclina x'"})
        g_nullcline = solve(self.g_sym, self.y)
        if g_nullcline:
            nullclines.append({'variable': self.y, 'solutions': g_nullcline, 'label': "Nuclina y'"})
        else:
            g_nullcline = solve(self.g_sym, self.x)
            if g_nullcline:
                nullclines.append({'variable': self.x, 'solutions': g_nullcline, 'label': "Nuclina y'"})
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
                ev_simplified = nsimplify(ev, rational=True)
                ev_str = str(ev_simplified).replace('I', 'i')
                print(f"λ = {ev_str}")
            print("Vectores propios:")
            for ev, mult, vects in eigenvects:
                ev_simplified = nsimplify(ev, rational=True)
                ev_str = str(ev_simplified).replace('I', 'i')
                for vect in vects:
                    vect_simplified = vect.applyfunc(lambda x: nsimplify(x, rational=True))
                    vect_components = [str(comp).replace('I', 'i') for comp in vect_simplified]
                    vect_str = f"({', '.join(vect_components)})"
                    print(f"Vector propio asociado a λ = {ev_str}:\n{vect_str}")
            eigenvalues = list(eigenvals.keys())
            classification = self.classify_equilibrium(eigenvalues)
            print("\nClasificación del sistema:", classification)

    def classify_equilibrium(self, eigenvalues):
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
        self.general_solution = (simplify(sol_x), simplify(sol_y))

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
            self.parameters,
            results=self.results,
            nullclines=self.nullclines
        )
