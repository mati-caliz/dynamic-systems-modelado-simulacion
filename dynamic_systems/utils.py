# utils.py
import sympy as sp
from sympy import sympify, nsimplify


def ensure_sympy_expression(func):
    expr = sympify(func, evaluate=False)
    expr = nsimplify(expr, rational=True)
    return expr


def classify_equilibrium(eigenvalues):
    real_parts = [ev.as_real_imag()[0] for ev in eigenvalues]
    imag_parts = [ev.as_real_imag()[1] for ev in eigenvalues]
    real_positive = [re.is_positive for re in real_parts]
    real_negative = [re.is_negative for re in real_parts]
    imag_nonzero = [im != 0 for im in imag_parts]

    if None in real_positive or None in real_negative or None in imag_nonzero:
        return "No se puede determinar la clasificación del equilibrio debido a valores propios simbólicos."

    if all(im for im in imag_nonzero):
        if all(re for re in real_positive):
            return "Foco Inestable (valores propios complejos con parte real positiva)"
        elif all(re for re in real_negative):
            return "Foco Estable (valores propios complejos con parte real negativa)"
        elif all(not re for re in real_positive + real_negative):
            return "Centro (valores propios puramente imaginarios)"
        else:
            return "Espiral Silla (valores propios complejos con partes reales de signos opuestos)"
    else:
        if all(re for re in real_positive):
            return "Nodo Inestable (valores propios reales y positivos)"
        elif all(re for re in real_negative):
            return "Nodo Estable (valores propios reales y negativos)"
        elif any(re for re in real_positive) and any(re for re in real_negative):
            return "Punto Silla (valores propios reales de signos opuestos)"
        elif all(not re for re in real_positive + real_negative):
            return "Centro o Nodo Degenerado (valores propios reales nulos o repetidos)"
        else:
            return "Otro tipo de equilibrio"


def format_eigenvalue(ev):
    ev_simplified = sp.nsimplify(ev, rational=True)
    return str(ev_simplified).replace('I', 'i')
