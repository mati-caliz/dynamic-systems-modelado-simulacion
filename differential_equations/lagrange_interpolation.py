import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify, expand, Rational, pretty

def interpolacion_lagrange(puntos, cantidad_puntos=100):
    x = symbols('x')
    polinomio = 0
    
    print("Construcción del polinomio de Lagrange:")
    for i in range(len(puntos)):
        li = 1
        numeradores = []
        denominadores = []
        
        for j in range(len(puntos)):
            if i != j:
                numerador = f"(x - {puntos[j][0]})"
                denominador = f"({puntos[i][0]} - {puntos[j][0]})"
                numeradores.append(numerador)
                denominadores.append(denominador)
                li *= (x - puntos[j][0]) / (puntos[i][0] - puntos[j][0])
        
        numerador_planteo = ' * '.join(numeradores)
        denominador_planteo = ' * '.join(denominadores)
        
        print(f"\nTérmino l_{i}(x):")
        print(f"   {numerador_planteo}")
        print(f"  {'-' * max(len(numerador_planteo), len(denominador_planteo))}")
        print(f"   {denominador_planteo}")
        print(f"\n\n{pretty(simplify(li))}")
        polinomio += Rational(puntos[i][1]) * li

    polinomio_expandido = expand(polinomio)
    print(f"\nPolinomio de Lagrange: \n{pretty(polinomio_expandido)}")

    _min = min(punto[0] for punto in puntos)
    x_max = max(punto[0] for punto in puntos)
    puntos_evaluacion_x = np.linspace(_min, x_max, cantidad_puntos)
    polinomio_func = np.vectorize(lambda x_val: polinomio_expandido.evalf(subs={x: x_val}))
    valores_y = polinomio_func(puntos_evaluacion_x)
    graficar_lagrange(puntos, puntos_evaluacion_x, valores_y)

def graficar_lagrange(puntos, puntos_evaluacion_x, valores_y):
    x_nodos = [p[0] for p in puntos]
    y_nodos = [p[1] for p in puntos]
    plt.plot(x_nodos, y_nodos, 'o', label='Nodos', color='red')
    plt.plot(puntos_evaluacion_x, valores_y, '-', label='Interpolación de Lagrange', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Interpolación de Lagrange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()