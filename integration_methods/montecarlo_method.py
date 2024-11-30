import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tabulate import tabulate
import sympy as sp

def montecarlo_integracion(x, f, desde, hasta, cantidad_puntos, repeticiones=10, precision=5):
    f = sp.lambdify(x, f, 'numpy')
    a = desde
    b = hasta
    resultados = []
    areas = []
    for _ in range(repeticiones):
        x_random = np.random.uniform(a, b, cantidad_puntos)
        f_values = f(x_random)
        f_mean = np.mean(f_values)
        area = abs((b - a) * f_mean)
        areas.append(area)
        resultados.append([len(resultados) + 1, a, b, round(area, precision)])
    print(tabulate(resultados, headers=["Repetición", "Desde", "Hasta", "Área estimada"], tablefmt="grid"))
    area_promedio = np.mean(areas)
    print(f"\nPromedio del área estimada sobre {repeticiones} repeticiones: {round(area_promedio, precision)}")
    graficar_montecarlo(x, f, area_promedio, x_random, f_values, a, b)
    return area_promedio

def graficar_montecarlo(x, f, area, x_random, f_values, desde, hasta):
    x_vals = np.linspace(desde, hasta, 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    plt.scatter(x_random, f_values, color='red', alpha=0.5, label='Puntos aleatorios')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Método de Monte Carlo. Área bajo la curva: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()

def montecarlo_pi(cantidad_puntos, repeticiones=10, precision=8):
    resultados = []
    pi_promedio = 0
    for _ in range(repeticiones):
        x_random = np.random.uniform(-1, 1, cantidad_puntos)
        y_random = np.random.uniform(-1, 1, cantidad_puntos)
        dentro_del_circulo = x_random**2 + y_random**2 <= 1
        num_dentro_del_circulo = np.sum(dentro_del_circulo)
        pi_estimado = 4 * num_dentro_del_circulo / cantidad_puntos
        pi_promedio += pi_estimado
        resultados.append([len(resultados) + 1, num_dentro_del_circulo, round(pi_estimado, precision)])
    pi_promedio /= repeticiones
    print(tabulate(resultados, headers=["Repetición", "Puntos dentro", "Pi estimado"], tablefmt="grid"))
    print(f"\nPromedio de pi estimado sobre {repeticiones} repeticiones: {round(pi_promedio, precision)}")
    graficar_montecarlo_pi(x_random, y_random, dentro_del_circulo)
    return pi_promedio

def graficar_montecarlo_pi(x_random, y_random, dentro_del_circulo):
    plt.figure(figsize=(6, 6))
    plt.scatter(x_random[dentro_del_circulo], y_random[dentro_del_circulo], color='green', alpha=0.5, label='Dentro del círculo')
    plt.scatter(x_random[~dentro_del_circulo], y_random[~dentro_del_circulo], color='red', alpha=0.5, label='Fuera del círculo')
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2, label='Círculo unitario')
    plt.gca().add_patch(circle)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('Estimación de π usando Monte Carlo')
    plt.legend()
    plt.show()

def montecarlo_integracion_entre_curvas(x, f1, f2, cantidad_puntos, repeticiones=10, precision=5):
    f1 = sp.lambdify(x, f1, 'numpy')
    f2 = sp.lambdify(x, f2, 'numpy')
    interseccion = lambda x_val: f1(x_val) - f2(x_val)
    desde_x = fsolve(interseccion, 0)[0]
    hasta_x = fsolve(interseccion, 1)[0]
    desde_y = min(min(f1(desde_x), f2(desde_x)), min(f1(hasta_x), f2(hasta_x)))
    hasta_y = max(max(f1(desde_x), f2(desde_x)), max(f1(hasta_x), f2(hasta_x)))
    resultados = []
    areas = []
    for _ in range(repeticiones):
        x_random = np.random.uniform(desde_x, hasta_x, cantidad_puntos)
        y_random = np.random.uniform(desde_y, hasta_y, cantidad_puntos)
        puntos_dentro = np.sum((y_random > np.minimum(f1(x_random), f2(x_random))) &
                                (y_random < np.maximum(f1(x_random), f2(x_random))))
        area_rectangulo = (hasta_x - desde_x) * (hasta_y - desde_y)
        area = (puntos_dentro / cantidad_puntos) * area_rectangulo
        areas.append(area)
        resultados.append([len(resultados) + 1, puntos_dentro, round(area, precision)])
    area_promedio = np.mean(areas)
    print(tabulate(resultados, headers=["Repetición", "Puntos dentro", "Área estimada"], tablefmt="grid"))
    print(f"\nPromedio del área estimada sobre {repeticiones} repeticiones: {round(area_promedio, precision)}")
    graficar_montecarlo_entre_curvas(f1, f2, x_random, y_random, desde_x, hasta_x, area_promedio, desde_y, hasta_y)
    return area_promedio

def graficar_montecarlo_entre_curvas(f1, f2, x_random, y_random, desde, hasta, area_promedio, desde_y, hasta_y):
    x_vals = np.linspace(desde, hasta, 400)
    f1_vals = f1(x_vals)
    f2_vals = f2(x_vals)
    plt.plot(x_vals, f1_vals, label='$f_1(x) = x^2$', color='blue')
    plt.plot(x_vals, f2_vals, label=r'$f_2(x) = \sqrt{x}$', color='green')
    puntos_dentro_mask = (y_random > np.minimum(f1(x_random), f2(x_random))) & \
                          (y_random < np.maximum(f1(x_random), f2(x_random)))
    plt.scatter(x_random[puntos_dentro_mask], y_random[puntos_dentro_mask], color='orange', alpha=0.5, label='Puntos dentro')
    plt.scatter(x_random[~puntos_dentro_mask], y_random[~puntos_dentro_mask], color='red', alpha=0.5, label='Puntos fuera')
    plt.text(desde + (hasta - desde) * 0.5, hasta_y * 0.9, f'Área estimada: {area_promedio}', fontsize=12, ha='center', color='black')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Monte Carlo: Área entre curvas')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.show()

def main():
    x = sp.symbols('x')
    desde_integracion = 0
    hasta_integracion = 3
    cantidad_puntos_integracion = 1000
    repeticiones_integracion = 10
    precision_integracion = 5
    print("=== Integración Monte Carlo ===")
    f_x = x**3 - 2*x - 5
    montecarlo_integracion(
        x,
        f_x,
        desde=desde_integracion,
        hasta=hasta_integracion,
        cantidad_puntos=cantidad_puntos_integracion,
        repeticiones=repeticiones_integracion,
        precision=precision_integracion
    )
    print("\n=== Estimación de π usando Monte Carlo ===")
    cantidad_puntos_pi = 1000
    repeticiones_pi = 10
    precision_pi = 8
    montecarlo_pi(
        cantidad_puntos=cantidad_puntos_pi,
        repeticiones=repeticiones_pi,
        precision=precision_pi
    )
    print("\n=== Integración Monte Carlo entre Curvas ===")
    f1_x = x**2
    f2_x = sp.sqrt(x)
    cantidad_puntos_curvas = 1000
    repeticiones_curvas = 10
    precision_curvas = 5
    montecarlo_integracion_entre_curvas(
        x,
        f1_x,
        f2_x,
        cantidad_puntos=cantidad_puntos_curvas,
        repeticiones=repeticiones_curvas,
        precision=precision_curvas
    )

if __name__ == "__main__":
    main()
