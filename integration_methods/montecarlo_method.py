import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tabulate import tabulate

def montecarlo_integracion(f, desde, hasta, cantidad_puntos, repeticiones=10, precision=5):
    a = desde
    b = hasta

    x_random = np.random.uniform(a, b, cantidad_puntos)
    f_values = f(x_random)
    f_mean = np.mean(f_values)
    area = abs((b - a) * f_mean)
    resultados = []

    print(tabulate(resultados, headers=["i", "x_i", "f(x_i)"], tablefmt="grid"))
    print(f"\nEstimación de la integral con {cantidad_puntos} puntos aleatorios: {round(area, precision)}")
    graficar_montecarlo(f, area, x_random, f_values, a, b)
    return area

def graficar_montecarlo(f, area, x_random, f_values, desde, hasta):
    x_vals = np.linspace(desde, hasta, 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    plt.scatter(x_random, f_values, color='red', alpha=0.5, label='Puntos aleatorios')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Método de Montecarlo. Area bajo la curva: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()





def montecarlo_pi(cantidad_puntos, repeticiones=10, precision=8):
    resultados = []
    pi_promedio = 0
    for i in range(repeticiones):  # Solo mostrar los primeros 10 puntos para la tabla
        # Generar puntos aleatorios en el cuadrado [-1, 1] x [-1, 1]
        x_random = np.random.uniform(-1, 1, cantidad_puntos)
        y_random = np.random.uniform(-1, 1, cantidad_puntos)

        dentro_del_circulo = x_random**2 + y_random**2 <= 1
        num_dentro_del_circulo = np.sum(dentro_del_circulo)

        # Estimar pi
        pi_estimado = 4 * num_dentro_del_circulo / cantidad_puntos
        pi_promedio += pi_estimado
        resultados.append([i+1, round(num_dentro_del_circulo, precision), pi_estimado])
    pi_promedio = pi_promedio/10
    print(tabulate(resultados, headers=["i", "Puntos en el circulo", "Pi estimado"], tablefmt="grid"))
    print(f"\nEstimación de pi con {cantidad_puntos} puntos: {pi_promedio}")
    graficar_montecarlo_pi(x_random, y_random, dentro_del_circulo)

    return pi_estimado

def graficar_montecarlo_pi(x_random, y_random, dentro_del_circulo):
    plt.figure(figsize=(6, 6))
    plt.scatter(x_random[dentro_del_circulo], y_random[dentro_del_circulo], color='green', alpha=0.5, label='Dentro del círculo')
    plt.scatter(x_random[~dentro_del_circulo], y_random[~dentro_del_circulo], color='red', alpha=0.5, label='Fuera del círculo')
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('Estimación de pi usando Montecarlo')
    plt.legend()
    plt.show()


def montecarlo_integracion_entre_curvas(f1, f2, cantidad_puntos, repeticiones=10, precision=5):
    interseccion = lambda x: f1(x) - f2(x)
    desde_x = fsolve(interseccion, 0)[0]
    hasta_x = fsolve(interseccion, 1)[0]
    desde_y = min(min(f1(desde_x), f2(desde_x)), min(f1(hasta_x), f2(hasta_x)))
    hasta_y = max(max(f1(desde_x), f2(desde_x)), max(f1(hasta_x), f2(hasta_x)))
    cantidad_total_puntos = []
    areas = []

    for i in range(repeticiones):
        x_random = np.random.uniform(desde_x, hasta_x, cantidad_puntos)
        y_random = np.random.uniform(desde_y, hasta_y, cantidad_puntos)
        
        puntos_dentro = np.sum((y_random > np.minimum(f1(x_random), f2(x_random))) & (y_random < np.maximum(f1(x_random), f2(x_random))))
        area_rectangulo = (hasta_x - desde_x) * (hasta_y - desde_y)
        area = (puntos_dentro / cantidad_puntos) * area_rectangulo
        areas.append(area)
        cantidad_total_puntos.append([i+1, puntos_dentro, area])

    print(tabulate(cantidad_total_puntos, headers=["Iteración", "Puntos dentro", "Area"], tablefmt="grid"))
    area_promedio = np.average(areas)
    print(f"\nPromedio del área estimada {repeticiones} veces: {round(area_promedio, precision)}")

    graficar_montecarlo(f1, f2, x_random, y_random, desde_x, hasta_x, area_promedio)
    
    return np.average(areas)

def graficar_montecarlo(f1, f2, x_random, y_random, desde, hasta, area_promedio):
    x_vals = np.linspace(desde, hasta, 400)
    f1_vals = f1(x_vals)
    f2_vals = f2(x_vals)
    plt.plot(x_vals, f1_vals, label='$f(x) = x^2$', color='blue')
    plt.plot(x_vals, f2_vals, label='$g(x) = \sqrt{x}$', color='green')
    puntos_dentro_mask = (y_random > f1(x_random)) & (y_random < f2(x_random))
    plt.scatter(x_random[puntos_dentro_mask], y_random[puntos_dentro_mask], color='orange', alpha=0.5, label='Puntos dentro')
    plt.scatter(x_random[~puntos_dentro_mask], y_random[~puntos_dentro_mask], color='red', alpha=0.5, label='Puntos fuera')
    plt.text(0.5, max(np.max(f1_vals), np.max(f2_vals)) * 0.9, f'Área estimada: {area_promedio}', fontsize=12, ha='center', color='black')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Montecarlo: puntos dentro y fuera del área entre curvas')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.show()