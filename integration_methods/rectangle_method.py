import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def area_rectangulos(f, desde, hasta, cantidad_rectangulos, punto="medio", precision=5):
    a = desde
    b = hasta
    n = cantidad_rectangulos
    h = (b-a)/n
    area_acumulada = 0
    resultados = []
    x_rect = []
    y_rect = []
    for i in range(n):
        if(punto == "izquierdo"):
            x_i = a + i * h
        elif(punto == "derecho"):
            x_i = a + i * h + h
        else: #punto medio
            x_i = a + i * h + h / 2
        
        area_i = f(x_i)
        area_acumulada += area_i
        x_rect.append(a + i * h + h)
        y_rect.append(f(a + i * h + h))
        resultados.append([round(i+1, precision), round(a + i * h + h, precision), round(x_i, precision), round(area_i, precision)])
    area = round(area_acumulada * h, precision)
    print(f"Area aproximada bajo la curva desde {a} hasta {b}: {area}")
    print(f"Se calculó utilizando punto {punto}.")

    resultados.append(["","","",round(area_acumulada, precision),area])
    print(tabulate(resultados, headers=["i", "x", "xi", "f(xi)","area  [f(xi)*h]"], tablefmt="grid"))
    graficar(f, area, a, b, x_rect, y_rect, h, punto)
    return area

# Función para graficar la curva y los rectángulos
def graficar(f, area, desde, hasta, x_rect, y_rect, ancho_rect, punto):
    # Valores para la curva de la función
    x_vals = np.linspace(desde, hasta, 400)
    y_vals = f(x_vals)

    # Crear la gráfica
    plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
    
    # Graficar los rectángulos
    for x, y in zip(x_rect, y_rect):
        if punto == "izquierdo":
            plt.bar(x, y, width=ancho_rect, align='edge', edgecolor='red', color='red', alpha=0.3)
        elif punto == "derecho":
            plt.bar(x - ancho_rect, y, width=ancho_rect, align='edge', edgecolor='green', color='green', alpha=0.3)
        else:  # punto medio
            plt.bar(x - ancho_rect/2, y, width=ancho_rect, align='center', edgecolor='orange', color='orange', alpha=0.3)
    
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Área bajo la curva usando punto {punto}: {area}')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    
    # Mostrar la gráfica
    plt.show()