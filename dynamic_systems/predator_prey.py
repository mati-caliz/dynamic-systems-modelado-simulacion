import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Definir el sistema de ecuaciones diferenciales
def lotka_volterra(state, t, a, b, c, d):
    x, y = state  # x: población presa, y: población depredador
    dxdt = a * x - b * x * y  # Crecimiento de las presas
    dydt = -c * y + d * x * y  # Crecimiento de los depredadores
    return [dxdt, dydt]

# Parámetros del modelo
a = 1.0  # Tasa de crecimiento de las presas (cuando no hay depredadores)
b = 0.1  # Tasa de depredación (eficiencia del depredador al capturar presas)
c = 1.5  # Tasa de mortalidad de los depredadores (cuando no hay presas)
d = 0.075  # Tasa de reproducción del depredador (eficiencia al convertir presas en nacimientos)

# Condiciones iniciales
x0 = 40  # Población inicial de las presas
y0 = 9   # Población inicial de los depredadores

# Tiempo de simulación
t = np.linspace(0, 200, 1000)  # Tiempo desde 0 hasta 200 dividido en 1000 puntos

# Resolver el sistema de ecuaciones diferenciales
solution = odeint(lotka_volterra, [x0, y0], t, args=(a, b, c, d))
x, y = solution.T  # Extraer las soluciones para x (presas) e y (depredadores)

# Graficar los resultados
plt.figure(figsize=(10, 6))

# Poblaciones a lo largo del tiempo
plt.subplot(2, 1, 1)
plt.plot(t, x, label="Presas (x)", color="blue")
plt.plot(t, y, label="Depredadores (y)", color="orange")
plt.title("Modelo Lotka-Volterra (Depredador-Presa)")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.legend()
plt.grid()

# Diagrama de fases
plt.subplot(2, 1, 2)
plt.plot(x, y, color="purple")
plt.title("Diagrama de fases: Presas vs Depredadores")
plt.xlabel("Población de Presas (x)")
plt.ylabel("Población de Depredadores (y)")
plt.grid()

plt.tight_layout()
plt.show()
