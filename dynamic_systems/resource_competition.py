import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parámetros del modelo
r1 = 0.1  # Tasa de crecimiento de los conejos
r2 = 0.05  # Tasa de crecimiento de las ovejas
K1 = 1000  # Capacidad de carga para los conejos
K2 = 500  # Capacidad de carga para las ovejas
alpha12 = 0.01  # Efecto de las ovejas sobre los conejos
alpha21 = 0.02  # Efecto de los conejos sobre las ovejas


# Sistema de ecuaciones diferenciales no lineales
def non_linear_lotka_volterra(N, t, r1, r2, K1, K2, alpha12, alpha21):
    N1, N2 = N
    dN1_dt = r1 * N1 * (1 - (N1 + alpha12 * N2 ** 2) / K1)
    dN2_dt = r2 * N2 * (1 - (N2 + alpha21 * N1 ** 2) / K2)
    return [dN1_dt, dN2_dt]


# Condiciones iniciales
N0 = [50, 30]  # Poblaciones iniciales de conejos y ovejas

# Intervalo de tiempo
t = np.linspace(0, 200, 1000)

# Resolver el sistema de ecuaciones diferenciales
sol = odeint(non_linear_lotka_volterra, N0, t, args=(r1, r2, K1, K2, alpha12, alpha21))

# Graficar los resultados
plt.plot(t, sol[:, 0], label='Conejos')
plt.plot(t, sol[:, 1], label='Ovejas')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Competencia No Lineal entre Conejos y Ovejas')
plt.show()