# Modelado y Simulación

Métodos numéricos y análisis de sistemas dinámicos, hechos para la materia Modelado y
Simulación de UADE.

## Qué hay

| Carpeta | Contenido |
|---|---|
| `dynamic_systems/` | Sistemas dinámicos lineales y no lineales en 2D: puntos de equilibrio y su clasificación por el jacobiano, retratos de fase y diagramas de bifurcación. Incluye los modelos de depredador-presa y de competencia por recursos. |
| `root_finding/` | Búsqueda de raíces: Newton-Raphson, secante y Steffensen-Aitken |
| `differential_equations/` | Euler, Euler mejorado, Runge-Kutta, punto fijo e interpolación de Lagrange |
| `integration_methods/` | Integración numérica: rectángulos, trapecios, Simpson y Monte Carlo |

## Uso

Necesita Python 3.8 o más nuevo.

```bash
pip install -r requirements.txt
cd dynamic_systems && python main_dynamic_systems.py
```

El sistema se define en `main_dynamic_systems.py` con expresiones de SymPy, con o sin
parámetros; `run_full_analysis()` calcula los equilibrios, los clasifica y grafica.
