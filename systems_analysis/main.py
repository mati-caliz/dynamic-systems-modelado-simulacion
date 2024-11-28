from dynamic_system import DynamicSystem
import sympy as sp

def main():
    x, y = sp.symbols('x y')
    # Definir las funciones simbólicas
    f_sym = x**2 - y**2 + 1
    g_sym = 0
    # Ejecutar el análisis completo
    system = DynamicSystem(f_sym, g_sym)
    system.run_full_analysis()

if __name__ == "__main__":
    main()
