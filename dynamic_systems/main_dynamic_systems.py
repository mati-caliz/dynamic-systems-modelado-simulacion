import sympy as sp
from dynamic_system import DynamicSystem

def main():
    x, y = sp.symbols('x y')
    f_x = x**2-y**2+1
    g_x = y+x+1
    system = DynamicSystem(
        f_sym=f_x,
        g_sym=g_x
    )
    system.run_full_analysis()


if __name__ == "__main__":
    main()
