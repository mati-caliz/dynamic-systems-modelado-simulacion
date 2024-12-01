import sympy as sp
from dynamic_system import DynamicSystem

def main():
    x, y, a = sp.symbols('x y a')
    f_x = 3/2 * x + 1/2 * y
    g_x = 1/2 * x + 3/2 * y

    system = DynamicSystem(
        f_sym=f_x,
        g_sym=g_x
    )

    system.run_full_analysis()


if __name__ == "__main__":
    main()
