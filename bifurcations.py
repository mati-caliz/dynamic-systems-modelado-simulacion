import numpy as np
import matplotlib.pyplot as plt



def system_silla_nodo(x, mu):
    return mu + x ** 2


def system_transcritica(x, mu):
    return mu * x - x ** 2


def bifurcation_silla_nodo(mu_range):
    x_positive = [np.sqrt(abs(mu)) if mu > 0 else None for mu in mu_range]
    x_negative = [-np.sqrt(abs(mu)) if mu > 0 else None for mu in mu_range]
    return x_positive, x_negative


def bifurcation_transcritica(mu_range):
    x_equilibrium_1 = np.zeros_like(mu_range)
    x_equilibrium_2 = mu_range.copy()
    return x_equilibrium_1, x_equilibrium_2


def plot_sistema(ax, system_func, x, mu, title):
    y = system_func(x, mu)
    ax.plot(x, y, label=rf"$\mu = {mu}$")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.legend()
    ax.grid()


def plot_bifurcation(ax, mu_range, equilibria, title, xlabel, ylabel):
    for eq, label in equilibria:
        ax.plot(mu_range, eq, label=label)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()


def setup_subplots(rows, cols, figsize=(18, 15)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    return fig, axes.flatten()


def clean_extra_subplots(fig, axes, used_axes):
    for idx, ax in enumerate(axes):
        if idx not in used_axes:
            fig.delaxes(ax)


def main():
    x = np.linspace(-2, 2, 400)
    mu_range = np.linspace(-2, 2, 200)
    fig, axes = setup_subplots(3, 3, figsize=(18, 15))
    used_axes = []
    silla_nodo_mus = [-1, 0, 1]
    for i, mu in enumerate(silla_nodo_mus):
        ax = axes[i]
        plot_sistema(
            ax,
            system_silla_nodo,
            x,
            mu,
            title=rf"Silla-Nodo: $\mu = {mu}$"
        )
        used_axes.append(i)
    x_positive, x_negative = bifurcation_silla_nodo(mu_range)
    equilibria_silla_nodo = [
        (x_positive, "Equilibrio estable"),
        (x_negative, "Equilibrio inestable")
    ]
    ax_bif_silla = axes[3]
    plot_bifurcation(
        ax_bif_silla,
        mu_range,
        equilibria_silla_nodo,
        title="Diagrama de bifurcación (Silla-Nodo)",
        xlabel=r"$\mu$",
        ylabel="$x$"
    )
    used_axes.append(3)

    # ===== Transcrítica =====
    transcritica_mus = [-1, 0, 1]
    for i, mu in enumerate(transcritica_mus):
        ax = axes[i + 4]  # Índices 4, 5, 6
        plot_sistema(
            ax,
            system_transcritica,
            x,
            mu,
            title=rf"Transcrítica: $\mu = {mu}$"
        )
        used_axes.append(i + 4)

    # Diagrama de bifurcación para Transcrítica
    x_eq1, x_eq2 = bifurcation_transcritica(mu_range)
    equilibria_transcritica = [
        (x_eq1, "Equilibrio 1 (0)"),
        (x_eq2, r"Equilibrio 2 ($\mu$)")
    ]
    ax_bif_trans = axes[7]
    plot_bifurcation(
        ax_bif_trans,
        mu_range,
        equilibria_transcritica,
        title="Diagrama de bifurcación (Transcrítica)",
        xlabel=r"$\mu$",
        ylabel="$x$"
    )
    used_axes.append(7)
    clean_extra_subplots(fig, axes, used_axes)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
