import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

def load_samples_with_omegam(path, ignore_rows=0.3):
    """
    Carrega chains do GetDist e adiciona Omega_m derivado.
    """
    samples = loadMCSamples(
        path,
        settings={'ignore_rows': ignore_rows}
    )

    p = samples.getParams()
    Omega_m = p.Omega_cdm + p.Omega_b

    samples.addDerived(
        Omega_m,
        name='Omega_m',
        label=r'\Omega_m'
    )

    return samples


# ============================================================
# Exemplo de como carregar cadeias
# ============================================================

# LCDM — PPS
samples_PPS = load_samples_with_omegam(
    'chains/LCDM_PPS/LCDM'
)

# LCDM — BAO
samples_BAO = load_samples_with_omegam(
    'chains/LCDM_BAO/LCDM_BAO'
)

# LCDM — CMB
samples_CMB = load_samples_with_omegam(
    'chains/LCDM_CMB/LCDM_CMB'
)

# LCDM — PPS + BAO
samples_PPS_BAO = load_samples_with_omegam(
    'chains/LCDM_PPS_BAO/LCDM_PPS_BAO'
)

# ============================================================
# Parâmetros a plotar
# ============================================================

params_to_plot = [
    'Omega_m',
    'H0',
    'M_B',
]


# ============================================================
# Configuração do plot
# ============================================================

g = plots.get_subplot_plotter()

g.settings.alpha_filled_add = 0.5
g.settings.title_limit_fontsize = 14
g.settings.figure_legend_frame = False
g.settings.legend_fontsize = 19
g.settings.axes_labelsize = 16


# ============================================================
# Triangle plot
# ============================================================

g.triangle_plot(
    [
        samples_PPS,
        samples_BAO,
        samples_CMB,
    ],
    params_to_plot,
    colors=[
        'green',
        'black',
        'blue',
    ],
    filled=True,
    legend_labels=[
        r'PP&SHOES: $\Lambda$CDM',
        r'BAO: $\Lambda$CDM',
        r'CMB: $\Lambda$CDM',
    ],
    legend_loc='upper right'
)


# ============================================================
# Estatísticas
# ============================================================

loglike = samples_PPS_eta.loglikes
chi2_min = -2.0 * np.max(loglike)

print(r"$\chi^2_{\rm min} =$", chi2_min)

# Gelman–Rubin (somente se houver múltiplas cadeias)
try:
    print(
        "Gelman–Rubin diagnostic =",
        samples_PPS_eta.getGelmanRubin()
    )
except Exception:
    print("Gelman–Rubin não disponível (cadeia única).")

# Tabela LaTeX (best-fit)
print(samples_PPS_CMB_eta.getTable(limit=1).tableTex())


# ============================================================
# Mostrar figura
# ============================================================

plt.show()


