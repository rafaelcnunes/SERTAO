import numpy as np
from getdist import MCSamples, loadMCSamples, plots
import matplotlib.pyplot as plt

# ================================================================
# Configuration
# ================================================================

CHAIN_ROOTS = ['chains/LCDM']
COLORS  = ['royalblue', 'tomato', 'green', 'orange']
LABELS  = ['LCDM']

IGNORE_ROWS = 0.3   # burn-in fraction to discard

# ================================================================
# Parameters to plot
# ================================================================
PLOT_PARAMS = ['hundredh', 'omegacdm100', 'omegab100', 'Omegam', 'sigma8']

PARAM_LABELS = {
    'hundredh':   r'100\,h',
    'omegacdm100':r'100\,\Omega_{\rm cdm}h^2',
    'omegab100':  r'100\,\Omega_b h^2',
    'Omegam':     r'\Omega_m',
    'sigma8':     r'\sigma_8',
}

# ================================================================
# Load samples and add derived parameters
# ================================================================

all_samples = []

for root in CHAIN_ROOTS:
    samp = loadMCSamples(root, settings={'ignore_rows': IGNORE_ROWS})
    pnames = [p.name for p in samp.getParamNames().names]

    # Raw chains: columns are weight, -logL, H0, Omega_cdm, Omega_b, [sigma8, ...]
    H0       = samp.getParams().H0
    Omega_cdm= samp.getParams().Omega_cdm
    Omega_b  = samp.getParams().Omega_b

    h        = H0 / 100.0
    Omega_m  = Omega_cdm + Omega_b

    # Derived quantities
    hundredh    = 100.0 * h
    omegacdm100 = 100.0 * Omega_cdm * h**2
    omegab100   = 100.0 * Omega_b   * h**2

    # Add derived parameters — skip any that are already in the chains
    def _add(arr, name, label):
        if name not in pnames:
            samp.addDerived(arr, name=name, label=label)

    _add(hundredh,    'hundredh',    PARAM_LABELS['hundredh'])
    _add(omegacdm100, 'omegacdm100', PARAM_LABELS['omegacdm100'])
    _add(omegab100,   'omegab100',   PARAM_LABELS['omegab100'])
    _add(Omega_m,     'Omegam',      PARAM_LABELS['Omegam'])

    # sigma8: sampled → already in chains; not sampled → add as fixed derived
    if 'sigma8' not in pnames:
        samp.addDerived(np.full(len(H0), 0.8101),
                        name='sigma8', label=PARAM_LABELS['sigma8'])

    all_samples.append(samp)

# ================================================================
# Triangle plot
# ================================================================

g = plots.get_subplot_plotter(subplot_size=2.2)
g.settings.axes_labelsize  = 14
g.settings.legend_fontsize = 13
g.settings.axes_fontsize   = 11

g.triangle_plot(
    all_samples,
    params      = PLOT_PARAMS,
    colors      = COLORS[:len(all_samples)],
    legend_labels=LABELS[:len(all_samples)],
    legend_loc  = 'upper right',
    filled      = True,
    line_args   = [{'lw': 1.5} for _ in all_samples],
)

plt.savefig('triangle_plot.pdf', bbox_inches='tight', dpi=150)
plt.savefig('triangle_plot.png', bbox_inches='tight', dpi=150)
plt.show()

# ================================================================
# Summary statistics
# ================================================================

print("\n" + "="*60)
for label, samp in zip(LABELS, all_samples):
    print(f"\n{label}")
    print(samp.getTable(limit=1).tableTex())
    
