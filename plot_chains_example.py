import numpy as np
from getdist import MCSamples, loadMCSamples, plots
import matplotlib.pyplot as plt

# Carregar os resultados
samples1 = loadMCSamples('chains/BAO_BBN/2026-01-23_6098', settings={'ignore_rows':0.3})
samples2 = loadMCSamples('chains/U3_BAO_BBN/2026-01-23_6290', settings={'ignore_rows':0.3})
samples3 = loadMCSamples('chains/PP_BAO_BBN/2026-01-25_4743', settings={'ignore_rows':0.3})

# Plotar
g = plots.get_subplot_plotter()
g.triangle_plot([samples1,samples2,samples3], colors=['green', 'blue', 'black'], legend_labels=['BAO + BBN', 'U3 + BAO + BBN', 'PP + BAO + BBN'], legend_loc='upper right', filled=True)

# 6. Gerar tabela de estatísticas
print(samples2.getTable(limit=1).tableTex())

plt.show()
    
