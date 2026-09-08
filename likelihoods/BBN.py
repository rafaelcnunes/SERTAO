import numpy as np

class BBNLikelihood:
    """
    Likelihood para BBN (Big Bang Nucleosynthesis)
    Baseado em: ω_b h² = 0.02230 ± 0.00014 - Planck 2018 paper
    """
    
    def __init__(self, omega_b_h2_mean=0.02230, omega_b_h2_std=0.00014):
        """
        Inicializa likelihood BBN
        
        Args:
            omega_b_h2_mean: Valor médio de ω_b h² (default: Planck 2018)
            omega_b_h2_std: Desvio padrão de ω_b h² (default: Planck 2018)
        """
        self.omega_b_h2_mean = omega_b_h2_mean
        self.omega_b_h2_std = omega_b_h2_std
        
    def calculate_omega_b_h2(self, H0, Omega_b):
        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        return omega_b_h2
    
    def loglkl(self, H0, Omega_b):
        omega_b_h2 = self.calculate_omega_b_h2(H0, Omega_b)
        
        # Likelihood gaussiana
        chi2 = ((omega_b_h2 - self.omega_b_h2_mean) / self.omega_b_h2_std)**2
        logL = -0.5 * chi2
        
        return logL, omega_b_h2

