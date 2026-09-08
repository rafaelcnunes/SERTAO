# ============================================================
# Union 3.0 likelihood
# arXiv:2311.12098
# ============================================================

import numpy as np
import pandas as pd
import scipy.linalg as la
import os

class Union3Likelihood:
    """
    Union3 likelihood
    """
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self._load_data_files()
        self._compute_cholesky()
    
    def _load_data_files(self):
        """Load Union3 data files"""
        data_file = os.path.join(self.data_dir, 'lcparam_full.txt')
        cov_file = os.path.join(self.data_dir, 'mag_covmat.txt')

        self.data = pd.read_csv(data_file, sep='\s+', comment='#', header=None)
        
        column_names = ['name', 'zcmb', 'zhel', 'dz', 'mb', 'dmb', 
                       'x1', 'dx1', 'color', 'dcolor', '3rdvar', 'd3rdvar',
                       'cov_m_s', 'cov_m_c', 'cov_s_c', 'set', 'ra', 'dec', 'biascor']
        self.data.columns = column_names[:len(self.data.columns)]
        
        self.zcmb = self.data['zcmb'].values
        self.zhel = self.data['zhel'].values
        
        # Observed apparent magnitude (already corrected by Union3)
        self.mu_obs = self.data['mb'].values
        
        self.n_sne = len(self.zcmb)
        
        # Load covariance matrix
        with open(cov_file, 'r') as f:
            first_line = f.readline().strip()
            elements = list(map(float, first_line.split()))
            
            if len(elements) == 1:
                n = int(elements[0])
                remaining = f.read()
                if remaining:
                    all_elements = list(map(float, remaining.split()))
                else:
                    next_line = f.readline()
                    all_elements = list(map(float, next_line.split()))
            else:
                all_elements = elements
            
            total_elements = len(all_elements)
            n = int(np.sqrt(total_elements))
            
            if n * n != total_elements:
                raise ValueError(f"Invalid matrix: {total_elements} elements")
            
            self.cov_matrix = np.array(all_elements).reshape((n, n))
            
            if n != self.n_sne:
                if n > self.n_sne:
                    self.cov_matrix = self.cov_matrix[:self.n_sne, :self.n_sne]
                else:
                    self.zcmb = self.zcmb[:n]
                    self.zhel = self.zhel[:n]
                    self.mu_obs = self.mu_obs[:n]
                    self.n_sne = n
    
    def _compute_cholesky(self):
        """Compute Cholesky decomposition"""
        try:
            self.cov_cholesky = la.cholesky(self.cov_matrix, lower=True)
        except la.LinAlgError:
            eigvals = la.eigvalsh(self.cov_matrix)
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            reg = np.eye(self.n_sne) * (abs(min_eig) + 1e-8 * max_eig)
            self.cov_matrix += reg
            self.cov_cholesky = la.cholesky(self.cov_matrix, lower=True)
    
    def calculate_mu_theory(self, cosmo, Mcal):

        mu_theo = np.zeros(self.n_sne)
        const = 43.15861
        
        for i in range(self.n_sne):
            z_cmb = self.zcmb[i]
            z_hel = self.zhel[i]

            r = cosmo.comoving_distance(z_cmb)

            dL = (1.0 + z_hel) * r

            mu_theo[i] = Mcal + const + 5.0 * np.log10(dL)
        
        return mu_theo
    
    def loglkl(self, cosmo, Mcal):
        """
        Compute log-likelihood for Union3
        """
        mu_theo = self.calculate_mu_theory(cosmo, Mcal)
        
        residuals = self.mu_obs - mu_theo
        
        residuals_whitened = la.solve_triangular(
            self.cov_cholesky, residuals, lower=True, check_finite=False
        )
        
        chi2 = np.sum(residuals_whitened**2)
        logL = -0.5 * chi2
        
        return logL, chi2
    
    def get_data(self):
        """Return data for plotting"""
        mu_err = np.sqrt(np.diag(self.cov_matrix))
        return self.zcmb, self.mu_obs, mu_err


# Função de conveniência
def union3_likelihood(cosmo, Mcal, data_dir='./data'):
    """
    Convenience function for Union3 likelihood
    """
    if not hasattr(union3_likelihood, '_instance'):
        union3_likelihood._instance = Union3Likelihood(data_dir=data_dir)
    
    return union3_likelihood._instance.loglkl(cosmo, Mcal)

