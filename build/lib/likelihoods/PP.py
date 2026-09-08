# ============================================================
# Pantheon+ likelihood
# arXiv:2202.04077
# ============================================================

import numpy as np
import scipy.linalg as la
import pandas as pd
import os


class PantheonPlusLikelihood:
    """
    Pantheon+ likelihood (sem SH0ES)
    - M_B é parâmetro livre
    """

    def __init__(
        self,
        data_dir='.',
        use_sys_cov=True,
        z_min=0.01,
        cache_DA=True,
    ):
        self.data_dir = data_dir
        self.use_sys_cov = use_sys_cov
        self.z_min = z_min
        self.cache_DA = cache_DA
        
        # Cache para DA(z)
        self._DA_cache = {}
        self._last_cosmo_hash = None

        # Carregar e pré-processar tudo uma vez
        self._load_and_preprocess()
        
        # Pré-calcular tudo que não depende da cosmologia
        self._precompute_static()

    # --------------------------------------------------
    # CARREGAMENTO E PRÉ-PROCESSAMENTO
    # --------------------------------------------------
    def _load_and_preprocess(self):
        data_file = os.path.join(self.data_dir, 'Pantheon+SH0ES.dat')
        self.data = pd.read_csv(data_file, sep=r'\s+', comment='#').values
        
        self.zHD_all = self.data[:, 2].astype(np.float32)      # zHD
        self.zHEL_all = self.data[:, 6].astype(np.float32)     # zHEL
        self.m_b_corr_all = self.data[:, 8].astype(np.float32) # m_b_corr
        self.is_calibrator_all = self.data[:, 13].astype(bool) # IS_CALIBRATOR
        
        self.origlen = len(self.data)
        
        mask = (self.zHD_all > self.z_min) & (~self.is_calibrator_all)

        self.zHD = self.zHD_all[mask]
        self.zHEL = self.zHEL_all[mask]
        self.m_b_corr = self.m_b_corr_all[mask]
        
        self.n_sne = len(self.zHD)
    
        self.mask = mask
        
        self._load_covariance(mask)
        
        self._mu_theory = np.empty(self.n_sne, dtype=np.float32)
        self._residuals = np.empty(self.n_sne, dtype=np.float32)
    
    def _load_covariance(self, mask):
        """Carrega matriz de covariância de forma otimizada"""
        if self.use_sys_cov:
            cov_file = os.path.join(self.data_dir, 'Pantheon+SH0ES_STAT+SYS.cov')
        else:
            cov_file = os.path.join(self.data_dir, 'Pantheon+SH0ES_STATONLY.cov')
        
        # Ler todo o arquivo de uma vez
        with open(cov_file, 'r') as f:
            lines = f.readlines()
        
        n_total = int(lines[0].strip())
        
        if n_total != self.origlen:
            raise ValueError(
                f"Tamanho da matriz de covariância ({n_total}) não corresponde "
                f"ao número de SNs no arquivo de dados ({self.origlen})"
            )
        
        # Converter todas as linhas de valores de uma vez
        data_lines = lines[1:]
        all_values = np.fromstring(' '.join(data_lines), sep=' ', dtype=np.float64)
        
        C_full = all_values.reshape(n_total, n_total)
        
        self.C = C_full[mask, :][:, mask]
        
        # Decomposição de Cholesky
        self._prepare_covariance()
    
    def _prepare_covariance(self):
        """Prepara matriz de covariância (Cholesky)"""
        try:
            self.C_cholesky = la.cholesky(self.C, lower=True)
        except la.LinAlgError:
            # Regularização mínima
            eigvals = la.eigvalsh(self.C)
            min_eig = np.min(eigvals)
            if min_eig < 0:
                self.C += np.eye(self.n_sne) * (abs(min_eig) + 1e-10)
                self.C_cholesky = la.cholesky(self.C, lower=True)

    def _precompute_static(self):
        """Pré-calcula fatores constantes"""
        # Fatores (1+z_cmb)*(1+z_hel)
        self.z_factor = (1.0 + self.zHD) * (1.0 + self.zHEL)

    # --------------------------------------------------
    # CÁLCULO DA(z) COM CACHE
    # --------------------------------------------------
    def _get_DA(self, cosmo, z):
        if not self.cache_DA:
            return cosmo.DA(z)
        try:
            if hasattr(cosmo, 'get_params'):
                cosmo_hash = hash(tuple(cosmo.get_params()))
            elif hasattr(cosmo, 'params'):
                cosmo_hash = hash(tuple(sorted(cosmo.params.items())))
            else:
                return cosmo.DA(z)
        except (AttributeError, TypeError):
            return cosmo.DA(z)
        
        if cosmo_hash != self._last_cosmo_hash:
            self._DA_cache.clear()
            self._last_cosmo_hash = cosmo_hash
        
        # Usar arredondamento para evitar problemas numéricos
        z_key = tuple(np.round(z, 6))
        
        if z_key in self._DA_cache:
            return self._DA_cache[z_key]
        else:
            DA = cosmo.DA(z).astype(np.float32)
            self._DA_cache[z_key] = DA
            return DA

    # --------------------------------------------------
    # CÁLCULO TEÓRICO
    # --------------------------------------------------
    def calculate_mu_theory(self, cosmo):
        """
        Calcula vetor teórico de módulos de distância.
        
        Parameters
        ----------
        cosmo : objeto cosmologia
            Deve ter método DA(z) que retorna distância angular em Mpc
            
        Returns
        -------
        mu_theory : array
            Vetor de módulos de distância teóricos
        """
        # Obter distância angular (com cache se habilitado)
        DA = self._get_DA(cosmo, self.zHD)
        
        # Cálculo vetorizado otimizado
        log_arg = self.z_factor * DA
        
        if hasattr(log_arg, '__array_interface__'):
            np.log10(log_arg, out=log_arg)
        else:
            log_arg = np.log10(log_arg)
        
        log_arg *= 5.0
        log_arg += 25.0
        
        self._mu_theory[:] = log_arg
        return self._mu_theory

    # --------------------------------------------------
    # FUNÇÃO DE VEROSSIMILHANÇA - M_B É OBRIGATÓRIO
    # --------------------------------------------------
    def loglkl(self, cosmo, M_B):
        """
        Calcula o logaritmo da verossimilhança.
        """
        # M_B é OBRIGATÓRIO
        mu_theo = self.calculate_mu_theory(cosmo)
        
        self._residuals[:] = self.m_b_corr  
        self._residuals -= mu_theo          
        self._residuals -= M_B            
        
        r = la.solve_triangular(
            self.C_cholesky,
            self._residuals,
            lower=True,
            check_finite=False,
            overwrite_b=True
        )
        
        chi2 = np.dot(r, r)
        logL = -0.5 * chi2
        
        return logL, chi2

    # --------------------------------------------------
    # MÉTODOS AUXILIARES
    # --------------------------------------------------
    def get_residuals(self, cosmo, M_B):
        """
        Retorna os resíduos para análise.
        
        Returns
        -------
        dict
            Dicionário com arrays de resíduos e informações
        """
        mu_theo = self.calculate_mu_theory(cosmo)
        residuals = self.m_b_corr - (mu_theo + M_B)
        
        return {
            'residuals': residuals,
            'mu_theory': mu_theo,
            'zHD': self.zHD,
            'zHEL': self.zHEL,
            'm_b_corr': self.m_b_corr,
            'M_B': M_B,
        }

    def batch_evaluate(self, cosmo, M_B_values):
        # Calcular vetor teórico uma vez
        mu_theo = self.calculate_mu_theory(cosmo)
        
        # Pré-calcular m_obs - μ
        m_minus_mu = self.m_b_corr - mu_theo
        
        results = []
        for M_B in M_B_values:
            # Resíduos para este M_B
            self._residuals[:] = m_minus_mu
            self._residuals -= M_B
            
            # Solver triangular
            r = la.solve_triangular(
                self.C_cholesky,
                self._residuals,
                lower=True,
                check_finite=False,
                overwrite_b=True
            )
            
            chi2 = np.dot(r, r)
            results.append((-0.5 * chi2, chi2))
        
        return results

    def get_data(self):
        """
        Retorna dados observacionais para plotagem.
        
        Returns
        -------
        tuple
            (zHD, m_b_corr, m_b_err)
        """
        # Obter erros da diagonal da matriz de covariância
        m_b_err = np.sqrt(np.diag(self.C))
        
        return self.zHD, self.m_b_corr, m_b_err

    def loglkl_from_params(self, cosmo, params):
        """
        Versão alternativa que aceita dicionário de parâmetros.
        """
        if 'M_B' not in params:
            raise KeyError("Parâmetro 'M_B' não encontrado no dicionário de parâmetros")
        
        M_B = params['M_B']
        return self.loglkl(cosmo, M_B)
