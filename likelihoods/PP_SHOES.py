# ============================================================
# Pantheon+&SHOES likelihood
# arXiv:2112.04510
# ============================================================

import numpy as np
import scipy.linalg as la
import pandas as pd
import os


class PantheonPlusSHOESLikelihood:
    """
    Pantheon+ + SH0ES likelihood
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

        self._load_data()
        
        self._filter_data()
        
        self._load_covariance()
        self._prepare_covariance()
        
        self._precompute_static()
        
        self._mu_theory = np.empty(self.n_sne, dtype=np.float32)
        self._residuals = np.empty(self.n_sne, dtype=np.float32)

    # --------------------------------------------------
    # CARREGAMENTO DE DADOS
    # --------------------------------------------------
    def _load_data(self):
        """Carrega o arquivo de dados Pantheon+SH0ES de forma otimizada"""
        data_file = os.path.join(self.data_dir, 'Pantheon+SH0ES.dat')
        
        # Ler como array numpy diretamente
        self.data = pd.read_csv(data_file, sep=r'\s+', comment='#').values
        
        # Extrair colunas como arrays numpy de uma vez
        self.zCMB_all = self.data[:, 2].astype(np.float32)      # zHD (índice 2)
        self.zHEL_all = self.data[:, 6].astype(np.float32)      # zHEL (índice 6)
        self.m_b_corr_all = self.data[:, 8].astype(np.float32)  # m_b_corr (índice 8)
        self.is_calibrator_all = self.data[:, 13].astype(bool)  # IS_CALIBRATOR (índice 13)
        self.cepheid_distance_all = self.data[:, 12].astype(np.float32)  # CEPH_DIST (índice 12)
        
        self.origlen = len(self.data)

    def _filter_data(self):
        mask = (self.zCMB_all > self.z_min) | self.is_calibrator_all
        
        self.zCMB = self.zCMB_all[mask]
        self.zHEL = self.zHEL_all[mask]
        self.m_b_corr = self.m_b_corr_all[mask]
        self.is_calibrator = self.is_calibrator_all[mask]
        self.cepheid_distance = self.cepheid_distance_all[mask]
        
        self.n_calibrators = np.sum(self.is_calibrator)
        self.n_hubble = np.sum(~self.is_calibrator)
        self.n_sne = len(self.zCMB)
        
        self.idx_calibrator = np.where(self.is_calibrator)[0]
        self.idx_hubble = np.where(~self.is_calibrator)[0]
        
        self.mask = mask

    def _precompute_static(self):
        if self.n_hubble > 0:
            self.z_factor_hubble = (1.0 + self.zCMB[self.idx_hubble]) * (1.0 + self.zHEL[self.idx_hubble])
        if self.n_calibrators > 0:
            self._mu_calibrators = self.cepheid_distance[self.idx_calibrator]

    # --------------------------------------------------
    # MATRIZ DE COVARIÂNCIA
    # --------------------------------------------------
    def _load_covariance(self):
        """
        Carrega a matriz de covariância de forma otimizada.
        """
        if self.use_sys_cov:
            cov_file = os.path.join(self.data_dir, 'Pantheon+SH0ES_STAT+SYS.cov')
        else:
            cov_file = os.path.join(self.data_dir, 'Pantheon+SH0ES_STATONLY.cov')
        
        with open(cov_file, 'r') as f:
            lines = f.readlines()
        
        n_total = int(lines[0].strip())
        
        if n_total != self.origlen:
            raise ValueError(
                f"Tamanho da matriz de covariância ({n_total}) não corresponde "
                f"ao número de SNs no arquivo de dados ({self.origlen})"
            )
        
        data_lines = lines[1:]
        all_values = np.fromstring(' '.join(data_lines), sep=' ', dtype=np.float64)
        
        C_full = all_values.reshape(n_total, n_total)
        
        self.C = C_full[self.mask, :][:, self.mask]

    def _prepare_covariance(self):
        """
        Prepara a matriz de covariância para uso eficiente.
        """
        try:
            # Decomposição de Cholesky
            self.C_cholesky = la.cholesky(self.C, lower=True)
        except la.LinAlgError:
            # Matriz não positiva definida - regularizar
            eigvals = la.eigvalsh(self.C)
            min_eig = np.min(eigvals)
            
            if min_eig < 0:
                regularization = abs(min_eig) + 1e-10
                self.C += np.eye(self.n_sne) * regularization
                self.C_cholesky = la.cholesky(self.C, lower=True)
            else:
                raise RuntimeError(
                    "Matriz de covariância não positiva definida, "
                    "mas todos autovalores são positivos"
                )

    # --------------------------------------------------
    # CÁLCULO DO VETOR TEÓRICO
    # --------------------------------------------------
    def _get_DA(self, cosmo, z):
        """Obtém DA(z) com cache opcional para otimização"""
        if not self.cache_DA:
            return cosmo.DA(z)
        
        # Criar hash da cosmologia atual
        try:
            # Tentar obter parâmetros da cosmologia
            if hasattr(cosmo, 'get_params'):
                cosmo_hash = hash(tuple(cosmo.get_params()))
            elif hasattr(cosmo, 'params'):
                cosmo_hash = hash(tuple(sorted(cosmo.params.items())))
            else:
                # Se não conseguir hash, não usar cache
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

    def calculate_mu_theory(self, cosmo):
        """
        Calcula o vetor teórico de módulos de distância.
        """
        # --- Calibradores Cefeidas (constante) ---
        if self.n_calibrators > 0:
            self._mu_theory[self.idx_calibrator] = self._mu_calibrators
        
        # --- Hubble flow ---
        if self.n_hubble > 0:
            z_hubble = self.zCMB[self.idx_hubble]
            
            # Obter distância angular
            DA = self._get_DA(cosmo, z_hubble)
            
            # Cálculo vetorizado: μ = 5·log₁₀[(1+z_cmb)(1+z_hel)·D_A] + 25
            log_arg = self.z_factor_hubble * DA
            
            if hasattr(log_arg, '__array_interface__'):
                np.log10(log_arg, out=log_arg)
            else:
                log_arg = np.log10(log_arg)
            
            log_arg *= 5.0
            log_arg += 25.0
            
            self._mu_theory[self.idx_hubble] = log_arg
        
        return self._mu_theory

    # --------------------------------------------------
    # FUNÇÃO DE VEROSSIMILHANÇA
    # --------------------------------------------------
    def loglkl(self, cosmo, M_B):
        """
        Calcula o logaritmo da verossimilhança.
        """
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

    def get_residuals(self, cosmo, M_B):
        """
        Retorna os resíduos para análise.
        """
        # M_B é obrigatório aqui também
        mu_theo = self.calculate_mu_theory(cosmo)
        residuals = self.m_b_corr - (mu_theo + M_B)
        
        return {
            'residuals': residuals,
            'mu_theory': mu_theo,
            'zCMB': self.zCMB,
            'zHEL': self.zHEL,
            'is_calibrator': self.is_calibrator,
            'm_b_corr': self.m_b_corr,
            'M_B': M_B,  # Incluir M_B usado
        }

    def batch_evaluate(self, cosmo, M_B_values):
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

    # --------------------------------------------------
    # MÉTODO COMPATÍVEL PARA USO COM DICIONÁRIO DE PARÂMETROS
    # --------------------------------------------------
    def loglkl_from_params(self, cosmo, params):
        if 'M_B' not in params:
            raise KeyError("Parâmetro 'M_B' não encontrado no dicionário de parâmetros") 
        M_B = params['M_B']
        return self.loglkl(cosmo, M_B)

