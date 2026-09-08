import numpy as np

# ============================================================
# DESI 2025 DR2 data 
# arXiv:2503.14738
# ============================================================

DESI_DATA = [
    # BGS (isotropic) - z=0.295
    {'z': 0.295, 'Dv_rd': 7.93, 'Dv_rd_err': 0.15, 'type': 'isotropic'},
    
    # LRG1 (anisotropic) - z=0.510
    {'z': 0.510, 'DM_rd': 13.62, 'DM_rd_err': 0.25,
     'DH_rd': 20.98, 'DH_rd_err': 0.61, 'corr': -0.445, 'type': 'anisotropic'},
    
    # LRG2 (anisotropic) - z=0.706
    {'z': 0.706, 'DM_rd': 16.85, 'DM_rd_err': 0.32,
     'DH_rd': 20.08, 'DH_rd_err': 0.60, 'corr': -0.420, 'type': 'anisotropic'},
    
    # LRG3+ELG1 (anisotropic) - z=0.930
    {'z': 0.930, 'DM_rd': 21.71, 'DM_rd_err': 0.28,
     'DH_rd': 17.88, 'DH_rd_err': 0.35, 'corr': -0.389, 'type': 'anisotropic'},
    
    # ELG2 (anisotropic) - z=1.317
    {'z': 1.317, 'DM_rd': 27.79, 'DM_rd_err': 0.69,
     'DH_rd': 13.82, 'DH_rd_err': 0.42, 'corr': -0.444, 'type': 'anisotropic'},
    
    # QSO (isotropic) - z=1.491
    {'z': 1.491, 'Dv_rd': 26.07, 'Dv_rd_err': 0.67, 'type': 'isotropic'},
    
    # Lya QSO (anisotropic) - z=2.330
    {'z': 2.330, 'DM_rd': 39.71, 'DM_rd_err': 0.94,
     'DH_rd': 8.52, 'DH_rd_err': 0.17, 'corr': -0.477, 'type': 'anisotropic'}
]

def desi_likelihood(cosmo):
    """
    Calculate log-likelihood
    """
    
    rd = cosmo.rd_sound_horizon()
    chi2_total = 0.0
    
    for data_point in DESI_DATA:
        z = data_point['z']
        
        # Theoretical quantities
        DM_theo = cosmo.DM(z)  # Comoving angular diameter distance
        DH_theo = cosmo.DH(z)  # Hubble distance
        Dv_theo = (z * DM_theo**2 * DH_theo)**(1/3)  # Dilatation distance
        
        if data_point['type'] == 'isotropic':
            # Isotropic measurement: Dv/rd
            Dv_rd_theo = Dv_theo / rd
            Dv_rd_obs = data_point['Dv_rd']
            Dv_rd_err = data_point['Dv_rd_err']
            
            chi2_i = ((Dv_rd_theo - Dv_rd_obs) / Dv_rd_err)**2
            
        else:  # anisotropic
            # Anisotropic measurements: DM/rd and DH/rd
            DM_rd_theo = DM_theo / rd
            DH_rd_theo = DH_theo / rd
            
            DM_rd_obs = data_point['DM_rd']
            DH_rd_obs = data_point['DH_rd']
            DM_rd_err = data_point['DM_rd_err']
            DH_rd_err = data_point['DH_rd_err']
            rho = data_point['corr']
        
            delta_DM = DM_rd_theo - DM_rd_obs
            delta_DH = DH_rd_theo - DH_rd_obs
            
            chi2_i = (delta_DM**2 / DM_rd_err**2 +
                     delta_DH**2 / DH_rd_err**2 -
                     2 * rho * delta_DM * delta_DH / (DM_rd_err * DH_rd_err)) / (1 - rho**2)
        
        chi2_total += chi2_i
    
    logL = -0.5 * chi2_total
    return logL, chi2_total
