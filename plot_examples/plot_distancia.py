import numpy as np
import matplotlib.pyplot as plt

from SERTAO.cosmology import GenericCosmology 

# --------------------------------------------------
# Definir modelo cosmológico (exemplo LCDM)
# --------------------------------------------------

H0 = 70.0
Omega_m = 0.3
Omega_b = 0.05

def H_LCDM(z):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

cosmo = GenericCosmology(
    H_of_z=H_LCDM,
    H0=H0,
    Omega_m=Omega_m,
    Omega_b=Omega_b
)

# --------------------------------------------------
# Redshift array
# --------------------------------------------------

z = np.linspace(0.0, 2.0, 200)

# --------------------------------------------------
# Distâncias
# --------------------------------------------------

Dc = cosmo.comoving_distance(z)
Dl = cosmo.DL(z)
Da = cosmo.DA(z)

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure()
plt.plot(z, Dc, label=r"$D_C(z)$")
plt.plot(z, Dl, label=r"$D_L(z)$")
plt.plot(z, Da, label=r"$D_A(z)$")

plt.xlabel(r"$z$")
plt.ylabel(r"Distance [Mpc]")
plt.legend()
plt.grid(True)

plt.show()

