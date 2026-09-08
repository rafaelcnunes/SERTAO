import numpy as np
import matplotlib.pyplot as plt

from SERTAO.cosmology import GenericCosmology 

H0 = 67.4
Omega_m = 0.315
Omega_b = 0.049

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
z = np.linspace(0.1, 2.5, 100)

# --------------------------------------------------
#BAO
# --------------------------------------------------
DM = cosmo.DM(z)
DH = cosmo.DH(z)
DV = cosmo.DV(z)
rd = cosmo.rd_sound_horizon()

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure()
plt.plot(z, DM / rd, label=r"$D_M(z)/r_d$")
plt.plot(z, DH / rd, label=r"$D_H(z)/r_d$")
plt.plot(z, DV / rd, label=r"$D_V(z)/r_d$")

plt.xlabel(r"$z$")
plt.ylabel(r"Distance / $r_d$")
plt.legend()
plt.grid(True)

plt.show()
# --------------------------------------------------
# Print
# --------------------------------------------------
print("=" * 50)
print(f"Sound horizon r_d = {rd:.2f} Mpc")
print("=" * 50)

