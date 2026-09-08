import numpy as np
import matplotlib.pyplot as plt
from SERTAO.neutrinos import create_neutrinos

# ============================================================
# 1. Criar algumas configurações de neutrinos
# ============================================================
print("=== TESTANDO DIFERENTES CONFIGURAÇÕES ===")
configs = [
    ('Standard (0.06 eV)', {'config': 'standard', 'total_mass': 0.06}),
    ('1 NCDM (0.06 eV)', {'config': '1ncdm', 'm_ncdm': 0.06}),
    ('Massa zero', {'config': 'standard', 'total_mass': 0.0}),  # Massa zero corretamente
]

neutrino_instances = []
for name, params in configs:
    nu = create_neutrinos(**params)
    neutrino_instances.append((name, nu))
    print(f"\n{name}:")
    print(f"  Massas: {nu.masses} eV")
    print(f"  Σm_ν: {nu.total_mass_eV:.3f} eV")
    print(f"  ρ_ν0/ρ_crit0: {nu.Omega_nu0:.6e}")

# ============================================================
# 2. Plot básico: Densidade vs Redshift
# ============================================================
print("\n=== GERANDO GRÁFICOS ===")
z = np.logspace(-2, 3, 200)  # Redshift de 0.01 a 1000

plt.figure(figsize=(12, 4))

# Subplot 1: ρ_ν(z)/ρ_crit0
plt.subplot(1, 2, 1)
for name, nu in neutrino_instances:
    density = nu.density_parameter(z)
    plt.loglog(z, density, linewidth=2, label=name)

plt.xlabel('Redshift (z)')
plt.ylabel('ρ_ν(z)/ρ_crit0')
plt.title('Densidade de energia dos neutrinos')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Equação de estado w(z)
plt.subplot(1, 2, 2)
for name, nu in neutrino_instances:
    w = nu.equation_of_state(z)
    plt.semilogx(z, w, linewidth=2, label=name)

plt.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='w=1/3')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='w=0')
plt.xlabel('Redshift (z)')
plt.ylabel('w(z)')
plt.title('Equação de estado')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 3. Teste simples de conservação de energia
# ============================================================
print("\n=== TESTE SIMPLES DE CONSERVAÇÃO ===")
nu_test = create_neutrinos(config='standard', total_mass=0.06)

z_test = np.array([0, 1, 10, 100, 1000])
density_test = nu_test.density_parameter(z_test)
w_test = nu_test.equation_of_state(z_test)

print(f"\nConfiguração: {nu_test.description}")
print("Redshift | ρ_ν(z)/ρ_crit0 | w(z)     | Comportamento")
print("-" * 50)

for z_val, rho_val, w_val in zip(z_test, density_test, w_test):
    if w_val > 0.3:
        behavior = "Relativístico"
    elif w_val < 0.1:
        behavior = "Não-relativístico"
    else:
        behavior = "Transição"
    
    print(f"z={z_val:4.0f} | {rho_val:.2e} | {w_val:.4f}  | {behavior}")

# ============================================================
# 4. Teste do limite de altos redshifts
# ============================================================
print("\n=== TESTE DE ALTOS REDSHIFTS (z→∞) ===")
if nu_test.total_mass_eV > 0:
    print("Para neutrinos massivos em z alto:")
    print("w(z) deve se aproximar de 1/3 ≈ 0.3333")
    print(f"w(z=1000) = {w_test[-1]:.6f}")
    print(f"Diferença: {abs(w_test[-1] - 1/3):.2e}")
    
    if abs(w_test[-1] - 1/3) < 0.01:
        print("✓ Comportamento correto em altos redshifts")
    else:
        print("✗ Possível problema")

# ============================================================
# 5. Teste da contribuição para H²(z)/H₀²
# ============================================================
print("\n=== TESTE DE H²(z)/H₀² ===")
print("Verificando que contribution_to_H2(z) = density_parameter(z)")
z_check = np.array([0, 1, 10, 100])
for z_val in z_check:
    h2_contrib = nu_test.contribution_to_H2(z_val)
    density = nu_test.density_parameter(z_val)
    diff = abs(h2_contrib - density) / density
    print(f"z={z_val:3}: H²_contrib={h2_contrib:.2e}, density={density:.2e}, "
          f"diferença relativa={diff:.2e}")

# ============================================================
# 6. Verificação de ordem de grandeza comparando com matéria
# ============================================================
print("\n=== VERIFICAÇÃO DE ORDEM DE GRANDEZA ===")
print("Comparando com matéria (Ω_m=0.3):")

Ω_m0 = 0.3  # Valor de referência para matéria
for z_val in z_test:
    rho_nu = nu_test.density_parameter(z_val)
    rho_m = Ω_m0 * (1 + z_val)**3
    
    ratio = rho_nu / rho_m * 100  # Percentual
    
    print(f"z={z_val:4}: ρ_ν/ρ_crit0={rho_nu:.2e}, "
          f"ρ_m/ρ_crit0={rho_m:.2e}, "
          f"ν/matéria={ratio:.1f}%")

print("\n=== ANÁLISE CONCLUÍDA ===")






