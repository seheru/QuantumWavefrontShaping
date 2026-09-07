import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL CONSTANTS & EXPERIMENTAL INPUTS
# ==========================================
lambda_p = 405e-9          # Pump wavelength [m] (405 nm)
k_p = 2 * np.pi / lambda_p    # Pump wavenumber [rad/m]
w_0 = 100e-6               # Pump beam waist radius [m] (100 um)
L = 2.0e-3                 # Crystal length [m] (2 mm)

# -------------------------------------------------------------
# Spatial Parameters (for the 8 and 16 denominators)
# -------------------------------------------------------------
sigma_pump = w_0 / np.sqrt(2)         # Pump beam spatial variance
sigma_corr = np.sqrt(L / (8 * k_p))    # Microscopic correlation width

# Dual Momentum Parameters
Omega_pump = 1.0 / (4 * sigma_pump)   # Pump momentum variance
Omega_corr = 1.0 / (4 * sigma_corr)   # Phase-matching momentum bandwidth

# Schmidt Number (Quantifies continuous spatial entanglement)
K_schmidt = 0.5 * (sigma_pump / (2 * sigma_corr) + (2 * sigma_corr) / sigma_pump)

print(f"--- Double Gaussian Model with 8 and 16 Denominators ---")
print(f"sigma_pump (w0 / sqrt(2)) : {sigma_pump * 1e6:.2f} um")
print(f"sigma_corr (sqrt(L/8kp))  : {sigma_corr * 1e6:.2f} um")
print(f"Schmidt Number (K)        : {K_schmidt:.2f}")

# ==========================================
# 2. WAVEFUNCTIONS (EXACT ANALYTICAL EQUATIONS)
# ==========================================
N_pts = 400

# -----------------
# 1. POSITION SPACE
# -----------------
x_max = 160e-6  # +/- 160 um
x_vec = np.linspace(-x_max, x_max, N_pts)
Xs, Xi = np.meshgrid(x_vec, x_vec)

# YOUR EXACT FORMULA:
# Psi(xs, xi) = exp[ - (xs + xi)^2 / (8 * sigma_pump^2) ] * exp[ - (xs - xi)^2 / (16 * sigma_corr^2) ]
Psi_x = np.exp(- (Xs + Xi)**2 / (8 * sigma_pump**2)) * \
        np.exp(- (Xs - Xi)**2 / (16 * sigma_corr**2))

JPD = np.abs(Psi_x)**2
JPD /= np.max(JPD)

# -----------------
# 2. MOMENTUM SPACE
# -----------------
q_max = 0.15e6  # +/- 0.15 rad/um
q_vec = np.linspace(-q_max, q_max, N_pts)
Qs, Qi = np.meshgrid(q_vec, q_vec)

# Dual Momentum Formula:
# Psi(qs, qi) = exp[ - 2 * sigma_pump^2 * (qs + qi)^2 ] * exp[ - 4 * sigma_corr^2 * (qs - qi)^2 ]
Psi_q = np.exp(- 2 * (sigma_pump**2) * (Qs + Qi)**2) * \
        np.exp(- 4 * (sigma_corr**2) * (Qs - Qi)**2)

JMD = np.abs(Psi_q)**2
JMD /= np.max(JMD)

# ==========================================
# 3. VISUALIZATION
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(13, 11))
plt.subplots_adjust(hspace=0.34, wspace=0.3)

x_scale = 1e6   # m to um
q_scale = 1e-6  # rad/m to rad/um

# Plot 1: Position Space (JPD)
im0 = axs[0, 0].imshow(JPD, origin='lower', cmap='viridis',
                       extent=[-x_max*x_scale, x_max*x_scale, -x_max*x_scale, x_max*x_scale])
axs[0, 0].set_title(r"$\mathbf{Position\ Space\ |\Psi(x_s, x_i)|^2}$" + "\n" +
                    r"$\Psi(x_s, x_i) = C_x \exp\left[-\frac{(x_s+x_i)^2}{\mathbf{8}\sigma_p^2}\right] \exp\left[-\frac{(x_s-x_i)^2}{\mathbf{16}\sigma_c^2}\right]$", fontsize=11)
axs[0, 0].set_xlabel(r"Signal Position $x_s\ [\mathrm{\mu m}]$", fontsize=10)
axs[0, 0].set_ylabel(r"Idler Position $x_i\ [\mathrm{\mu m}]$", fontsize=10)
axs[0, 0].axline((0, 0), slope=1, color='yellow', linestyle='--', alpha=0.7, label=r"$x_s = x_i$ (Coincidence)")
axs[0, 0].legend(loc='upper left', fontsize=8)
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04, label="Intensity")

# Plot 2: Momentum Space (JMD)
im1 = axs[0, 1].imshow(JMD, origin='lower', cmap='inferno',
                       extent=[-q_max*q_scale, q_max*q_scale, -q_max*q_scale, q_max*q_scale])
axs[0, 1].set_title(r"$\mathbf{Momentum\ Space\ |\Psi(q_s, q_i)|^2}$" + "\n" +
                    r"$\Psi(q_s, q_i) = C_q \exp\left[-2\sigma_p^2(q_s+q_i)^2\right] \exp\left[-4\sigma_c^2(q_s-q_i)^2\right]$", fontsize=11)
axs[0, 1].set_xlabel(r"Signal Momentum $q_s\ [\mathrm{rad/\mu m}]$", fontsize=10)
axs[0, 1].set_ylabel(r"Idler Momentum $q_i\ [\mathrm{rad/\mu m}]$", fontsize=10)
axs[0, 1].axline((0, 0), slope=-1, color='cyan', linestyle='--', alpha=0.7, label=r"$q_s = -q_i$ (Anti-correlation)")
axs[0, 1].legend(loc='upper right', fontsize=8)
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, label="Intensity")

# Plot 3: Position Cross-Sections
mid_idx = N_pts // 2
diag_coinc_x = np.diagonal(JPD)

axs[1, 0].plot(x_vec * x_scale, JPD[mid_idx, :], 'g-', lw=2, label=r"Conditional $P(x_s | x_i=0)$ (Width $\propto \sigma_c$)")
axs[1, 0].plot(x_vec * x_scale, diag_coinc_x, 'm--', lw=2, label=r"Coincidence $x_s = x_i$ (Width $\propto \sigma_p$)")
axs[1, 0].set_title(r"$\mathbf{Position\ Profiles}$", fontsize=11)
axs[1, 0].set_xlabel(r"Position $x\ [\mathrm{\mu m}]$", fontsize=10)
axs[1, 0].set_ylabel("Intensity", fontsize=10)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend(fontsize=8)

# Plot 4: Momentum Cross-Sections
diag_anti_q = np.diagonal(np.fliplr(JMD))

axs[1, 1].plot(q_vec * q_scale, JMD[mid_idx, :], 'r-', lw=2, label=r"Conditional $P(q_s | q_i=0)$")
axs[1, 1].plot(q_vec * q_scale, diag_anti_q, 'b--', lw=2, label=r"Anti-diagonal $q_s = -q_i$")
axs[1, 1].set_title(r"$\mathbf{Momentum\ Profiles}$", fontsize=11)
axs[1, 1].set_xlabel(r"Momentum $q\ [\mathrm{rad/\mu m}]$", fontsize=10)
axs[1, 1].set_ylabel("Intensity", fontsize=10)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend(fontsize=8)

plt.suptitle(f"Continuous-Variable EPR Entanglement in SPDC\nSchmidt Number $K = {K_schmidt:.2f}$", fontsize=13, y=0.98)
plt.show()