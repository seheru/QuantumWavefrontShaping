import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL & EXPERIMENTAL PARAMETERS
# ==========================================
lambda_p = 405e-9        # Pump wavelength [m] (405 nm)
k_p = 2 * np.pi / lambda_p  # Pump wavenumber [rad/m]
w_0 = 120e-6             # Pump beam waist radius [m] (120 um)
L = 2.0e-3               # Crystal length [m] (2 mm)
gamma = 1.776            # Gaussian-sinc matching factor

# Double Gaussian Bandwidths
sigma_plus = 1.0 / w_0                     # Pump momentum width [rad/m]
sigma_minus = np.sqrt(2 * k_p / (gamma * L)) # Phase-matching bandwidth [rad/m]

# Analytical Schmidt Number (Spatial Entanglement measure)
K_schmidt = 0.5 * (sigma_minus / sigma_plus + sigma_plus / sigma_minus)

print(f"--- SPDC Double Gaussian Parameters ---")
print(f"Pump Waist (w0)        : {w_0*1e6:.1f} um")
print(f"Crystal Length (L)     : {L*1e3:.1f} mm")
print(f"sigma_+ (Center-of-mass): {sigma_plus*1e-3:.2f} mrad^-1")
print(f"sigma_- (Difference)    : {sigma_minus*1e-3:.2f} mrad^-1")
print(f"Schmidt Number (K)      : {K_schmidt:.2f} (High continuous entanglement)")

# ==========================================
# 2. WAVEFUNCTIONS & DISTRIBUTIONS
# ==========================================
# Spatial Momentum Grid (q in rad/um)
N_pts = 400
q_max = 0.15 # rad/um
q_vec = np.linspace(-q_max, q_max, N_pts) * 1e6 # in rad/m
Qs, Qi = np.meshgrid(q_vec, q_vec)

# Transverse Momentum Wavefunction Psi(qs, qi)
alpha_q = np.exp(- (Qs + Qi)**2 / (4 * sigma_plus**2))
phi_q   = np.exp(- (Qs - Qi)**2 / (4 * sigma_minus**2))
Psi_q   = alpha_q * phi_q
JMD     = np.abs(Psi_q)**2 # Joint Momentum Distribution
JMD    /= np.max(JMD)

# Position Grid (x in um)
x_max = 200 # um
x_vec = np.linspace(-x_max, x_max, N_pts) * 1e-6 # in m
Xs, Xi = np.meshgrid(x_vec, x_vec)

# Transverse Position Wavefunction Psi(xs, xi)
alpha_x = np.exp(- (sigma_plus**2) * ((Xs + Xi)**2) / 4.0)
phi_x   = np.exp(- (sigma_minus**2) * ((Xs - Xi)**2) / 4.0)
Psi_x   = alpha_x * phi_x
JPD     = np.abs(Psi_x)**2 # Joint Position Distribution
JPD    /= np.max(JPD)

# ==========================================
# 3. NUMERICAL SVD FOR ENTANGLEMENT
# ==========================================
U, S, Vh = np.linalg.svd(Psi_q)
norm_S = S**2 / np.sum(S**2)
K_num = 1.0 / np.sum(norm_S**2)

# ==========================================
# 4. PLOTTING THE RESULTS
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(13, 11))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 1. Joint Momentum Distribution (JMD)
im0 = axs[0, 0].imshow(JMD, origin='lower', cmap='plasma',
                       extent=[-q_max, q_max, -q_max, q_max])
axs[0, 0].set_title(r"$\mathbf{(a)\ Joint\ Momentum\ Distribution\ | \Psi(q_s, q_i) |^2}$", fontsize=12)
axs[0, 0].set_xlabel(r"Signal Momentum $q_s\ [\mathrm{rad/\mu m}]$", fontsize=11)
axs[0, 0].set_ylabel(r"Idler Momentum $q_i\ [\mathrm{rad/\mu m}]$", fontsize=11)
axs[0, 0].axline((0, 0), slope=-1, color='cyan', linestyle='--', alpha=0.6, label=r"$q_s = -q_i$ (Anti-correlation)")
axs[0, 0].legend(loc='upper right', fontsize=9)
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04, label="Normalized Intensity")

# 2. Joint Position Distribution (JPD)
im1 = axs[0, 1].imshow(JPD, origin='lower', cmap='viridis',
                       extent=[-x_max, x_max, -x_max, x_max])
axs[0, 1].set_title(r"$\mathbf{(b)\ Joint\ Position\ Distribution\ | \Psi(x_s, x_i) |^2}$", fontsize=12)
axs[0, 1].set_xlabel(r"Signal Position $x_s\ [\mathrm{\mu m}]$", fontsize=11)
axs[0, 1].set_ylabel(r"Idler Position $x_i\ [\mathrm{\mu m}]$", fontsize=11)
axs[0, 1].axline((0, 0), slope=1, color='yellow', linestyle='--', alpha=0.6, label=r"$x_s = x_i$ (Coincidence)")
axs[0, 1].legend(loc='upper left', fontsize=9)
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, label="Normalized Intensity")

# 3. Momentum Cross-Sections
mid_idx = N_pts // 2
diag_q = np.diagonal(np.fliplr(JMD)) # along qs = -qi
axs[1, 0].plot(q_vec * 1e-6, JMD[mid_idx, :], 'r-', lw=2, label=r"Conditional $P(q_s | q_i=0)$")
axs[1, 0].plot(np.linspace(-q_max, q_max, N_pts), diag_q, 'b--', lw=2, label=r"Anti-diagonal $q_s = -q_i$")
axs[1, 0].set_title(r"$\mathbf{(c)\ Momentum\ Profiles}$", fontsize=12)
axs[1, 0].set_xlabel(r"$q_s\ [\mathrm{rad/\mu m}]$", fontsize=11)
axs[1, 0].set_ylabel("Intensity [a.u.]", fontsize=11)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend(fontsize=9)

# 4. Position Cross-Sections
diag_x = np.diagonal(JPD) # along xs = xi
axs[1, 1].plot(x_vec * 1e6, JPD[mid_idx, :], 'g-', lw=2, label=r"Conditional $P(x_s | x_i=0)$")
axs[1, 1].plot(np.linspace(-x_max, x_max, N_pts), diag_x, 'm--', lw=2, label=r"Coincidence Diagonal $x_s = x_i$")
axs[1, 1].set_title(r"$\mathbf{(d)\ Position\ Profiles}$", fontsize=12)
axs[1, 1].set_xlabel(r"$x_s\ [\mathrm{\mu m}]$", fontsize=11)
axs[1, 1].set_ylabel("Intensity [a.u.]", fontsize=11)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend(fontsize=9)

plt.suptitle(f"SPDC Transverse Spatial Entanglement (Double Gaussian Model)\nAnalytical Schmidt Number $K = {K_schmidt:.1f}$ | SVD Numerical $K = {K_num:.1f}$", fontsize=14, y=0.98)
plt.show()