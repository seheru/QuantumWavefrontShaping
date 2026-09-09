import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL & EXPERIMENTAL PARAMETERS
# ==========================================
lambda_p = 405e-9          # Pump wavelength [m] (405 nm)
k_p = 2 * np.pi / lambda_p    # Pump wavenumber [rad/m]
w_0 = 100e-6               # Pump beam waist radius [m] (100 um)
L = 1.5e-3                 # Crystal length [m] (1.5 mm)

print(f"--- Sinc-Gauss SPDC Parameters ---")
print(f"Pump Wavelength (lambda_p) : {lambda_p*1e9:.1f} nm")
print(f"Pump Waist (w0)           : {w_0*1e6:.1f} um")
print(f"Crystal Length (L)        : {L*1e3:.2f} mm")

# ==========================================
# 2. MOMENTUM GRID SETUP & WAVEFUNCTION
# ==========================================
N_pts = 512                # FFT grid resolution (power of 2)
q_max = 0.18e6             # Maximum transverse momentum [rad/m] (0.18 rad/um)

q_vec = np.linspace(-q_max, q_max, N_pts)
dq = q_vec[1] - q_vec[0]
Qs, Qi = np.meshgrid(q_vec, q_vec)

# 1. Pump Envelope Function (Gaussian)
alpha_q = np.exp(- (w_0**2 * (Qs + Qi)**2) / 4.0)

# 2. Longitudinal Phase Mismatch (Paraxial: Delta_kz = (qs^2 + qi^2) / kp)
delta_kz = (Qs**2 + Qi**2) / k_p

# 3. Exact Sinc Phase-Matching Function (with phase factor)
# np.sinc(x) calculates sin(pi*x) / (pi*x)
arg_sinc = (delta_kz * L) / (2.0 * np.pi)
phi_q = np.sinc(arg_sinc) * np.exp(1j * delta_kz * L / 2.0)

# Joint Momentum Wavefunction Psi(qs, qi)
Psi_q = alpha_q * phi_q
JMD = np.abs(Psi_q)**2
JMD /= np.max(JMD)

# ==========================================
# 3. NUMERICAL 2D-IFFT TO POSITION SPACE
# ==========================================
# 2D Inverse Fourier Transform with proper frequency shifting
Psi_x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Psi_q)))
JPD = np.abs(Psi_x)**2
JPD /= np.max(JPD)

# Corresponding spatial grid coordinates [m]
# dx = 2*pi / (N * dq)
x_vec = np.fft.fftshift(np.fft.fftfreq(N_pts, d=dq / (2 * np.pi)))

# ==========================================
# 4. SCHMIDT ENTANGLEMENT ANALYSIS (SVD)
# ==========================================
U, S, Vh = np.linalg.svd(Psi_q)
prob_modes = (S**2) / np.sum(S**2)
schmidt_K = 1.0 / np.sum(prob_modes**2)
print(f"Schmidt Number (K)         : {schmidt_K:.2f}")

# ==========================================
# 5. VISUALIZATION (Sol: Konum | Sağ: Momentum)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(13, 11))
plt.subplots_adjust(hspace=0.32, wspace=0.3)

q_scale = 1e-6 # Convert rad/m to rad/um
x_scale = 1e6  # Convert m to um

# --- SOL ÜST (axs[0, 0]): Joint Position Distribution (JPD via 2D-FFT) ---
x_limit = 180 # Limit plot window to +/- 180 um
mask_x = (x_vec * x_scale >= -x_limit) & (x_vec * x_scale <= x_limit)
x_sub = x_vec[mask_x] * x_scale
JPD_sub = JPD[np.ix_(mask_x, mask_x)]

im0 = axs[0, 0].imshow(JPD_sub, origin='lower', cmap='viridis',
                       extent=[-x_limit, x_limit, -x_limit, x_limit])
axs[0, 0].set_title(r"$\mathbf{(a)\ Position\ Space\ |\Psi(x_s, x_i)|^2\ (2D\text{-}FFT)}$", fontsize=11)
axs[0, 0].set_xlabel(r"Signal Position $x_s\ [\mathrm{\mu m}]$", fontsize=11)
axs[0, 0].set_ylabel(r"Idler Position $x_i\ [\mathrm{\mu m}]$", fontsize=11)
axs[0, 0].axline((0, 0), slope=1, color='yellow', linestyle='--', alpha=0.6, label=r"$x_s = x_i$")
axs[0, 0].legend(loc='upper left', fontsize=9)
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04, label="Normalized Intensity")

# --- SAĞ ÜST (axs[0, 1]): Joint Momentum Distribution (JMD) ---
im1 = axs[0, 1].imshow(JMD, origin='lower', cmap='inferno',
                       extent=[-q_max*q_scale, q_max*q_scale, -q_max*q_scale, q_max*q_scale])
axs[0, 1].set_title(r"$\mathbf{(b)\ Momentum\ Space\ |\Psi(q_s, q_i)|^2\ (Sinc\text{-}Gauss)}$", fontsize=11)
axs[0, 1].set_xlabel(r"Signal Momentum $q_s\ [\mathrm{rad/\mu m}]$", fontsize=11)
axs[0, 1].set_ylabel(r"Idler Momentum $q_i\ [\mathrm{rad/\mu m}]$", fontsize=11)
axs[0, 1].axline((0, 0), slope=-1, color='cyan', linestyle='--', alpha=0.6, label=r"$q_s = -q_i$")
axs[0, 1].legend(loc='upper right', fontsize=9)
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, label="Normalized Intensity")

# --- SOL ALT (axs[1, 0]): Position Cross-Sections (Conditional vs Diagonal) ---
mid_idx = N_pts // 2
diag_coinc_x = np.diagonal(JPD)

axs[1, 0].plot(x_vec * x_scale, JPD[mid_idx, :], 'g-', lw=2, label=r"Conditional $P(x_s | x_i=0)$ (Narrow)")
axs[1, 0].plot(x_vec * x_scale, diag_coinc_x, 'm--', lw=2, label=r"Coincidence $x_s = x_i$ (Pump Envelope)")
axs[1, 0].set_title(r"$\mathbf{(c)\ Position\ Profiles\ (Spatial\ Coincidence)}$", fontsize=11)
axs[1, 0].set_xlim([-x_limit, x_limit])
axs[1, 0].set_xlabel(r"Transverse Position $x\ [\mathrm{\mu m}]$", fontsize=11)
axs[1, 0].set_ylabel("Intensity [Linear]", fontsize=11)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend(fontsize=9)

# --- SAĞ ALT (axs[1, 1]): Momentum Cross-Section along anti-diagonal qs = -qi ---
diag_anti_q = np.diagonal(np.fliplr(JMD))
q_axis_anti = np.linspace(-q_max, q_max, N_pts) * q_scale

axs[1, 1].plot(q_axis_anti, diag_anti_q, 'r-', lw=2, label="Anti-diagonal ($q_s = -q_i$)")
axs[1, 1].set_title(r"$\mathbf{(d)\ Momentum\ Profile\ along\ } q_s = -q_i$", fontsize=11)
axs[1, 1].set_xlabel(r"Transverse Momentum $q\ [\mathrm{rad/\mu m}]$", fontsize=11)
axs[1, 1].set_ylabel("Intensity [Linear]", fontsize=11)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend(fontsize=9)

# Inset / Log-Scale plot to highlight Sinc Sidelobes (Sağ alt grafiğin içine eklendi)
ax_inset = axs[1, 1].inset_axes([0.60, 0.42, 0.36, 0.5])
ax_inset.semilogy(q_axis_anti, np.maximum(diag_anti_q, 1e-5), 'm-', lw=1.5)
ax_inset.set_title("Log-scale (Sinc Sidelobes)", fontsize=8)
ax_inset.set_ylim([1e-4, 1.1])
ax_inset.grid(True, which='both', alpha=0.3)

plt.suptitle(f"Sinc-Gauss Model for SPDC Transverse Spatial Entanglement\nSchmidt Number $K = {schmidt_K:.1f}$", fontsize=13, y=0.98)
plt.show()