# ============================================================
# DOUBLE-GAUSSIAN SPDC SIMULATION (2D - Full 4D Tensor)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# GPU varsa CuPy kullan, yoksa otomatik NumPy (CPU) kullan
try:
    import cupy as xp
    is_gpu = True
    print("Donanım: NVIDIA GPU (CuPy) ile çalıştırılıyor...")
except ImportError:
    import numpy as xp
    is_gpu = False
    print("Donanım: CuPy bulunamadı, CPU (NumPy) ile çalıştırılıyor...")

# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

lambda_p = 405e-9          # Pump wavelength [m]
Lz = 1e-3                  # Crystal length [m]
sigma_p = 50e-6            # Pump beam waist [m]

sigma_r_plus = xp.sqrt(2) * sigma_p
sigma_r_minus = xp.sqrt(Lz * lambda_p / (12 * xp.pi))

# 2D Schmidt number (Theoretical)
ratio = sigma_r_plus / sigma_r_minus
K_per_dim = 0.5 * (ratio + 1.0 / ratio)
K_theory = K_per_dim**2

print(f"Theoretical Schmidt (2D) = {float(K_theory):.2f}")

# ============================================================
# 2. 4D GRID CREATION
# ============================================================

N = 64
rmax = 200e-6

x = xp.linspace(-rmax, rmax, N)
xs, ys = xp.meshgrid(x, x)

# Build 4D tensor grids: (xs, ys, xi, yi)
Xs = xs[:, :, None, None]
Ys = ys[:, :, None, None]
Xi = xs[None, None, :, :]
Yi = ys[None, None, :, :]

# ============================================================
# 3. DOUBLE GAUSSIAN BIPHOTON AMPLITUDE
# ============================================================

r_plus_sq = (Xs + Xi)**2 + (Ys + Yi)**2
r_minus_sq = (Xs - Xi)**2 + (Ys - Yi)**2

phi = xp.exp(-r_plus_sq / (4.0 * sigma_r_plus**2)) * \
      xp.exp(-r_minus_sq / (4.0 * sigma_r_minus**2))

# Normalize wavefunction
norm = xp.sqrt(xp.sum(xp.abs(phi)**2))
phi /= norm

G2 = xp.abs(phi)**2

# ============================================================
# 4. NEAR FIELD (REAL SPACE)
# ============================================================

Intensity_nf = xp.sum(G2, axis=(2, 3))
ref = N // 4
Conditional_nf = G2[ref, ref, :, :]

# ============================================================
# 5. FAR FIELD (MOMENTUM SPACE VIA 4D-FFT)
# ============================================================

phi_k = xp.fft.fftn(phi, axes=(0, 1, 2, 3))
phi_k = xp.fft.fftshift(phi_k)

G2_k = xp.abs(phi_k)**2

Intensity_ff = xp.sum(G2_k, axis=(2, 3))
Conditional_ff = G2_k[ref, ref, :, :]

# ============================================================
# 6. CORRELATION IMAGES
# ============================================================

mid = N // 2

# Position-position correlation: G2(xs, xi) at ys=yi=0
C_pos = G2[mid, :, mid, :]

# Momentum-momentum correlation: G2(kxs, kxi) at kys=kyi=0
C_mom = G2_k[mid, :, mid, :]

# ============================================================
# 7. NUMERICAL SCHMIDT (SVD)
# ============================================================

phi_matrix = phi.reshape(N * N, N * N)

U, s, Vh = xp.linalg.svd(phi_matrix, full_matrices=False)

lambdas = s**2
lambdas /= xp.sum(lambdas)

K_numeric = 1.0 / xp.sum(lambdas**2)
print(f"Numerical Schmidt (2D)   = {float(K_numeric):.2f}")

# ============================================================
# 8. CONVERT TO CPU FOR PLOTTING
# ============================================================

def to_cpu(arr):
    if is_gpu:
        return xp.asnumpy(arr)
    return arr

x_cpu = to_cpu(x) * 1e6  # to micrometers

Intensity_nf = to_cpu(Intensity_nf)
Conditional_nf = to_cpu(Conditional_nf)
Intensity_ff = to_cpu(Intensity_ff)
Conditional_ff = to_cpu(Conditional_ff)
C_pos = to_cpu(C_pos)
C_mom = to_cpu(C_mom)

# ============================================================
# 9. PLOTS: NEAR-FIELD & FAR-FIELD
# ============================================================

extent_pos = [x_cpu[0], x_cpu[-1], x_cpu[0], x_cpu[-1]]

plt.figure(figsize=(14, 10))

# 1. Near-field intensity
plt.subplot(2, 2, 1)
plt.imshow(Intensity_nf, origin='lower', extent=extent_pos, cmap='viridis')
plt.xlabel(r"$x_s\ (\mu\mathrm{m})$")
plt.ylabel(r"$y_s\ (\mu\mathrm{m})$")
plt.title("Near-field Total Intensity")
plt.colorbar()

# 2. Near-field conditional
plt.subplot(2, 2, 2)
plt.imshow(Conditional_nf, origin='lower', extent=extent_pos, cmap='viridis')
plt.xlabel(r"$x_i\ (\mu\mathrm{m})$")
plt.ylabel(r"$y_i\ (\mu\mathrm{m})$")
plt.title(f"Near-field Conditional (at $x_s, y_s = {x_cpu[ref]:.1f}\\mu m$)")
plt.colorbar()

# 3. Far-field intensity
plt.subplot(2, 2, 3)
plt.imshow(Intensity_ff, origin='lower', cmap='inferno')
plt.xlabel(r"$k_{s,x}\ \mathrm{(arb.)}$")
plt.ylabel(r"$k_{s,y}\ \mathrm{(arb.)}$")
plt.title("Far-field Total Intensity")
plt.colorbar()

# 4. Far-field conditional
plt.subplot(2, 2, 4)
plt.imshow(Conditional_ff, origin='lower', cmap='inferno')
plt.xlabel(r"$k_{i,x}\ \mathrm{(arb.)}$")
plt.ylabel(r"$k_{i,y}\ \mathrm{(arb.)}$")
plt.title("Far-field Conditional (Momentum Lock)")
plt.colorbar()

plt.tight_layout()

# ============================================================
# 10. PLOTS: 2D CORRELATION MATRICES
# ============================================================

plt.figure(figsize=(13, 5.5))

# Position-position correlation
plt.subplot(1, 2, 1)
plt.imshow(C_pos, origin='lower', extent=extent_pos, cmap='viridis')
plt.axline((0, 0), slope=1, color='yellow', linestyle='--', alpha=0.6, label=r"$x_s = x_i$")
plt.xlabel(r"$x_i\ (\mu\mathrm{m})$")
plt.ylabel(r"$x_s\ (\mu\mathrm{m})$")
plt.title(r"$\mathbf{Position\text{-}Position\ Correlation}$" + "\n" + r"$G^{(2)}(x_s, x_i)\ \mathrm{at}\ y_s=y_i=0$")
plt.legend(loc='upper left')
plt.colorbar()

# Momentum-momentum correlation
plt.subplot(1, 2, 2)
plt.imshow(C_mom, origin='lower', cmap='inferno')
plt.xlabel(r"$k_{i,x}\ \mathrm{(arb.)}$")
plt.ylabel(r"$k_{s,x}\ \mathrm{(arb.)}$")
plt.title(r"$\mathbf{Momentum\text{-}Momentum\ Correlation}$" + "\n" + r"$G^{(2)}(k_{s,x}, k_{i,x})\ \mathrm{at}\ k_y=0$")
plt.colorbar()

plt.tight_layout()
plt.show()