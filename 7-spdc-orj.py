# ============================================================
# DOUBLE-GAUSSIAN SPDC SIMULATION (2D)
# ============================================================

import numpy as cp
import numpy as np
import matplotlib.pyplot as plt

cp.asnumpy = np.asarray ###

# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

lambda_p = 405e-9
Lz = 1e-3
sigma_p = 50e-6

sigma_r_plus = cp.sqrt(2) * sigma_p
sigma_r_minus = cp.sqrt(Lz * lambda_p / (12 * cp.pi))

# 2D Schmidt number
ratio = sigma_r_plus / sigma_r_minus
K_per_dim = 0.5 * (ratio + 1/ratio)
K_theory = K_per_dim**2

print("Theoretical Schmidt (2D) =", float(K_theory))

# ============================================================
# 2. GRID
# ============================================================

N = 64
rmax = 200e-6

x = cp.linspace(-rmax, rmax, N)
xs, ys = cp.meshgrid(x, x)

# Build 4D grids
Xs = xs[:, :, None, None]
Ys = ys[:, :, None, None]
Xi = xs[None, None, :, :]
Yi = ys[None, None, :, :]

# ============================================================
# 3. DOUBLE GAUSSIAN BIPHOTON AMPLITUDE
# ============================================================

r_plus_sq = (Xs + Xi)**2 + (Ys + Yi)**2
r_minus_sq = (Xs - Xi)**2 + (Ys - Yi)**2

phi = cp.exp(-r_plus_sq / (4 * sigma_r_plus**2)) \
  * cp.exp(-r_minus_sq / (4 * sigma_r_minus**2))

# Normalize wavefunction
norm = cp.sqrt(cp.sum(cp.abs(phi)**2))
phi /= norm

G2 = cp.abs(phi)**2

# ============================================================
# 4. NEAR FIELD
# ============================================================

Intensity_nf = cp.sum(G2, axis=(2,3))
ref = N // 4
Conditional_nf = G2[ref, ref, :, :]

# ============================================================
# 5. FAR FIELD (FFT of full biphoton amplitude)
# ============================================================

phi_k = cp.fft.fftn(phi, axes=(0,1,2,3))
phi_k = cp.fft.fftshift(phi_k)

G2_k = cp.abs(phi_k)**2

Intensity_ff = cp.sum(G2_k, axis=(2,3))
Conditional_ff = G2_k[ref, ref, :, :]


# 6. CORRELATION IMAGES

# ============================================================


mid = N // 2


# Position-position correlation: G2(xs, xi) at ys=yi=0

C_pos = G2[mid, :, mid, :]


# Momentum-momentum correlation: G2(kxs, kxi) at kys=kyi=0

C_mom = G2_k[mid, :, mid, :]


# Sum coordinate marginal

C_plus = cp.sum(G2[mid, :, mid, :], axis=1)


# Difference coordinate marginal

C_minus = cp.sum(G2[mid, :, mid, :], axis=0)


# ============================================================

# 7. NUMERICAL SCHMIDT (SVD)

# ============================================================


phi_matrix = phi.reshape(N*N, N*N)


U, s, Vh = cp.linalg.svd(phi_matrix, full_matrices=False)


lambdas = s**2

lambdas /= cp.sum(lambdas)


K_numeric = 1.0 / cp.sum(lambdas**2)


print("Numerical Schmidt (2D) =", float(K_numeric))


# ============================================================

# 9. MOVE TO CPU

# ============================================================


x_cpu = cp.asnumpy(x) * 1e6 # convert to micrometers


Intensity_nf = cp.asnumpy(Intensity_nf)

Conditional_nf = cp.asnumpy(Conditional_nf)

Intensity_ff = cp.asnumpy(Intensity_ff)

Conditional_ff = cp.asnumpy(Conditional_ff)

C_pos     = cp.asnumpy(C_pos)

C_mom     = cp.asnumpy(C_mom)

C_minus    = cp.asnumpy(C_minus)

C_plus    = cp.asnumpy(C_plus)


# ============================================================

# 10. PLOTS (WITH AXES + CONDITIONAL MARKERS)

# ============================================================


extent = [x_cpu[0], x_cpu[-1], x_cpu[0], x_cpu[-1]]


plt.figure(figsize=(15,11))


# Near-field intensity

plt.subplot(2,2,1)

plt.imshow(Intensity_nf, origin='lower', extent=extent)

plt.xlabel("x_s (µm)")

plt.ylabel("y_s (µm)")

plt.title("Near-field Intensity")

plt.colorbar()


# Near-field conditional

plt.subplot(2,2,2)

plt.imshow(Conditional_nf, origin='lower', extent=extent)

plt.xlabel("x_i (µm)")

plt.ylabel("y_i (µm)")

plt.title("Near-field Conditional")

plt.colorbar()


# Far-field intensity

plt.subplot(2,2,3)

plt.imshow(Intensity_ff, origin='lower', extent=extent)

plt.xlabel("k_x (arb.)")

plt.ylabel("k_y (arb.)")

plt.title("Far-field Intensity")

plt.colorbar()


# Far-field conditional

plt.subplot(2,2,4)

plt.imshow(Conditional_ff, origin='lower', extent=extent)

plt.xlabel("k_{i,x}")

plt.ylabel("k_{i,y}")

plt.title("Far-field Conditional")

plt.colorbar()


plt.tight_layout()

plt.show()


# ============================================================

# 11. CORRELATION IMAGES

# ============================================================


plt.figure(figsize=(15,11))


# Position-position correlation

plt.subplot(1,2,1)

plt.imshow(C_pos, origin='lower', extent=extent)

plt.xlabel("x_i (µm)")

plt.ylabel("x_s (µm)")

plt.title("Position-Position Correlation\nG2(xs, xi) at ys=yi=0")

plt.colorbar()


# Momentum-momentum correlation

plt.subplot(1,2,2)

plt.imshow(C_mom, origin='lower', extent=extent)

plt.xlabel("k_{x,i} (arb.)")

plt.ylabel("k_{x,s} (arb.)")

plt.title("Momentum-Momentum Correlation\nG2(kxs, kxi) at kys=kyi=0")

plt.colorbar()

plt.tight_layout()

plt.show()
