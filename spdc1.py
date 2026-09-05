import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. FİZİKSEL PARAMETRELER (Howell et al. 2004)
# ==========================================================
lambda_p = 390e-9       # 390 nm
L_z = 2.0e-3            # 2 mm
sigma_p = 0.5e-3        # 0.5 mm

# ==========================================================
# 2. MOMENTUM IZGARASI VE TERS FOURIER HESABI
# ==========================================================
N_points = 512
k_max = 3.0e5           # rad/m
k_vals = np.linspace(-k_max, k_max, N_points)
dk = k_vals[1] - k_vals[0]
K1, K2 = np.meshgrid(k_vals, k_vals)

# Ham Sinc Dalga Fonksiyonu (Denklem 26)
sinc_arg = (L_z * lambda_p) / (8 * np.pi) * (K1 - K2)**2
sinc_part = np.sinc(sinc_arg / np.pi)
gauss_pump = np.exp(- (sigma_p**2 / 2) * (K1 + K2)**2)
Phi_k1_k2 = sinc_part * gauss_pump

# 2D Ters Fourier Dönüşümü (Momentum -> Konum)
psi_exact = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phi_k1_k2)))

# Konum Ekseni (dx = 2*pi / (N*dk))
dx = 2 * np.pi / (N_points * dk)
x_vals = (np.arange(N_points) - N_points // 2) * dx

# Kuantum Olasılık Yoğunluğu: rho = |psi|^2
rho_exact = np.abs(psi_exact)**2
rho_exact /= np.sum(rho_exact) * (dx**2)

# ==========================================================
# 3. STANDARTLAŞTIRILMIŞ ÇİZİM
# ==========================================================
plt.figure(figsize=(7, 6))
extent_um = [x_vals[0]*1e6, x_vals[-1]*1e6, x_vals[0]*1e6, x_vals[-1]*1e6]

plt.imshow(rho_exact, origin='lower', extent=extent_um, cmap='plasma')
plt.title("YÖNTEM 1: Ham Kuantum Mekaniği Dağılımı\n$\\rho(x_1, x_2) = |\\mathcal{F}^{-1}[\\Phi_{\\text{sinc}}]|^2$", fontsize=12, fontweight='bold')
plt.xlabel("1. Foton Konumu $x_1$ (mikrometre)", fontsize=11)
plt.ylabel("2. Foton Konumu $x_2$ (mikrometre)", fontsize=11)

# Sabit Eksen Limitleri ([-60, +60] mikrometre)
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.colorbar(label='Olasılık Yoğunluğu $\\rho(x_1, x_2)$')
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()