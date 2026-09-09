import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 


# --- 1. Fiziksel Parametreler ---
lambda_p = 405e-9      # Pompa dalgaboyu (405 nm)
lambda_s = 2 * lambda_p  # Signal/Idler dalgaboyu (810 nm)
L_z = 2e-3             # Kristal kalınlığı (2 mm)
sigma_p = 100e-6       # Pompa lazeri yarıçapı (100 um)

# --- 2. Momentum Izgarası (Momentum Grid) ---
# Sinc-Gaussian momentumda tanımlı olduğu için önce momentum ızgarasını kuruyoruz.
k_max = 2.0e6          # rad/m cinsinden maksimum momentum
N_points = 512         # Sinc saçaklarını net görmek için çözünürlüğü artırdık
k = np.linspace(-k_max, k_max, N_points)
dk = k[1] - k[0]       # Momentum adım boyu
K1, K2 = np.meshgrid(k, k)

# --- 3. Momentum Uzayında Sinc-Gaussian Dalga Fonksiyonu (Denklem 26) ---
# Sinc terimi argümanı
sinc_arg = (L_z * lambda_p) / (8 * np.pi) * (K1 - K2)**2
# numpy'ın gizli pi çarpanını kompanse ediyoruz:
term_sinc = np.sinc(sinc_arg / np.pi) 

# Gaussian terimi
term_gauss = np.exp(-(sigma_p**2) * (K1 + K2)**2)

# Biphoton Momentum Dalga Fonksiyonu
psi_k = term_sinc * term_gauss
psi_k = psi_k / np.sqrt(np.sum(np.abs(psi_k)**2))  # Normalizasyon
JPD_k = np.abs(psi_k)**2  # Momentum JPD'si

# --- 4. 2D IFFT ile Konum Uzayına Geçiş (Near-Field) ---
# Momentumdan konuma geçmek için Ters Fourier Dönüşümü (IFFT) yapıyoruz.
psi_x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psi_k)))
psi_x = psi_x / np.sqrt(np.sum(np.abs(psi_x)**2))  # Konumda normalizasyon
JPD_x = np.abs(psi_x)**2  # Konum JPD'si

# --- 5. Konum Izgarasının Hesaplanması (Position Grid) ---
# dx * dk = 2 * pi / N ilişkisinden konum adım boyunu buluyoruz.
dx = (2 * np.pi) / (N_points * dk)
x = np.fft.fftshift(np.fft.fftfreq(N_points, d=1/N_points)) * dx
X1, X2 = np.meshgrid(x, x)


# --- 7. Görselleştirme (np.log10 ile Kusursuz Çizim) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Çok küçük değerlerde log(0) hatası almamak için küçük bir epsilon (1e-10) ekliyoruz
log_JPD_k = np.log10(JPD_k + 1e-10)
log_JPD_x = np.log10(JPD_x + 1e-10)

# Sol Grafik: Momentum Uzayı (Logaritmik Ölçekli)
im1 = ax1.pcolormesh(K1 * 1e-6, K2 * 1e-6, log_JPD_k, cmap='jet', shading='auto',
                     vmin=-6, vmax=log_JPD_k.max()) # -6 ile max arası (yani 1e-6'ya kadar olanları göster)
fig.colorbar(im1, ax=ax1, label='Log10(Coincidence Probability)')
ax1.set_xlabel('Signal Momentum $k_s$ ($rad/\mu m$)')
ax1.set_ylabel('Idler Momentum $k_i$ ($rad/\mu m$)')
ax1.set_title('Momentum JPD (Log10 Ölçek)')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_aspect('equal')
k_limit = 1.2
ax1.set_xlim(-k_limit, k_limit)
ax1.set_ylim(-k_limit, k_limit)

# Sağ Grafik: Konum Uzayı (Logaritmik Ölçekli)
im2 = ax2.pcolormesh(X1 * 1e6, X2 * 1e6, log_JPD_x, cmap='jet', shading='auto',
                     vmin=-7, vmax=log_JPD_x.max()) # -7 ile max arası
fig.colorbar(im2, ax=ax2, label='Log10(Coincidence Probability)')
ax2.set_xlabel('Signal Position $x_s$ ($\mu m$)')
ax2.set_ylabel('Idler Position $x_i$ ($\mu m$)')
ax2.set_title('Position JPD (Log10 Ölçek)')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_aspect('equal')
x_limit = 200
ax2.set_xlim(-x_limit, x_limit)
ax2.set_ylim(-x_limit, x_limit)

plt.tight_layout()
plt.show()