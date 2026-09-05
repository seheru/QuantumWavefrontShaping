import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. FİZİKSEL PARAMETRELER (Howell et al. 2004)
# ==========================================================
lambda_p = 390e-9       # 390 nm
L_z = 2.0e-3            # 2 mm
sigma_p = 0.5e-3        # 0.5 mm

# ==========================================================
# 2. TEORİK GENİŞLİKLER VE KONUM IZGARASI HESABI
# ==========================================================
a = (L_z * lambda_p) / (4 * np.pi)
sigma_x_plus = np.sqrt(2) * sigma_p
sigma_x_minus = np.sqrt(8 * a / 9)  # Peak-matching genişliği

# 1. Kod ile Birebir Aynı Boyutta Konum Izgarası
N_points = 512
k_max = 3.0e5
dk = (2 * k_max) / (N_points - 1)
dx = 2 * np.pi / (N_points * dk)
x_vals = (np.arange(N_points) - N_points // 2) * dx
X1, X2 = np.meshgrid(x_vals, x_vals)

# Denklem 32 Çift-Gauss Analitik Formülü
rho_gauss = (1.0 / (2 * np.pi * sigma_x_plus * sigma_x_minus)) * \
            np.exp(-((X1 - X2)**2) / (4 * sigma_x_minus**2) - ((X1 + X2)**2) / (4 * sigma_x_plus**2))
rho_gauss /= np.sum(rho_gauss) * (dx**2)

# ==========================================================
# 3. STANDARTLAŞTIRILMIŞ ÇİZİM (1. Kod ile Birebir Aynı Format)
# ==========================================================
plt.figure(figsize=(7, 6))
extent_um = [x_vals[0]*1e6, x_vals[-1]*1e6, x_vals[0]*1e6, x_vals[-1]*1e6]

plt.imshow(rho_gauss, origin='lower', extent=extent_um, cmap='plasma')
plt.title("YÖNTEM 2: Çift-Gauss Analitik Dağılımı\n(Makalenin Denklem 32 Formülü)", fontsize=12, fontweight='bold')
plt.xlabel("1. Foton Konumu $x_1$ (mikrometre)", fontsize=11)
plt.ylabel("2. Foton Konumu $x_2$ (mikrometre)", fontsize=11)

# Sabit Eksen Limitleri ([-60, +60] mikrometre)
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.colorbar(label='Olasılık Yoğunluğu $\\rho(x_1, x_2)$')
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()