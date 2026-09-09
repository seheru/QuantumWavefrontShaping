import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==============================================================================
# 1. PARAMETERS
# ==============================================================================
lambda_p = 404e-9          # Pompa dalgaboyu (m)
L_z      = 0.5e-3          # Kristal boyu
sigma_p  = 60e-6           # Pompa bel yarıçapı

# ==============================================================================
# 2. 2D GRIDS
# ==============================================================================
N     = 48
k_max = 1.5e6              # rad/m
k     = np.linspace(-k_max, k_max, N)
dk    = k[1] - k[0]

dx = 2 * np.pi / (N * dk)
x  = (np.arange(N) - N // 2) * dx

X, Y   = np.meshgrid(x, x, indexing="ij")
Kx, Ky = np.meshgrid(k, k, indexing="ij")

# ==============================================================================
# 3. 2D SLM SETUP
# ==============================================================================
n_seg_1d = 6               # 6x6 = 36 kontrol segmenti
n_seg    = n_seg_1d * n_seg_1d

slm_w = 2.0 * sigma_p
seg_edges = np.linspace(-slm_w, slm_w, n_seg_1d + 1)
seg_idx_x = np.clip(np.digitize(x, seg_edges) - 1, 0, n_seg_1d - 1)
seg_idx_y = np.clip(np.digitize(x, seg_edges) - 1, 0, n_seg_1d - 1)
SEG_X, SEG_Y = np.meshgrid(seg_idx_x, seg_idx_y, indexing="ij")
SEG_ID = SEG_X * n_seg_1d + SEG_Y

def pump_field_crystal(phi_slm):
    return np.exp(-(X**2 + Y**2) / (2 * sigma_p**2)) * np.exp(1j * phi_slm[SEG_ID])

# ==============================================================================
# 4. 2D DIFFUSER (TAM GELİŞMİŞ SPECKLE)
# ==============================================================================
corr_len = 12e-6
raw_phase = rng.uniform(-np.pi, np.pi, size=(N, N))

kx_filter = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KFX, KFY = np.meshgrid(kx_filter, kx_filter)
gauss_filter = np.exp(-(KFX**2 + KFY**2) * (corr_len**2) / 4.0)

diff_smooth = np.fft.ifft2(np.fft.fft2(np.exp(1j * raw_phase)) * gauss_filter)
phi_diff = np.angle(diff_smooth) * 2.5

# Dalgaboyu ilişkisi: Pompa = D^2, Sinyal/İdler tek foton = D
D_single = np.exp(1j * phi_diff)
D_pump   = D_single**2

# ==============================================================================
# 5. FORWARD MODEL
# ==============================================================================
def pump_far_field(phi_slm):
    Ep = pump_field_crystal(phi_slm) * D_pump
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ep)))

def get_quantum_coincidence(phi_slm):
    # İdler optik eksende (qi = 0) sabitlendiğinde iki fotonun toplam fazı
    # klasik pompanın gördüğü saçıcı fazına özdeştir: D_single(xs) * D_single(xi=0)
    Ep = pump_field_crystal(phi_slm) * D_single
    psi_far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ep)))
    
    # Kristal faz uyumu filtreleme zarfı
    q_sq = Kx**2 + Ky**2
    Phi_env = np.sinc(((L_z * lambda_p) / (8 * np.pi) * q_sq) / np.pi)
    
    return np.abs(psi_far * Phi_env)**2

# ==============================================================================
# 6. OPTİMİZASYON (IŞIĞIN OLDUĞU GERÇEKÇİ HEDEF SEÇİMİ)
# ==============================================================================
# Hedef: Merkezin hemen yanındaki canlı speckle bölgesi (2 piksel ötede)
target_row = N // 2 + 3
target_col = N // 2 - 2

def target_cost(phi_slm):
    far = pump_far_field(phi_slm)
    # Hedef piksel ve komşularının ortalaması
    return np.abs(far[target_row, target_col])**2

phi_slm = np.zeros(n_seg)

# Başlangıç
I_pump_before = np.abs(pump_far_field(phi_slm))**2
C_before      = get_quantum_coincidence(phi_slm)

print("2D Optimizasyon Başladı...")
theta_scan = np.linspace(0, 2 * np.pi, 6, endpoint=False)
n_iter = 100

for it in range(n_iter):
    group = rng.random(n_seg) < 0.5
    vals = []
    for th in theta_scan:
        trial = phi_slm.copy()
        trial[group] += th
        vals.append(target_cost(trial))
    
    A = np.stack([np.ones_like(theta_scan), np.cos(theta_scan), np.sin(theta_scan)], axis=1)
    a, bc, bs = np.linalg.lstsq(A, vals, rcond=None)[0]
    phi_slm[group] += np.arctan2(bs, bc)

print("Optimizasyon Tamamlandı!")

# Bitiş
I_pump_after = np.abs(pump_far_field(phi_slm))**2
C_after      = get_quantum_coincidence(phi_slm)

pump_enh  = I_pump_after[target_row, target_col] / (I_pump_before[target_row, target_col] + 1e-9)
coinc_enh = C_after[target_row, target_col] / (C_before[target_row, target_col] + 1e-9)

print(f"\n--> Hedefte Pompa Güçlenmesi:    {pump_enh:.1f}x")
print(f"--> Hedefte Kuantum Güçlenmesi:  {coinc_enh:.1f}x")

# ==============================================================================
# 7. GÖRSELLEŞTİRME (EKSENLER VE İŞARETÇİ TAM KİLİTLİ)
# ==============================================================================
q_axis = k * 1e-6
extent = [q_axis[0], q_axis[-1], q_axis[0], q_axis[-1]]

# Hedef noktanın fiziksel koordinatları:
target_qx = q_axis[target_col]
target_qy = q_axis[target_row]

fig, axs = plt.subplots(2, 2, figsize=(10, 9))

# Sol Üst: Pompa Önce
im0 = axs[0, 0].imshow(I_pump_before, origin="lower", extent=extent, cmap="viridis")
axs[0, 0].plot(target_qx, target_qy, 'r+', markersize=14, markeredgewidth=2.5)
axs[0, 0].set_title("Pompa Uzak Alan (Önce)")
axs[0, 0].set_xlabel(r"$q_{px}$ (rad/µm)"); axs[0, 0].set_ylabel(r"$q_{py}$ (rad/µm)")
plt.colorbar(im0, ax=axs[0, 0], fraction=0.046)

# Sağ Üst: Kuantum Önce
im1 = axs[0, 1].imshow(C_before, origin="lower", extent=extent, cmap="plasma")
axs[0, 1].plot(target_qx, target_qy, 'r+', markersize=14, markeredgewidth=2.5)
axs[0, 1].set_title(r"Kuantum Çakışma $C(q_s | q_i=0)$ (Önce)")
axs[0, 1].set_xlabel(r"$q_{sx}$ (rad/µm)"); axs[0, 1].set_ylabel(r"$q_{sy}$ (rad/µm)")
plt.colorbar(im1, ax=axs[0, 1], fraction=0.046)

# Sol Alt: Pompa Sonra (Odaklanmış)
im2 = axs[1, 0].imshow(I_pump_after, origin="lower", extent=extent, cmap="viridis")
axs[1, 0].plot(target_qx, target_qy, 'r+', markersize=14, markeredgewidth=2.5)
axs[1, 0].set_title(f"Pompa Uzak Alan (Sonra - {pump_enh:.1f}x)")
axs[1, 0].set_xlabel(r"$q_{px}$ (rad/µm)"); axs[1, 0].set_ylabel(r"$q_{py}$ (rad/µm)")
plt.colorbar(im2, ax=axs[1, 0], fraction=0.046)

# Sağ Alt: Kuantum Sonra (Hedefe Odaklanmış)
im3 = axs[1, 1].imshow(C_after, origin="lower", extent=extent, cmap="plasma")
axs[1, 1].plot(target_qx, target_qy, 'r+', markersize=14, markeredgewidth=2.5)
axs[1, 1].set_title(f"Kuantum Çakışma (Sonra - {coinc_enh:.1f}x)")
axs[1, 1].set_xlabel(r"$q_{sx}$ (rad/µm)"); axs[1, 1].set_ylabel(r"$q_{sy}$ (rad/µm)")
plt.colorbar(im3, ax=axs[1, 1], fraction=0.046)

plt.tight_layout()
plt.show()