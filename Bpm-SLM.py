import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. PARAMETRELER VE IZGARALAR
# =========================================================================
wavelength = 1.0e-6          # 1 mikron dalgaboyu
k0 = 2 * np.pi / wavelength   # Boşluktaki dalga sayısı
n0 = 1.0                     # HAVA (n = 1.0)
k = n0 * k0

Lx = 200.0e-6                # 200 mikron pencere
Ly = 200.0e-6
Nx = 256
Ny = 256
dx = Lx / Nx
dy = Ly / Ny

Lz = 1000.0e-6               # 1000 mikron (1 mm) toplam yol
Nz = 400                     # 400 adım
dz = Lz / Nz

x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)
R_sq = X**2 + Y**2

# Frekans Izgarası ve ASM Operatörü
fx = np.fft.fftfreq(Nx, d=dx)
fy = np.fft.fftfreq(Ny, d=dy)
FX, FY = np.meshgrid(fx, fy)
kx = 2 * np.pi * FX
ky = 2 * np.pi * FY
k_transverse_sq = kx**2 + ky**2

propagating_mask = k_transverse_sq <= k**2
kz = np.zeros_like(k_transverse_sq)
kz[propagating_mask] = np.sqrt(k**2 - k_transverse_sq[propagating_mask])
H_diffraction = np.exp(1j * kz * dz) * propagating_mask

absorber = np.exp(- (X / (0.85*Lx/2))**20 - (Y / (0.85*Ly/2))**20)


# =========================================================================
# 2. OPTİK ELEMANLARIN TANIMLANMASI (SLM + LENS + DİFÜZÖR)
# =========================================================================
# A) SLM: Işığı Yukarı Bükücü Faz Rampası (z = 120 um noktasına koyduk)
steer_angle = 0.030      # yukarı sapma açısı
kx_steer = k0 * np.sin(steer_angle)
phi_slm = kx_steer * X       # SLM piksellerine yüklenen lineer faz rampası
slm_mask = np.exp(1j * phi_slm)
z_slm = 120.0e-6
step_slm = int(z_slm / dz)

# B) İNCE LENS: Odak Uzaklığı f = 300 um (z = 350 um noktasına koyduk)
f_lens = 300.0e-6
lens_mask = np.exp(-1j * (k0 / (2 * f_lens)) * R_sq)
z_lens = 350.0e-6
step_lens = int(z_lens / dz)

# C) GERÇEKÇİ DİFÜZÖR: (z = 650 um noktasına koyduk)
np.random.seed(42)
raw_noise = np.random.randn(Ny, Nx)
sigma_f = 1 / (6.0e-6)
filter_2d = np.exp(- (FX**2 + FY**2) / (2 * sigma_f**2))
smooth_phase = np.real(np.fft.ifft2(np.fft.fft2(raw_noise) * filter_2d))
smooth_phase = (smooth_phase / np.std(smooth_phase)) * 2.2

diffuser_mask = np.exp(1j * smooth_phase)
z_diffuser = 650.0e-6
step_diffuser = int(z_diffuser / dz)


# =========================================================================
# 3. BAŞLANGIÇ FENERİ
# =========================================================================
w0 = 22.0e-6                 # 22 mikron bel yarıçaplı parlak fener
E = np.exp(-R_sq / (w0**2)).astype(np.complex128)


# =========================================================================
# 4. SİMÜLASYON DÖNGÜSÜ
# =========================================================================
xz_profile = np.zeros((Nz, Nx))

print("Simülasyon başladı...")
for step in range(Nz):
    # O anki kesiti kaydet
    xz_profile[step, :] = np.abs(E[Ny // 2, :])**2

    # 1. IŞIK SLM'E ÇARPIYOR:
    if step == step_slm:
        E = E * slm_mask

    # 2. IŞIK LENSE ÇARPIYOR:
    if step == step_lens:
        E = E * lens_mask

    # 3. IŞIK DİFÜZÖRE ÇARPIYOR:
    if step == step_diffuser:
        E = E * diffuser_mask

    # ASM ile Havada Uçuş:
    E_k = np.fft.fft2(E)
    E_k = E_k * H_diffraction
    E = np.fft.ifft2(E_k)
    E = E * absorber

print("Simülasyon başarıyla tamamlandı!")


# =========================================================================
# 5. GÖRSELLEŞTİRME (SLM, LENS VE DİFÜZÖRÜN FİZİKSEL ÇİZİMLERİ İLE)
# =========================================================================
fig, ax1 = plt.subplots(figsize=(14, 5))

extent_zx = [0, Lz * 1e6, -Lx / 2 * 1e6, Lx / 2 * 1e6]

input_peak = np.max(xz_profile[0, :])
scaled_profile = xz_profile.T / input_peak

# Fener modunda canlı çizim:
im1 = ax1.imshow(
    scaled_profile, extent=extent_zx, aspect="auto", cmap="inferno", origin="lower", vmin=0, vmax=1.8
)

# -------------------------------------------------------------------------
# ÇİZİM 1: SLM (Pikselli Altın Çip Aynası)
# -------------------------------------------------------------------------
slm_h = 45.0
slm_w = 7.0
z_slm_c = z_slm * 1e6
ax1.fill_betweenx([-slm_h, slm_h], z_slm_c - slm_w/2, z_slm_c + slm_w/2,
                  color='#ffd700', alpha=0.55, edgecolor='white', linewidth=1.5,
                  hatch='--', label='SLM (Yukarı Bükücü)')

# -------------------------------------------------------------------------
# ÇİZİM 2: İNCE LENS (Kavisli Saydam Cam Lens)
# -------------------------------------------------------------------------
lens_h = 48.0
lens_thick = 10.0
x_curve = np.linspace(-lens_h, lens_h, 100)
z_lens_c = z_lens * 1e6
z_left = z_lens_c - lens_thick * (1 - (x_curve / lens_h)**2)
z_right = z_lens_c + lens_thick * (1 - (x_curve / lens_h)**2)

ax1.fill_betweenx(x_curve, z_left, z_right, color='cyan', alpha=0.45, 
                  edgecolor='white', linewidth=1.5, label='İnce Lens (f=300 μm)')

# -------------------------------------------------------------------------
# ÇİZİM 3: DİFÜZÖR (Desenli Buzlu Cam Plaka)
# -------------------------------------------------------------------------
diff_h = 55.0
diff_w = 6.0
z_diff_c = z_diffuser * 1e6
ax1.fill_betweenx([-diff_h, diff_h], z_diff_c - diff_w/2, z_diff_c + diff_w/2,
                  color='magenta', alpha=0.4, edgecolor='white', linewidth=1.2, 
                  hatch='//', label='Difüzör (Buzlu Cam)')

# -------------------------------------------------------------------------
# BAŞLIKLAR VE ETİKETLER
# -------------------------------------------------------------------------
ax1.set_title("Optik Sistem: SLM (Bükme) ──► Lens (Odaklama) ──► Difüzör (Saçılma)", 
              fontsize=13, fontweight='bold')
ax1.set_xlabel("İlerleme Mesafesi z (μm) ──►", fontsize=11)
ax1.set_ylabel("Enine Konum x (μm)", fontsize=11)

ax1.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
fig.colorbar(im1, ax=ax1, label="Girişe Göre Işık Şiddeti")

plt.tight_layout()
plt.show()