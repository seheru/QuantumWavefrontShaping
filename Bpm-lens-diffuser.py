import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. PARAMETRELER VE IZGARALAR
# =========================================================================
wavelength = 1.0e-6          # 1 mikron
k0 = 2 * np.pi / wavelength
n0 = 1.0                     # HAVA
k = n0 * k0

Lx = 200.0e-6                # 200 mikron pencere
Ly = 200.0e-6
Nx = 256
Ny = 256
dx = Lx / Nx
dy = Ly / Ny

Lz = 1000.0e-6               # 1000 mikron toplam yol
Nz = 400
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
# 2. LENS VE GERÇEKÇİ DİFÜZÖRÜN HAZIRLANMASI
# =========================================================================
# A) İNCE LENS: Odak Uzaklığı f = 350 mikron (z = 150 um noktasına koyduk)
f_lens = 350.0e-6
lens_mask = np.exp(-1j * (k0 / (2 * f_lens)) * R_sq)
z_lens = 150.0e-6
step_lens = int(z_lens / dz)

# B) GERÇEKÇİ DİFÜZÖR (z = 400 um noktasına koyduk)
np.random.seed(42)
raw_noise = np.random.randn(Ny, Nx)
sigma_f = 1 / (6.0e-6)
filter_2d = np.exp(- (FX**2 + FY**2) / (2 * sigma_f**2))
smooth_phase = np.real(np.fft.ifft2(np.fft.fft2(raw_noise) * filter_2d))
smooth_phase = (smooth_phase / np.std(smooth_phase)) * 2.2

diffuser_mask = np.exp(1j * smooth_phase)
z_diffuser = 600.0e-6
step_diffuser = int(z_diffuser / dz)


# =========================================================================
# 3. BAŞLANGIÇ FENERİ (Güçlü ve Geniş Lazer Işını)
# =========================================================================
w0 = 25.0e-6                 # 25 mikron bel yarıçaplı fener demeti
E = np.exp(-R_sq / (w0**2)).astype(np.complex128)


# =========================================================================
# 4. SİMÜLASYON DÖNGÜSÜ
# =========================================================================
xz_profile = np.zeros((Nz, Nx))

print("Simülasyon başladı...")
for step in range(Nz):
    xz_profile[step, :] = np.abs(E[Ny // 2, :])**2

    # Lense Çarpma
    if step == step_lens:
        E = E * lens_mask

    # Difüzöre Çarpma
    if step == step_diffuser:
        E = E * diffuser_mask

    # ASM ile Uçuş
    E_k = np.fft.fft2(E)
    E_k = E_k * H_diffraction
    E = np.fft.ifft2(E_k)
    E = E * absorber

print("Simülasyon tamamlandı!")


# =========================================================================
# 5. GÖRSELLEŞTİRME (LENS VE DİFÜZÖRÜN FİZİKSEL ÇİZİMİ İLE)
# =========================================================================
fig, ax1 = plt.subplots(figsize=(13, 5))

extent_zx = [0, Lz * 1e6, -Lx / 2 * 1e6, Lx / 2 * 1e6]

# Işık yayılımını çiziyoruz:
input_peak = np.max(xz_profile[0, :])
scaled_profile = xz_profile.T / input_peak

im1 = ax1.imshow(
    scaled_profile, extent=extent_zx, aspect="auto", cmap="inferno", origin="lower", vmin=0, vmax=1.8
)

# -------------------------------------------------------------------------
# İŞTE GERÇEK LENS ÇİZİMİ (İki Tarafı Şişkin Biconvex Cam Lens)
# -------------------------------------------------------------------------
lens_h = 50.0         # Lensin enine yüksekliği (±45 mikron)
lens_thick = 5.0     # Lensin ortasındaki cam kalınlığı (10 mikron)
x_curve = np.linspace(-lens_h, lens_h, 100)

# Lensin sol ve sağ kavisli küresel yüzeyleri:
z_lens_center = z_lens * 1e6
z_left = z_lens_center - lens_thick * (1 - (x_curve / lens_h)**2)
z_right = z_lens_center + lens_thick * (1 - (x_curve / lens_h)**2)

# İki kavisin arasını saydam cam mavisiyle boyuyoruz:
ax1.fill_betweenx(x_curve, z_left, z_right, color='white', alpha=0.45, 
                  edgecolor='white', linewidth=1.5, label='İnce Lens (f=350 μm)')


# -------------------------------------------------------------------------
# DİFÜZÖR ÇİZİMİ (Saydam Buzlu Cam Plaka)
# -------------------------------------------------------------------------
diff_h = 50.0         # Difüzör yüksekliği
diff_thick = 6.0      # Plaka kalınlığı
z_diff_center = z_diffuser * 1e6

ax1.fill_betweenx([-diff_h, diff_h], z_diff_center - diff_thick/2, z_diff_center + diff_thick/2,
                  color='magenta', alpha=0.4, edgecolor='white', linewidth=1.2, 
                  hatch='//', label='Difüzör (Buzlu Cam)')


# -------------------------------------------------------------------------
# BAŞLIKLAR VE ETİKETLER
# -------------------------------------------------------------------------
ax1.set_title("Lazerin Optik Sistemden Geçişi (Lens + Difüzör)", fontsize=13, fontweight='bold')
ax1.set_xlabel("İlerleme Mesafesi z (μm) ──►", fontsize=11)
ax1.set_ylabel("Enine Konum x (μm)", fontsize=11)

# Lejant kutusu
ax1.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
fig.colorbar(im1, ax=ax1, label="Girişe Göre Işık Şiddeti")

plt.tight_layout()
plt.show()