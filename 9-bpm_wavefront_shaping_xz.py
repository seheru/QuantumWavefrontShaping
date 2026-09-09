# ==============================================================================
# JOURNAL-GRADE 2D (x-z) WAVEFRONT SHAPING PROPAGATION
# Nature / Science Advances Standartlarında Profesyonel Görselleştirme
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------------------------------------------------------------------
# 1. PARAMETRELER VE BPM MODELİ
# ------------------------------------------------------------------------------
lambda_p = 404e-9               
k0 = 2 * np.pi / lambda_p

Nx = 512
Lx = 200e-6                     
x = np.linspace(-Lx/2, Lx/2, Nx)
dx = x[1] - x[0]
kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi

z_max = 850e-6                  
Nz = 500
z = np.linspace(0, z_max, Nz)
dz = z[1] - z[0]

# ASM Propagatörü ve Kenar Soğurucu
H_free = np.exp(-1j * (kx**2) / (2 * k0) * dz)
absorber = np.exp(- (x / (0.44 * Lx))**20)

# Optik Konumlar
z_slm     = 100e-6
z_diff    = 400e-6
z_lens    = 620e-6
f_lens    = 180e-6
z_focal   = z_lens + f_lens     # 800 um
x_target  = 20e-6               # Hedef: +20 um

# Faz Ekranları
np.random.seed(42)
corr_len = 8e-6
raw = np.random.randn(Nx)
k_filt = np.exp(-(kx**2) * (corr_len**2) / 8.0)
phi_diff = np.real(np.fft.ifft(np.fft.fft(raw) * k_filt))
phi_diff = 2.4 * (phi_diff / np.std(phi_diff))
T_diff = np.exp(1j * 2.0 * phi_diff)

T_lens = np.exp(-1j * k0 / (2 * f_lens) * (x**2))

# Faz Konjugasyonu (SLM)
phi_target_tilt = k0 * (x_target / f_lens) * x
phi_slm_opt = - (2.0 * phi_diff) + phi_target_tilt

# ------------------------------------------------------------------------------
# 2. BPM HESAPLAMA
# ------------------------------------------------------------------------------
def propagate(use_slm=False):
    w0 = 22e-6
    E = np.exp(-x**2 / (w0**2)).astype(complex)
    I_xz = np.zeros((Nx, Nz))
    
    for iz, cur_z in enumerate(z):
        if abs(cur_z - z_slm) < dz/2 and use_slm:
            E *= np.exp(1j * phi_slm_opt)
        if abs(cur_z - z_diff) < dz/2:
            E *= T_diff
        if abs(cur_z - z_lens) < dz/2:
            E *= T_lens
            
        I_xz[:, iz] = np.abs(E)**2
        E = np.fft.ifft(np.fft.fft(E) * H_free) * absorber
        
    return I_xz

print("Yayılım hesaplanıyor...")
I_before = propagate(use_slm=False)
I_after  = propagate(use_slm=True)

# ------------------------------------------------------------------------------
# 3. YAYIN KALİTESİNDE ÇİZİM (PROFESSIONAL DARK-THEME LAYOUT)
# ------------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 13
})

fig = plt.figure(figsize=(14, 8), facecolor='#0e1117')
gs = GridSpec(2, 2, width_ratios=[4, 1.2], wspace=0.15, hspace=0.35)

x_um = x * 1e6
z_um = z * 1e6
extent = [z_um[0], z_um[-1], x_um[0], x_um[-1]]
vmax_val = np.max(I_after) * 0.85

planes = [
    (z_slm*1e6, 'SLM Plane'),
    (z_diff*1e6, 'Diffuser'),
    (z_lens*1e6, 'Fourier Lens'),
    (z_focal*1e6, 'Focal Target')
]

# Dedektör kesitleri (z = 800 um)
iz_det = np.argmin(np.abs(z - z_focal))
profile_before = I_before[:, iz_det] / np.max(I_after[:, iz_det])
profile_after  = I_after[:, iz_det] / np.max(I_after[:, iz_det])

# -------------------- PANEL A: OPTİMİZASYON ÖNCESİ --------------------
ax_a = fig.add_subplot(gs[0, 0], facecolor='#0e1117')
im_a = ax_a.imshow(I_before, origin='lower', extent=extent, cmap='magma', aspect='auto', vmax=vmax_val)
ax_a.set_ylim(-60, 60)
ax_a.set_ylabel("Transverse Position x (µm)", color='white')
ax_a.tick_params(colors='white')
ax_a.set_title("(a) Unoptimized Wavefront — Random Speckle Scattering", color='white', loc='left', pad=12, fontweight='bold')

for zp, name in planes:
    ax_a.axvline(zp, color='white', ls=':', lw=1.0, alpha=0.5)
    ax_a.text(zp, 50, f" {name}", color='#cccccc', rotation=90, va='top', fontsize=8, alpha=0.85)

ax_a.axhline(x_target*1e6, color='#00ffcc', ls='--', lw=1.0, alpha=0.7)

# Sağdaki 1D Profil (Öncesi)
ax_prof_a = fig.add_subplot(gs[0, 1], facecolor='#0e1117')
ax_prof_a.plot(profile_before, x_um, color='#ff77a8', lw=1.5)
ax_prof_a.axhline(x_target*1e6, color='#00ffcc', ls='--', lw=1.0, alpha=0.8)
ax_prof_a.set_ylim(-60, 60)
ax_prof_a.set_xlim(0, 1.05)
ax_prof_a.tick_params(colors='white')
ax_prof_a.set_title("Focal Plane Slice", color='white', fontsize=10)
ax_prof_a.set_xlabel("Norm. Intensity", color='white')
ax_prof_a.grid(True, color='#252a34', ls=':')

# -------------------- PANEL B: OPTİMİZASYON SONRASI --------------------
ax_b = fig.add_subplot(gs[1, 0], facecolor='#0e1117')
im_b = ax_b.imshow(I_after, origin='lower', extent=extent, cmap='magma', aspect='auto', vmax=vmax_val)
ax_b.set_ylim(-60, 60)
ax_b.set_xlabel("Propagation Distance z (µm)", color='white')
ax_b.set_ylabel("Transverse Position x (µm)", color='white')
ax_b.tick_params(colors='white')
ax_b.set_title("(b) Optimized Wavefront — Deterministic Focus at Target", color='white', loc='left', pad=12, fontweight='bold')

for zp, name in planes:
    ax_b.axvline(zp, color='white', ls=':', lw=1.0, alpha=0.5)

ax_b.axhline(x_target*1e6, color='#00ffcc', ls='--', lw=1.0, alpha=0.7)

# Sağdaki 1D Profil (Sonrası - Keskin Tepe)
ax_prof_b = fig.add_subplot(gs[1, 1], facecolor='#0e1117')
ax_prof_b.plot(profile_after, x_um, color='#00ffcc', lw=2.0)
ax_prof_b.fill_betweenx(x_um, 0, profile_after, color='#00ffcc', alpha=0.25)
ax_prof_b.axhline(x_target*1e6, color='#00ffcc', ls='--', lw=1.0, alpha=0.8)
ax_prof_b.set_ylim(-60, 60)
ax_prof_b.set_xlim(0, 1.05)
ax_prof_b.tick_params(colors='white')
ax_prof_b.set_xlabel("Norm. Intensity", color='white')
ax_prof_b.set_title("Focal Plane Slice", color='white', fontsize=10)
ax_prof_b.grid(True, color='#252a34', ls=':')

plt.show()