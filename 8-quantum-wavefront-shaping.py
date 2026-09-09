import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==============================================================================
# 1. PHYSICAL PARAMETERS
# ==============================================================================
lambda_p = 404e-9          # pump wavelength (m)
L_z      = 2e-3            # PPKTP crystal length (m)
sigma_p  = 120e-6          # Momentumda tam çözünürlük sağlayan bel yarıçapı

# ==============================================================================
# 2. MOMENTUM & POSITION GRIDS
# ==============================================================================
k_max = 2.0e6              # rad/m
N     = 512
k     = np.linspace(-k_max, k_max, N)
dk    = k[1] - k[0]
Ks, Ki = np.meshgrid(k, k, indexing="ij")

dx = 2 * np.pi / (N * dk)
x  = (np.arange(N) - N // 2) * dx

print(f"Transverse window: +-{x.max()*1e6:.1f} um, dx = {dx*1e9:.1f} nm")

# ==============================================================================
# 3. PHASE-MATCHING FUNCTION Phi(qs - qi)
# ==============================================================================
def phase_matching(dq):
    arg = (L_z * lambda_p) / (8 * np.pi) * dq**2
    return np.sinc(arg / np.pi)

Phi = phase_matching(Ks - Ki)

# ==============================================================================
# 4. PUMP FIELD, SLM, AND ITS ANGULAR SPECTRUM v(q)
# ==============================================================================
Np      = 4096
dx_pump = (np.pi / (2 * k_max)) * 0.98
x_pump  = (np.arange(Np) - Np // 2) * dx_pump
k_pump  = np.fft.fftshift(np.fft.fftfreq(Np, d=dx_pump)) * 2 * np.pi

n_seg   = 32
slm_active_width = 2.0 * sigma_p
seg_edges = np.linspace(-slm_active_width, slm_active_width, n_seg + 1)
seg_id_fine = np.clip(np.digitize(x_pump, seg_edges) - 1, 0, n_seg - 1)
seg_id_x    = np.clip(np.digitize(x, seg_edges) - 1, 0, n_seg - 1)

def pump_field(phi_slm, on_fine_grid=True):
    if on_fine_grid:
        xx, seg = x_pump, seg_id_fine
    else:
        xx, seg = x, seg_id_x
    return np.exp(-xx**2 / (2 * sigma_p**2)) * np.exp(1j * phi_slm[seg])

def v_on_grid(phi_slm, q_values):
    E = pump_field(phi_slm, on_fine_grid=True)
    v = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E))) * dx_pump
    v_re = np.interp(q_values, k_pump, v.real)
    v_im = np.interp(q_values, k_pump, v.imag)
    return v_re + 1j * v_im

# ==============================================================================
# 5. FULLY DEVELOPED THIN DIFFUSER (Makaledeki gibi balistik bileşensiz)
# ==============================================================================
corr_len = 16e-6
raw = rng.uniform(-np.pi, np.pi, size=N)
sig_px = max(corr_len / dx, 1.0)
half = int(4 * sig_px)
kernel = np.exp(-np.arange(-half, half + 1)**2 / (2 * sig_px**2))
kernel /= kernel.sum()

# Makaledeki tam gelişmiş speckle için faz genliği artırıldı:
phi_diff = np.angle(np.convolve(np.exp(1j * raw), kernel, mode="same")) * 2.2
D = np.exp(1j * phi_diff)          # Sinyal/İdler tek foton fazı

# ==============================================================================
# 6. FORWARD MODEL
# ==============================================================================
def pump_far_field(phi_slm):
    # Pompa dalgaboyu yarı olduğu için D**2 görür:
    Ep = pump_field(phi_slm, on_fine_grid=False) * (D**2)
    far = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Ep)))
    return far

def two_photon_far_field(phi_slm):
    v = v_on_grid(phi_slm, Ks + Ki)
    psi_k = v * Phi
    psi_k /= np.sqrt(np.sum(np.abs(psi_k)**2))

    psi_x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psi_k)))
    psi_x_after = psi_x * D[:, None] * D[None, :]
    psi_far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psi_x_after)))
    return psi_far, psi_x

def simulate(phi_slm):
    pump_far = pump_far_field(phi_slm)
    I_pump = np.abs(pump_far)**2
    I_pump /= I_pump.sum()

    psi_far, psi_x = two_photon_far_field(phi_slm)
    C = np.abs(psi_far)**2
    C /= C.sum()
    return I_pump, C, psi_x

TARGET_WIN = 3

def pump_target_intensity(phi_slm, idx):
    far = pump_far_field(phi_slm)
    sl = slice(idx - TARGET_WIN, idx + TARGET_WIN + 1)
    return np.mean(np.abs(far[sl])**2)

# ==============================================================================
# 7. NUMERICAL SCHMIDT NUMBER (SVD)
# ==============================================================================
_, _, psi_x0 = simulate(np.zeros(n_seg))
U, s, Vh = np.linalg.svd(psi_x0, full_matrices=False)
lambdas = s**2
lambdas /= lambdas.sum()
K_numeric_1D = 1.0 / np.sum(lambdas**2)
print(f"Numerical Schmidt number (1D): K = {K_numeric_1D:.1f}")
print(f"Estimated 2D Schmidt number:  K = {K_numeric_1D**2:.0f} (Paper reports K ~ 680)")

# ==============================================================================
# 8. WAVEFRONT-SHAPING OPTIMIZATION
# ==============================================================================
phi_slm    = np.zeros(n_seg)
target_idx = N // 2 + 25                   # Hedef speckle momentumu q (~0.2 rad/um)
sl = slice(target_idx - TARGET_WIN, target_idx + TARGET_WIN + 1)

# Makaledeki sabit idler dedektörü: Uzak alanda optik eksen (q_i = 0):
mid_idx = N // 2

theta_scan  = np.linspace(0, 2 * np.pi, 6, endpoint=False)
n_iter      = 180
checkpoint  = 9

pump_hist, coinc_hist = [], []

I0, C0, _ = simulate(phi_slm)              # Başlangıç durumu

print(f"\nOptimizasyon başladı ({n_iter} iterasyon)... Sadece Klasik Pompa okunuyor.")

for it in range(n_iter):
    group = rng.random(n_seg) < 0.5
    vals = []
    for th in theta_scan:
        trial = phi_slm.copy()
        trial[group] += th
        vals.append(pump_target_intensity(trial, target_idx))
    vals = np.array(vals)

    A = np.stack([np.ones_like(theta_scan), np.cos(theta_scan), np.sin(theta_scan)], axis=1)
    a, bc, bs = np.linalg.lstsq(A, vals, rcond=None)[0]
    theta_opt = np.arctan2(bs, bc)
    phi_slm[group] += theta_opt

    pump_hist.append(pump_target_intensity(phi_slm, target_idx))
    
    # Makaledeki gibi: Idler uzak alanda q_i = 0'da sabitken sinyal q_s taranır: C[:, mid_idx]
    if it % checkpoint == 0 or it == n_iter - 1:
        _, C, _ = simulate(phi_slm)
        coinc_scan = C[:, mid_idx]
        coinc_hist.append((it, np.mean(coinc_scan[sl])))

I1, C1, _ = simulate(phi_slm)              # Bitiş durumu

# Makale dedeksiyonu: Sabit Idler (x_i = 0) kesiti
C0_scan = C0[:, mid_idx]
C1_scan = C1[:, mid_idx]

pump_enh  = np.mean(I1[sl]) / np.mean(I0[sl])
coinc_enh = np.mean(C1_scan[sl]) / np.mean(C0_scan[sl])

print(f"\n--> Pump intensity enhancement at target:       {pump_enh:.1f}x")
print(f"--> Coincidence rate enhancement at target:      {coinc_enh:.1f}x")
print("(Makale Denklem 3: İki güçlenme değeri birbiriyle uyumlu çıkmalıdır.)")

# Pre-optimization korelasyonu (Pompa speckle vs Kuantum speckle)
corr = np.corrcoef(I0, C0_scan)[0, 1]
print(f"Correlation coefficient (Pump vs Coincidence speckle): {corr:.2f}")

# ==============================================================================
# 9. FIGURES (MAKALEDEKİ DÜZENEKLE BİREBİR AYNI EKSENLER)
# ==============================================================================
# Uzak alan momentum ekseni (rad / µm cinsinden):
q_axis = k * 1e-6
target_q = q_axis[target_idx]

fig, axs = plt.subplots(2, 2, figsize=(11, 9))

# Sol Üst: Pompa - Önce
axs[0, 0].plot(q_axis, I0 / I0.max(), lw=1.5)
axs[0, 0].axvline(target_q, color="r", ls="--", lw=1.5, label="Target")
axs[0, 0].set_title("Pump far-field intensity - before")
axs[0, 0].set_xlabel(r"$q_p$ (rad/µm)"); axs[0, 0].set_ylabel("normalized")
axs[0, 0].legend()

# Sağ Üst: Kuantum Çakışma Taraması C(q_s, q_i=0) - Önce
axs[0, 1].plot(q_axis, C0_scan / C0_scan.max(), color="tab:purple", lw=1.5)
axs[0, 1].axvline(target_q, color="r", ls="--", lw=1.5, label="Target")
axs[0, 1].set_title(r"Coincidence scan $C(q_s, q_i=0)$ - before")
axs[0, 1].set_xlabel(r"$q_s$ (rad/µm)"); axs[0, 1].set_ylabel("normalized")
axs[0, 1].legend()

# Sol Alt: Pompa - Sonra (ODAKLANMIŞ)
axs[1, 0].plot(q_axis, I1 / I1.max(), color="tab:orange", lw=1.5)
axs[1, 0].axvline(target_q, color="r", ls="--", lw=1.5, label="Target")
axs[1, 0].set_title(f"Pump far-field intensity - after ({pump_enh:.1f}x at target)")
axs[1, 0].set_xlabel(r"$q_p$ (rad/µm)"); axs[1, 0].set_ylabel("normalized")
axs[1, 0].legend()

# Sağ Alt: Kuantum Çakışma Taraması C(q_s, q_i=0) - Sonra (RECOVERED)
axs[1, 1].plot(q_axis, C1_scan / C1_scan.max(), color="tab:red", lw=1.5)
axs[1, 1].axvline(target_q, color="r", ls="--", lw=1.5, label="Target")
axs[1, 1].set_title(r"Coincidence scan $C(q_s, q_i=0)$ - after " + f"({coinc_enh:.1f}x at target)")
axs[1, 1].set_xlabel(r"$q_s$ (rad/µm)"); axs[1, 1].set_ylabel("normalized")
axs[1, 1].legend()

plt.tight_layout()

# --- Fig 2: Gerçek Zamanlı Optimizasyon Eğrisi (Makale Fig. 3) ---------------
fig2, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(np.arange(n_iter), pump_hist, color="tab:blue", lw=2, label="Pump feedback (used)")
ax1.set_xlabel("Iteration", fontsize=11)
ax1.set_ylabel("Pump intensity at target (arb.)", color="tab:blue", fontsize=11)
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
cx, cy = zip(*coinc_hist)
ax2.plot(cx, cy, "ro-", lw=1.8, label="Coincidence rate (never measured in feedback)")
ax2.set_ylabel("Coincidence rate at target (arb.)", color="tab:red", fontsize=11)
ax2.tick_params(axis="y", labelcolor="tab:red")

ax1.set_title("Real-Time Shaping: Classical Pump Drives Quantum Coincidence", fontsize=12, fontweight='bold')
fig2.tight_layout()

plt.show()