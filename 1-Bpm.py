import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. FİZİKSEL PARAMETRELER VE SABİTLER
# =========================================================================
wavelength = 1.0e-6          # Işığın dalga boyu: 1 mikron (1000 nm)
k0 = 2 * np.pi / wavelength   # Havada dalga sayısı
n0 = 1.0                     # Ortamın kırıcılık indisi (HAVA)
k = n0 * k0                  # Ortamdaki dalga sayısı

# Enine Boyutlar (X ve Y Ekseni)
Lx = 100.0e-6                # Pencere genişliği: 300 mikron
Ly = 100.0e-6                # Pencere yüksekliği: 300 mikron
Nx = 256                     # Piksel sayısı
Ny = 256
dx = Lx / Nx                 # Piksel çözünürlüğü
dy = Ly / Ny

# Boyuna Boyutlar (Z Ekseni - İlerleme Yolu)
Lz = 1000.0e-6               # Toplam ilerleme mesafesi: 1 mm (1000 mikron)
Nz = 200                     # Z eksenindeki adım sayısı
dz = Lz / Nz                 # Her bir adımın boyutu (5 mikron)


# =========================================================================
# 2. TÜM IZGARALARIN OLUŞTURULMASI (Gerçek Uzay + Frekans Uzayı)
# =========================================================================
# A) Gerçek Uzay Izgarası (X, Y)
x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)
R_sq = X**2 + Y**2           # Merkeze olan uzaklığın karesi (Pisagor)

# B) Frekans Uzayı Izgarası (kx, ky - Açı Ağı)
fx = np.fft.fftfreq(Nx, d=dx)
fy = np.fft.fftfreq(Ny, d=dy)
FX, FY = np.meshgrid(fx, fy)
kx = 2 * np.pi * FX
ky = 2 * np.pi * FY
k_transverse_sq = kx**2 + ky**2  # Enine açıların karesi (kx^2 + ky^2)
#k_transverse_sq tek bir toplam değil; her bir açının ayrı ayrı kx2+ky2 değerini tutan 2 Boyutlu bir matristir (tablodur).


# =========================================================================
# 3. ŞİMDİ LAZERİ BAŞLATIYORUZ (Giriş Alanı E)
# =========================================================================
w0 = 10.0e-6                  # Lazerin başlangıç bel yarıçapı: 5 mikron
E = np.exp(-R_sq / (w0**2)).astype(np.complex128)  # Kompleks Gauss Lazer Demeti
input_intensity_xy = np.abs(E)**2  # <-- Girişteki fotoğrafı hafızaya aldık!

# =========================================================================
# 4. YAYILIM OPERATÖRÜ VE YUTUCU MASKENİN HAZIRLANMASI
# =========================================================================
# A) kz Hesabı ve ASM Yayılım Operatörü (H) burası bi fonksiyon gibi çalışıyor. döngüde kullanılmak üzere hazırlanıyor.
propagating_mask = k_transverse_sq <= k**2  # Sadece propagasyon yapan dalgaları seçmek için maske (true false verir)
kz = np.zeros_like(k_transverse_sq) #kz için boş matris oluşturuyoruz 
kz[propagating_mask] = np.sqrt(k**2 - k_transverse_sq[propagating_mask])
H_diffraction = np.exp(1j * kz * dz) * propagating_mask  # Havada dz kadar uçuran operatör

# B) Süper-Gauss Kenar Yutucu Maske (Pac-Man ve yansıma önleyici)
wx_mask = 0.85 * (Lx / 2)
wy_mask = 0.85 * (Ly / 2)
m = 10
absorber = np.exp(- (X / wx_mask)**(2 * m) - (Y / wy_mask)**(2 * m))

###yarım adım faz operatörü (phase screen) için
#delta_n=np.zeros((Ny,Nx)) ,  delta_n=1.0e-6*np.random.randn(Ny,Nx) (gerçek hava için tribülanslı)
#phase_screen_half=np.exp(1j*k0*delta_n*dz/2)  # yarım adım faz ekranı (phase screen)


# =========================================================================
# 5. SİMÜLASYON DÖNGÜSÜ (Işığı Adım Adım İlerletme)
# =========================================================================
xz_profile = np.zeros((Nz, Nx))  # 2D yayılım kesitini kaydetmek için hafıza
mid_intensity_xy = None


#yz de aynı çizimi veriri çünkü lazerimiz yuvarlak. ama xy profili daha değişik olur.
print("Simülasyon başladı...")
for step in range(Nz):
    # Orta çizgiden (y = 0) bir yatay kesit kaydediyoruz (çizim için)
    xz_profile[step, :] = np.abs(E[Ny // 2, :])**2

    # Tam ortadaki (z = 500 um) fotoğrafı çek:
    if step == Nz // 2:
        mid_intensity_xy = np.abs(E)**2


    #E=E*phase_screen_half

    # 1. Fourier Dönüşümü ile Frekans Uzayına Geç (Açılara Ayır)
    E_k = np.fft.fft2(E)

    # 2. ASM ile Havada dz kadar Uçur
    E_k = E_k * H_diffraction

    # 3. Ters Fourier ile Gerçek Uzaya Geri Dön (Girişim Yaptır)
    E = np.fft.ifft2(E_k)

    #E=E*phase_screen_half

    # 4. Kenarlardan taşan ışığı süngerle yut
    E = E * absorber

# Çıkıştaki X-Y profilini kaydediyoruz (z = Lz)
output_intensity_xy = np.abs(E)**2

print("Simülasyon başarıyla tamamlandı!")




# 6. EN BAŞTAKİ ORİJİNAL ÇİZİM (Siyah Ekranda Beyaz Işık)
# =========================================================================
fig, ax1 = plt.subplots(figsize=(12, 5))

# Cetvel: [z_baslangic, z_bitis, x_alt, x_ust] (mikron cinsinden)
extent_zx = [0, Lz * 1e6, -Lx / 2 * 1e6, Lx / 2 * 1e6]

# Siyah zemin üzerine beyaz ışık (cmap='gray')
im1 = ax1.imshow(
    xz_profile.T, extent=extent_zx, aspect="auto", cmap="inferno", origin="lower" 
)

ax1.set_title(
    "Lazerin İlerlemesi (Yandan Görünüş - XZ Profili)",
    fontsize=12,
    fontweight="bold",
)
ax1.set_xlabel("İlerleme Mesafesi z (μm) ──►")
ax1.set_ylabel("Enine Konum x (μm)")
fig.colorbar(im1, ax=ax1, label="Işık Şiddeti")

plt.tight_layout()
plt.show()