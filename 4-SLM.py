import cv2
import numpy as np

# =========================================================================
# 1. SLM DONANIM VE EKRAN AYARLARI
# =========================================================================
# Laboratuvardaki SLM'in gerçek piksel çözünürlüğü (Gerekirse değiştirebilirsin)
SLM_WIDTH = 1920
SLM_HEIGHT = 1080

# 2. Monitörün Başlangıç Konumu:
# Eğer laptop ekranın 1920x1080 ise, 2. ekran (SLM) x=1920 noktasından başlar.
# (Eğer tek ekranda test ediyorsan bu değeri 0 yapabilirsin).
SLM_X_OFFSET = 1920  
SLM_Y_OFFSET = 0

# Koordinat Izgarası (-1 ile +1 arası normalize)
x = np.linspace(-1, 1, SLM_WIDTH)
y = np.linspace(-1, 1, SLM_HEIGHT)
X, Y = np.meshgrid(x, y)
R_sq = X**2 + Y**2
Theta = np.arctan2(Y, X)  # Vorteks için açı haritası


# =========================================================================
# 2. FAZI 8-BIT GRİ RESME ÇEVİREN FONKSİYON
# =========================================================================
def phase_to_grayscale(phi):
    """
    Radyan cinsinden faz matrisini [0 - 2pi] sarar ve
    SLM'in anlayacağı [0 - 255] uint8 gri resme dönüştürür.
    """
    wrapped_phase = phi % (2 * np.pi)
    gray_image = (wrapped_phase / (2 * np.pi) * 255).astype(np.uint8)
    return gray_image


# =========================================================================
# 3. İNTERAKTİF FAZ DESENİ SEÇENEKLERİ
# =========================================================================
def get_pattern(mode):
    if mode == 'flat':
        # 0. Düz Ayna Modu (Sıfır Faz)
        return np.zeros((SLM_HEIGHT, SLM_WIDTH))
    
    elif mode == 'steering':
        # 1. Prizma: Işığı Sağa Bükme (Linear Phase Ramp)
        return 40.0 * X
    
    elif mode == 'lens':
        # 2. Dijital Lens: Işığı Odaklama (Fresnel Parabolü)
        return -35.0 * R_sq
    
    elif mode == 'vortex':
        # 3. Optik Vorteks: Ortası Delik Donut Işını (OAM / Spiral Faz)
        topological_charge = 3  # Donut derecesi
        return topological_charge * Theta
    
    elif mode == 'lens_and_steering':
        # 4. Hem Bükme Hem Odaklama
        return (30.0 * X) - (25.0 * R_sq)


# =========================================================================
# 4. SLM PENCERESİNİ VE ÖNİZLEMEYİ AÇMA
# =========================================================================
# 1. SLM İçin Tam Ekran Penceresi (2. Monitöre Yansıtılır)
cv2.namedWindow("SLM_Projector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("SLM_Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow("SLM_Projector", SLM_X_OFFSET, SLM_Y_OFFSET)

# 2. Senin Laptop Ekranın İçin Canlı Önizleme Penceresi
cv2.namedWindow("Laptop_Onizleme", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Laptop_Onizleme", 640, 360)


# =========================================================================
# 5. CANLI KONTROL DÖNGÜSÜ
# =========================================================================
current_mode = 'flat'
print("\n" + "="*50)
print("🎛️  SLM LABORATUVAR KONTROL PANELİ BAŞLADI")
print("="*50)
print("Klavyeden tuşlara basarak lazeri kontrol edebilirsin:")
print(" [0] -> Düz Ayna Modu (Sıfır Faz)")
print(" [1] -> Işığı Sağa Bük (Prizma / Steering)")
print(" [2] -> Işığı Odakla (Dijital Lens)")
print(" [3] -> Donut / Vorteks Işını Üret (OAM)")
print(" [4] -> Hem Bük Hem Odakla")
print(" [q] -> Programdan Çık")
print("="*50 + "\n")

while True:
    # 1. Seçili moda göre fazı üret
    phi = get_pattern(current_mode)
    
    # 2. 8-bit gri resme çevir
    gray_frame = phase_to_grayscale(phi)
    
    # 3. Resmi SLM'e yansıt
    cv2.imshow("SLM_Projector", gray_frame)
    
    # 4. Kendi ekranında önizle
    cv2.imshow("Laptop_Onizleme", gray_frame)
    
    # 5. Klavyeden gelen tuş komutunu dinle (10 ms bekle)
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):
        print("Program kapatılıyor...")
        break
    elif key == ord('0'):
        current_mode = 'flat'
        print(">> Mod Değişti: Düz Ayna (Sıfır Faz)")
    elif key == ord('1'):
        current_mode = 'steering'
        print(">> Mod Değişti: Işık Sağa Bükülüyor (Prizma)")
    elif key == ord('2'):
        current_mode = 'lens'
        print(">> Mod Değişti: Işık Odaklanıyor (Dijital Lens)")
    elif key == ord('3'):
        current_mode = 'vortex'
        print(">> Mod Değişti: Donut / Vorteks Işını Üretiliyor")
    elif key == ord('4'):
        current_mode = 'lens_and_steering'
        print(">> Mod Değişti: Hem Bükme Hem Odaklama")

# Kapanış
cv2.destroyAllWindows()
print("Bağlantı güvenle sonlandırıldı.")