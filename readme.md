# Beam Propagation Method (BPM) Simülasyonları

Bu projede Açısal Spektrum Yöntemi (ASM) ve Split-Step BPM kullanılarak geliştirilen optik simülasyonlar ve çıktı grafikleri yer almaktadır.

---

### 1. Serbest Uzayda Lazer Yayılımı (`Bpm.py`)
Gauss lazer demetinin havada doğal kırınımı ve genişlemesi:

![Serbest Uzay Yayılımı](images/BPM.png)

---

### 2. Lens ve Difüzör Modeli (`Bpm-lens-diffuser.py`)
Lazerin lense çarparak odaklanması ve ardından difüzörden geçerek saçılması:

![Lens ve Difüzör](images/BPM-lens-diffusor.png)

---

### 3. SLM + Lens + Difüzör Düzeni (`Bpm-SLM.py`)
Lazerin SLM ile yukarı saptırılması (Beam Steering), lensle toplanması ve difüzörden saçılması:

![SLM Lens ve Difüzör](images/BPM-SLM.png)

---

### 4. SPDC Süreci - Çift Gauss Yaklaşımı ile EPR Dolanıklığı (`spdc-doublegaussian.py`)
Spontan Parametrik Aşağı Dönüşüm (SPDC) ile üretilen foton çiftlerinin sürekli değişkenli (Continuous-Variable) konum ve momentum kuantum dolanıklığının (EPR Hali) analitik modellenmesi:

* **Konum Uzayı ($x_s = x_i$):** Fotonların kristal içerisinde mikroskobik olarak aynı noktada eş-zamanlı doğumu (pozitif korelasyon).
* **Momentum Uzayı ($q_s = -q_i$):** Enine momentum korunumu gereği fotonların zıt yönlerde saçılması (ters korelasyon).
* **Dolanıklık Ölçütü:** Schmidt sayısı ($K \approx 4.46$) ve Heisenberg sınırını aşan $\Delta(x_s|x_i)\Delta(q_s|q_i) \ll 1/2$ koşullu varyansları ile EPR paradoksunun gösterimi.

![SPDC Double Gaussian](images/spdc-doublegaussian.png)

---

### 5. SPDC Süreci - Sinc-Gauss Modeli ve Kırınım Analizi (`spdc-sincgaussian.py`)
Kristalin sonlu boyutu ($L$) ve fiziksel faz uyumunun $\text{sinc}$ profili ile modellenmesi; 2D Hızlı Fourier Dönüşümü (2D-IFFT) ile gerçek kırınım saçaklarının simülasyonu:

* **Momentum Uzayı:** $\text{sinc}(\Delta k_z L / 2)$ faz uyumu fonksiyonu kaynaklı salınımlı kırınım halkaları ve yan loblar (side-lobes).
* **Konum Uzayı (2D-FFT):** İdeal Gaussiyen yerine kırınım etkilerini ve gerçekçi uzaysal korelasyon kuyruklarını içeren konum profili.

![SPDC Sinc-Gauss](images/spdc-sincgaussian.png)