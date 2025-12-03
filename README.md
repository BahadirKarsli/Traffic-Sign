# 🚦 Trafik İşareti Tanıma Sistemi (Traffic Sign Recognition)

Bu proje, Otonom Araçlar ve Gelişmiş Sürücü Destek Sistemleri (ADAS) için **PyTorch** ve **CNN (Evrişimli Sinir Ağları)** kullanılarak geliştirilmiş gerçek zamanlı bir trafik işareti sınıflandırma sistemidir.

## 🎯 Proje Özeti
* **Amaç:** Sürücü hatalarını azaltmak için yol kenarındaki levhaları otomatik tespit etmek.
* **Veri Seti:** German Traffic Sign Recognition Benchmark (GTSRB).
* **Başarı Oranı:** %95.95 Test Doğruluğu.
* **Yöntem:** Derin Öğrenme (Deep Learning) - CNN.

## 📂 Dosya Yapısı
* `src/`: Kaynak kodlar ve model eğitimi not defteri.
* `data/`: Veri setinin indirileceği klasör.
* `models/`: Eğitilmiş `.pth` model dosyası.
* `requirements.txt`: Gerekli kütüphaneler.

## 🚀 Kurulum ve Çalıştırma

1. Projeyi klonlayın:
   ```bash
   git clone [https://github.com/kullaniciadi/traffic-sign-recognition.git](https://github.com/kullaniciadi/traffic-sign-recognition.git)
   cd traffic-sign-recognition

2. Gerekli kütüphaneleri yükleyin:

   ```bash
    pip install -r requirements.txt

3. Modeli eğitin veya test edin:
   * traffic_sign.ipynb dosyasını Jupyter Notebook ile açarak hücreleri çalıştırın.

📊 Sonuçlar
Modelimiz test veri seti üzerinde %95.95 doğruluk oranına ulaşmıştır.

* Precision: 0.96

* Recall: 0.96

* F1-Score: 0.96
