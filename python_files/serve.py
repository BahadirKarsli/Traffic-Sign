import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F

# 1. Model Sınıfını İçe Aktarma
from model import TrafficSignNet

# 2. Cihaz Ayarı (Inference için CPU yeterlidir ve daha güvenlidir)
device = torch.device('cpu')

# 3. Modeli Yükleme
model_path = "traffic_sign_cnn.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

try:
    model = TrafficSignNet()
    # Ağırlıkları yükle
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    raise e

# 4. Sınıf İsimleri
classes = {
    0: 'Hız Limiti (20km/s)', 1: 'Hız Limiti (30km/s)', 2: 'Hız Limiti (50km/s)',
    3: 'Hız Limiti (60km/s)', 4: 'Hız Limiti (70km/s)', 5: 'Hız Limiti (80km/s)',
    6: 'Hız Limiti Sonu (80km/s)', 7: 'Hız Limiti (100km/s)', 8: 'Hız Limiti (120km/s)',
    9: 'Geçiş Yok', 10: 'Kamyonlar için geçiş yok', 11: 'Ana yol tali yol kavşağı',
    12: 'Anayol', 13: 'Yol Ver', 14: 'DUR (Stop)', 15: 'Taşıt Giremez',
    16: 'Kamyon Giremez', 17: 'Girişi Olmayan Yol', 18: 'Dikkat',
    19: 'Sola Tehlikeli Viraj', 20: 'Sağa Tehlikeli Viraj', 21: 'Birbiri ardına tehlikeli virajlar',
    22: 'Engebeli Yol', 23: 'Kaygan Yol', 24: 'Sağdan Daralan Yol', 25: 'Yol Çalışması',
    26: 'Trafik Işıkları', 27: 'Yaya Geçidi', 28: 'Okul Geçidi', 29: 'Bisiklet Geçidi',
    30: 'Buzlanma Uyarısı', 31: 'Vahşi Hayvan Çıkabilir', 32: 'Hız Sınırı ve Yasaklar Sonu',
    33: 'Sağa Mecburi Yön', 34: 'Sola Mecburi Yön', 35: 'İleri ve Sağa Mecburi Yön',
    36: 'İleri ve Sola Mecburi Yön', 37: 'Sola Dönüş Yok', 38: 'Sağa Dönüş Yok',
    39: 'Sol Tarafı İzleyiniz', 40: 'Sağ Tarafı İzleyiniz', 41: 'Geçiş Yasağı Sonu',
    42: 'Kamyonlar için geçiş yasağı sonu'
}

# 5. Görüntü Ön İşleme (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 6. Tahmin Fonksiyonu
def predict_image(image):
    if image is None:
        return None

    try:
        # Resmi PIL formatına çevirme (Eğer değilse)
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        # Ön işleme uygulama
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Tahmin yapma
        with torch.no_grad():
            outputs = model(image_tensor)
            # Softmax ile olasılık değerlerine çevir
            probabilities = F.softmax(outputs, dim=1)[0]

        # En yüksek 3 tahmini döndürme
        confidences = {classes[i]: float(probabilities[i]) for i in range(len(classes))}

        return confidences

    except Exception as e:
        return f"Hata: {str(e)}"


# 7. Gradio Arayüzünü Oluştur
interface = gr.Interface(
    fn=predict_image,  # Çalışacak fonksiyon
    inputs=gr.Image(type="numpy", label="Trafik İşaretini Yükleyin"),  # Girdi türü: Resim
    outputs=gr.Label(num_top_classes=3, label="Tahmin Sonuçları"),  # Çıktı türü: Etiket ve Olasılık barı
    title="🚦 Trafik İşareti Tanıma Sistemi",
    description="Bir trafik levhası resmi yükleyin, Yapay Zeka (CNN) modelimiz onun ne olduğunu tahmin etsin.",
    examples=[
        # Buraya test etmek için örnek resim yolları yazabilirsin varsa.
        ["test_images/30sign.jpg"],
        ["test_images/stopsign.jpg"],
        ["test_images/dlcsign.png"]
    ],
)

# 8. Uygulamayı Başlat
if __name__ == "__main__":
    print("Arayüz başlatılıyor... Linke tıklayarak tarayıcıda açabilirsiniz.")
    interface.launch(share=False)

