from PIL import Image
import matplotlib.pyplot as plt

# 1. GTSRB Sınıf İsimleri (Modelin çıktısını Türkçeye çevirmek için)
classes = { 
    0:'Hız Limiti (20km/s)',
    1:'Hız Limiti (30km/s)', 
    2:'Hız Limiti (50km/s)', 
    3:'Hız Limiti (60km/s)', 
    4:'Hız Limiti (70km/s)', 
    5:'Hız Limiti (80km/s)', 
    6:'Hız Limiti Sonu (80km/s)', 
    7:'Hız Limiti (100km/s)', 
    8:'Hız Limiti (120km/s)', 
    9:'Geçiş Yok', 
    10:'Kamyonlar için geçiş yok', 
    11:'Ana yol tali yol kavşağı', 
    12:'Anayol', 
    13:'Yol Ver', 
    14:'DUR (Stop)', 
    15:'Taşıt Giremez', 
    16:'Kamyon Giremez', 
    17:'Girişi Olmayan Yol', 
    18:'Dikkat', 
    19:'Sola Tehlikeli Viraj', 
    20:'Sağa Tehlikeli Viraj', 
    21:'Birbiri ardına tehlikeli virajlar', 
    22:'Engebeli Yol', 
    23:'Kaygan Yol', 
    24:'Sağdan Daralan Yol', 
    25:'Yol Çalışması', 
    26:'Trafik Işıkları', 
    27:'Yaya Geçidi', 
    28:'Okul Geçidi', 
    29:'Bisiklet Geçidi', 
    30:'Buzlanma Uyarısı', 
    31:'Vahşi Hayvan Çıkabilir', 
    32:'Hız Sınırı ve Yasaklar Sonu', 
    33:'Sağa Mecburi Yön', 
    34:'Sola Mecburi Yön', 
    35:'İleri ve Sağa Mecburi Yön', 
    36:'İleri ve Sola Mecburi Yön', 
    37:'Sola Dönüş Yok', 
    38:'Sağa Dönüş Yok', 
    39:'Sol Tarafı İzleyiniz', 
    40:'Sağ Tarafı İzleyiniz', 
    41:'Geçiş Yasağı Sonu', 
    42:'Kamyonlar için geçiş yasağı sonu' 
}

def predict_external_image(image_path):
    # Resmi aç ve RGB'ye çevir (Bazı resimler RGBA olabilir, hata vermemesi için)
    img = Image.open(image_path).convert('RGB')
    
    # NOT: Eğitimdeki transform değişkenini aynen kullanıyorum
    img_transformed = transform(img)
    
    # Modelin beklediği batch boyutunu ekleme işlemi: (3, 32, 32) -> (1, 3, 32, 32)
    img_batch = img_transformed.unsqueeze(0).to(device)
    
    # Tahmin yapma
    model.eval()
    with torch.no_grad():
        outputs = model(img_batch)
        _, predicted = torch.max(outputs, 1)
        
    class_id = predicted.item()
    class_name = classes[class_id]
    
    # Sonucu Görselleştirme
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Tahmin: {class_name} \n(Sınıf ID: {class_id})", color='green', fontsize=14)
    plt.show()
    
    return class_name

# --- TEST KISMI ---
# indirilen resimlerin path'i
image_path = "dlcsign.png"
try:
    prediction = predict_external_image(image_path)
    print(f"Modelin Tahmini: {prediction}")
except FileNotFoundError:
    print(f"Hata: '{image_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
