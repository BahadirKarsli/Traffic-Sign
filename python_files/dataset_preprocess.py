# Dönüşümler (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((32, 32)),     # Tüm görselleri 32x32 piksel yap
    transforms.ToTensor(),           # Tensör formatına çevir (0-1 arasına sıkıştırır)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizasyon (-1 ile 1 arasına)
])

# NOT: GTSRB veri setini PyTorch otomatik indirebilir.
train_dataset = torchvision.datasets.GTSRB(
    root='./data', 
    split='train', 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.GTSRB(
    root='./data', 
    split='test', 
    download=True, 
    transform=transform
)

# DataLoader (Veriyi batch'ler halinde modele besler)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"Eğitim Verisi Sayısı: {len(train_dataset)}")
print(f"Test Verisi Sayısı: {len(test_dataset)}")
