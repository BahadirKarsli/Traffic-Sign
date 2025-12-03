criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Grafikler için verileri tuttuğum listeler
loss_list = []
accuracy_list = []

print("Eğitim Başlıyor...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass (İleri yayılım)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass (Geri yayılım ve optimizasyon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Doğruluk hesaplama (Anlık takip için)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_acc = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    
    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: %{epoch_acc:.2f}')

print("Eğitim Tamamlandı.")
