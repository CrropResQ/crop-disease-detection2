import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
class_names):
    model.to(device)
    scaler = GradScaler()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        # --- TRAINING --
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        history['train_loss'].append(train_loss / total)
        history['train_acc'].append(correct / total)
        # --- VALIDATION --
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        history['val_loss'].append(val_loss / total)
        history['val_acc'].append(correct / total)
        print(f"Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")
    # --- FINAL METRICS & PLOTS --
    print("\n--- Final Model Evaluation ---")
    calculate_and_print_metrics(all_labels, all_preds, class_names)
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_learning_curves(history)
    return model, history