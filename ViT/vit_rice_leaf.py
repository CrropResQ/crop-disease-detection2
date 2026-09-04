# ==========================================================
# ViT - Rice Leaf Disease Classification (Kaggle)
# Designed to produce outputs similar to the SigLIP experiment
# ==========================================================

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages


# ==========================================================
# 1. REPRODUCIBILITY
# ==========================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================
DATASET_PATH = "/kaggle/input/datasets/varun2ks05/rice-leaf-aug/Rice_Leaf_AUG"

IMAGE_SIZE = 224
BATCH_SIZE = 16          # Safer for Tesla T4 than 32
TOTAL_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

OUTPUT_DIR = "ViT"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 3. DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ViT RICE LEAF DISEASE CLASSIFICATION")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================================
# 4. DATA TRANSFORMS
# ==========================================================
# ImageNet normalization is appropriate for a pretrained timm ViT.
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# 5. LOAD DATASET
# ==========================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATASET_PATH}\n\n"
        "Check that the Kaggle dataset is attached to this notebook."
    )

print(f"\nDataset path: {DATASET_PATH}")

full_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=data_transforms
)

class_names = full_dataset.classes
num_classes = len(class_names)

print(f"\nClasses ({num_classes}):")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print(f"\nTotal images: {len(full_dataset)}")


# ==========================================================
# 6. 80/20 TRAIN-VALIDATION SPLIT
# ==========================================================
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"Training images:   {train_size}")
print(f"Validation images: {val_size}")


# ==========================================================
# 7. DATA LOADERS
# ==========================================================
pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=pin_memory
)


# ==========================================================
# 8. CREATE PRETRAINED ViT
# ==========================================================
print("\nLoading pretrained ViT...")

vit_model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes
)

vit_model = vit_model.to(device)

print("ViT model loaded successfully.")


# ==========================================================
# 9. TRAINING FUNCTION
# ==========================================================
def train_vit(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_state = None

    start_time = time.time()

    print(f"\nStarting training for {TOTAL_EPOCHS} epochs...\n")

    for epoch in range(TOTAL_EPOCHS):

        # --------------------------------------------------
        # TRAINING
        # --------------------------------------------------
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:02d}/{TOTAL_EPOCHS} [Train]"
        )

        for images, labels in train_bar:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total


        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        epoch_preds = []
        epoch_labels = []

        with torch.no_grad():

            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1:02d}/{TOTAL_EPOCHS} [Val]"
            )

            for images, labels in val_bar:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

                epoch_preds.extend(
                    predictions.cpu().numpy()
                )

                epoch_labels.extend(
                    labels.cpu().numpy()
                )

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"\nEpoch {epoch + 1:02d}/{TOTAL_EPOCHS} Summary -> "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}\n"
        )

        # Save best model in memory
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    total_training_time = time.time() - start_time

    # Restore best validation model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final prediction using best model
    model.eval()

    final_preds = []
    final_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)

            predictions = outputs.argmax(dim=1)

            final_preds.extend(
                predictions.cpu().numpy()
            )

            final_labels.extend(
                labels.numpy()
            )

    return (
        model,
        history,
        final_labels,
        final_preds,
        total_training_time,
        best_val_acc
    )


# ==========================================================
# 10. TRAIN
# ==========================================================
(
    trained_model,
    history,
    y_true,
    y_pred,
    total_training_time,
    best_val_acc
) = train_vit(
    vit_model,
    train_loader,
    val_loader
)


# ==========================================================
# 11. METRICS
# ==========================================================
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

class_accuracy = cm.diagonal() / np.where(
    cm.sum(axis=1) == 0,
    1,
    cm.sum(axis=1)
)

hours, remainder = divmod(total_training_time, 3600)
minutes, seconds = divmod(remainder, 60)


# ==========================================================
# 12. PRINT FINAL RESULTS
# ==========================================================
print("\n" + "=" * 60)
print("FINAL ViT EVALUATION METRICS")
print("=" * 60)

print(
    f"Total Training Time : "
    f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"
)

print(
    f"Overall Accuracy    : "
    f"{accuracy * 100:.2f}% ({accuracy:.4f})"
)

print(f"Weighted Precision  : {precision:.4f}")
print(f"Weighted Recall     : {recall:.4f}")
print(f"Weighted F1 Score   : {f1:.4f}")
print(f"Best Validation Acc : {best_val_acc * 100:.2f}%")

print("\nCLASS-WISE ACCURACY")
print("-" * 40)

for i, name in enumerate(class_names):
    print(
        f"{name:<30}: "
        f"{class_accuracy[i] * 100:.2f}% "
        f"({class_accuracy[i]:.4f})"
    )


# ==========================================================
# 13. SAVE MAIN METRICS TXT
# ==========================================================
metrics_path = os.path.join(
    OUTPUT_DIR,
    "vit_metrics.txt"
)

with open(metrics_path, "w", encoding="utf-8") as f:

    f.write("FINAL ViT EVALUATION METRICS\n")
    f.write("=" * 50 + "\n")

    f.write(
        f"Model: vit_base_patch16_224\n"
    )

    f.write(
        f"Dataset: Rice_Leaf_AUG\n"
    )

    f.write(
        f"Total Images: {len(full_dataset)}\n"
    )

    f.write(
        f"Training Images: {train_size}\n"
    )

    f.write(
        f"Validation Images: {val_size}\n"
    )

    f.write(
        f"Epochs: {TOTAL_EPOCHS}\n"
    )

    f.write(
        f"Batch Size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Learning Rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Total Training Time: "
        f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s\n"
    )

    f.write(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}% ({accuracy:.4f})\n"
    )

    f.write(
        f"Weighted Precision: {precision:.4f}\n"
    )

    f.write(
        f"Weighted Recall: {recall:.4f}\n"
    )

    f.write(
        f"Weighted F1 Score: {f1:.4f}\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}% ({best_val_acc:.4f})\n"
    )


# ==========================================================
# 14. SAVE CLASS-WISE ACCURACY
# ==========================================================
classwise_path = os.path.join(
    OUTPUT_DIR,
    "vit_classwise_accuracy.txt"
)

with open(classwise_path, "w", encoding="utf-8") as f:

    f.write("ViT CLASS-WISE ACCURACY\n")
    f.write("=" * 50 + "\n\n")

    for i, name in enumerate(class_names):

        f.write(
            f"{name}: "
            f"{class_accuracy[i] * 100:.2f}% "
            f"({class_accuracy[i]:.4f})\n"
        )


# ==========================================================
# 15. SAVE CLASSIFICATION REPORT
# ==========================================================
report_path = os.path.join(
    OUTPUT_DIR,
    "vit_classification_report.txt"
)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("ViT CLASSIFICATION REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)


# ==========================================================
# 16. ACCURACY GRAPH
# ==========================================================
accuracy_plot_path = os.path.join(
    OUTPUT_DIR,
    "vit_accuracy.png"
)

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, TOTAL_EPOCHS + 1),
    history["train_acc"],
    label="Train Accuracy",
    marker="o"
)

plt.plot(
    range(1, TOTAL_EPOCHS + 1),
    history["val_acc"],
    label="Validation Accuracy",
    marker="o"
)

plt.title("ViT Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    accuracy_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================================
# 17. LOSS GRAPH
# ==========================================================
loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "vit_loss.png"
)

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, TOTAL_EPOCHS + 1),
    history["train_loss"],
    label="Train Loss",
    marker="o"
)

plt.plot(
    range(1, TOTAL_EPOCHS + 1),
    history["val_loss"],
    label="Validation Loss",
    marker="o"
)

plt.title("ViT Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================================
# 18. CONFUSION MATRIX
# ==========================================================
confusion_path = os.path.join(
    OUTPUT_DIR,
    "vit_confusion_matrix.png"
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("ViT Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================================
# 19. SAVE MODEL
# ==========================================================
model_path = os.path.join(
    OUTPUT_DIR,
    "vit_rice_leaf_final.pth"
)

torch.save(
    trained_model.state_dict(),
    model_path
)

print(f"\n[✓] Model saved to: {model_path}")


# ==========================================================
# 20. CREATE PDF REPORT
# ==========================================================
pdf_path = os.path.join(
    OUTPUT_DIR,
    "ViT Output.pdf"
)

with PdfPages(pdf_path) as pdf:

    # -------------------------
    # Page 1: Metrics
    # -------------------------
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")

    metric_text = (
        "ViT RICE LEAF DISEASE CLASSIFICATION\n"
        + "=" * 45
        + "\n\n"
        f"Model: vit_base_patch16_224\n"
        f"Dataset: Rice_Leaf_AUG\n"
        f"Total Images: {len(full_dataset)}\n"
        f"Training Images: {train_size}\n"
        f"Validation Images: {val_size}\n"
        f"Epochs: {TOTAL_EPOCHS}\n"
        f"Batch Size: {BATCH_SIZE}\n"
        f"Learning Rate: {LEARNING_RATE}\n\n"
        f"Training Time: "
        f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s\n\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n"
    )

    plt.text(
        0.08,
        0.92,
        metric_text,
        fontsize=13,
        verticalalignment="top",
        family="monospace"
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Page 2: Accuracy
    # -------------------------
    fig = plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, TOTAL_EPOCHS + 1),
        history["train_acc"],
        label="Train Accuracy",
        marker="o"
    )

    plt.plot(
        range(1, TOTAL_EPOCHS + 1),
        history["val_acc"],
        label="Validation Accuracy",
        marker="o"
    )

    plt.title("ViT Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------
    # Page 3: Loss
    # -------------------------
    fig = plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, TOTAL_EPOCHS + 1),
        history["train_loss"],
        label="Train Loss",
        marker="o"
    )

    plt.plot(
        range(1, TOTAL_EPOCHS + 1),
        history["val_loss"],
        label="Validation Loss",
        marker="o"
    )

    plt.title("ViT Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------
    # Page 4: Confusion Matrix
    # -------------------------
    fig = plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("ViT Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


# ==========================================================
# 21. FINAL OUTPUT SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("VIТ TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✓ {os.path.join(OUTPUT_DIR, filename)}")

print("\nFinal Metrics:")
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  Best Val : {best_val_acc * 100:.2f}%")

print("\nDone.")