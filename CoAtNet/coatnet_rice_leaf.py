# =============================================================
# CropResQ - CoAtNet Model (Train, Val, Test + Canonical PDF Report)
# =============================================================

# -------------------------------------------------------------
# 1. Environment & Package Installs
# -------------------------------------------------------------
# Note: We use `timm` (PyTorch Image Models) for CoAtNet
#!pip install -q timm reportlab seaborn matplotlib scikit-learn

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -------------------------------------------------------------
# 2. Configuration & Hyperparameters
# -------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
IMG_SIZE = 224
NUM_CLASSES = 6
# Standard CoAtNet-0 model available in timm
MODEL_NAME = "coatnet_0_rw_224.sw_in1k" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "/kaggle/working/coatnet_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Active Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# -------------------------------------------------------------
# 3. Locate Dataset Path (train, val, test)
# -------------------------------------------------------------
DATASET_DIR = None
if os.path.exists("/kaggle/input/newdata") and os.path.exists("/kaggle/input/newdata/train"):
    DATASET_DIR = "/kaggle/input/newdata"
else:
    matches = glob.glob("/kaggle/input/**/newdata", recursive=True)
    for p in matches:
        if os.path.exists(os.path.join(p, "train")):
            DATASET_DIR = p
            break

if DATASET_DIR is None:
    matches = glob.glob("/kaggle/input/**/train", recursive=True)
    if matches:
        DATASET_DIR = os.path.dirname(matches[0])
    else:
        raise FileNotFoundError("Could not find the dataset with 'train', 'val', and 'test' subdirectories.")

print(f"Dataset root identified at: {DATASET_DIR}")

# -------------------------------------------------------------
# 4. Data Transforms & Loaders (CoAtNet Standard Normalization)
# -------------------------------------------------------------
# CoAtNet uses standard ImageNet mean and std
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val") if os.path.exists(os.path.join(DATASET_DIR, "val")) else os.path.join(DATASET_DIR, "validation")
test_dir = os.path.join(DATASET_DIR, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=eval_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)
print(f"\nDetected {NUM_CLASSES} classes: {class_names}")
print(f"Dataset split sizes -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# -------------------------------------------------------------
# 5. CoAtNet Model Architecture (via timm)
# -------------------------------------------------------------
print(f"\nLoading CoAtNet model ({MODEL_NAME})...")
# timm automatically replaces the final classification layer when num_classes is provided!
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
print("CoAtNet model initialized successfully.")

# -------------------------------------------------------------
# 6. Loss, Optimizer & Scheduler
# -------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# -------------------------------------------------------------
# 7. Training & Validation Loop
# -------------------------------------------------------------
best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "best_coatnet_rice_model.pth")

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

print("\n--- Starting CoAtNet Training ---")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # TRAIN
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, targets in train_loader:
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == targets).sum().item()
        total_train += targets.size(0)

    scheduler.step()
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100

    # VALIDATE
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == targets).sum().item()
            total_val += targets.size(0)

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = (correct_val / total_val) * 100

    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_acc"].append(epoch_val_acc)

    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] "
        f"| Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% "
        f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%"
    )

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc:.2f}%)")

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed//60:.0f}m {elapsed%60:.0f}s. Peak Val Accuracy: {best_val_acc:.2f}%")

# -------------------------------------------------------------
# 8. Unbiased Evaluation on Test Set
# -------------------------------------------------------------
print("\n" + "=" * 50)
print("         EVALUATING BEST MODEL ON TEST SET")
print("=" * 50)

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

overall_acc = accuracy_score(all_targets, all_preds) * 100
macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
    all_targets, all_preds, average="macro", zero_division=0
)

print(f"\nFinal Test Accuracy: {overall_acc:.2f}%")
print(f"Macro Precision:     {macro_prec*100:.2f}%")
print(f"Macro Recall:        {macro_rec*100:.2f}%")
print(f"Macro F1-Score:      {macro_f1*100:.2f}%")

cm = confusion_matrix(all_targets, all_preds)
class_accuracies = (cm.diagonal() / cm.sum(axis=1)) * 100
rep_dict = classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0, output_dict=True)

# -------------------------------------------------------------
# 9. Plotting & Saving Figures
# -------------------------------------------------------------
loss_path = os.path.join(OUTPUT_DIR, "canonical_coatnet_loss.png")
plt.figure(figsize=(8, 4.2))
plt.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label="Train Loss", color="#3b6998", linewidth=1.8)
plt.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label="Validation Loss", color="#e07b42", linewidth=1.8)
plt.title("CoAtNet Training and Validation Loss", fontsize=12, fontweight="bold")
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Cross-Entropy Loss", fontsize=10)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(loss_path, dpi=300)
plt.close()

acc_path = os.path.join(OUTPUT_DIR, "canonical_coatnet_acc.png")
plt.figure(figsize=(8, 4.2))
plt.plot(range(1, NUM_EPOCHS + 1), history["train_acc"], label="Train Accuracy", color="#3b6998", linewidth=1.8)
plt.plot(range(1, NUM_EPOCHS + 1), history["val_acc"], label="Validation Accuracy", color="#e07b42", linewidth=1.8)
plt.axhline(y=best_val_acc, color="gray", linestyle="--", alpha=0.7, label=f"Best Val Accuracy: {best_val_acc:.2f}%")
plt.title("CoAtNet Training and Validation Accuracy", fontsize=12, fontweight="bold")
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=10)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(acc_path, dpi=300)
plt.close()

cm_path = os.path.join(OUTPUT_DIR, "canonical_coatnet_cm.png")
plt.figure(figsize=(7.5, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title("CoAtNet Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Predicted Class", fontsize=10, fontweight="bold")
plt.ylabel("True Class", fontsize=10, fontweight="bold")
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(cm_path, dpi=300)
plt.close()

bar_path = os.path.join(OUTPUT_DIR, "canonical_coatnet_bar.png")
plt.figure(figsize=(8, 4.2))
bars = plt.bar(class_names, class_accuracies, color="#3b75af", width=0.65)
plt.title("CoAtNet Class-wise Test Accuracy", fontsize=12, fontweight="bold")
plt.xlabel("Rice Leaf Disease Class", fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=10)
plt.ylim(0, 108)
plt.xticks(rotation=30, ha="right", fontsize=8)
for b in bars:
    y = b.get_height()
    plt.text(b.get_x() + b.get_width()/2.0, y + 1.5, f"{y:.2f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()

# -------------------------------------------------------------
# 10. Multi-Section Structured PDF Report Generation
# -------------------------------------------------------------
REPORT_PDF_PATH = os.path.join(OUTPUT_DIR, "CoAtNet_Canonical_V1_Experiment_Report.pdf")
styles = getSampleStyleSheet()

title_style = ParagraphStyle('DocTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=18, leading=22, alignment=0)
sub_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor("#2C3E50"))
desc_style = ParagraphStyle('DocDesc', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9, leading=12, textColor=colors.dimgrey)
h2_style = ParagraphStyle('Heading2Custom', fontName='Helvetica-Bold', fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)
body_style = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=8.5, leading=11)

def get_standard_table_style():
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
    ])

story = []

# Document Header
story.append(Paragraph("Rice Leaf Disease Classification", title_style))
story.append(Paragraph("CoAtNet-0 | Canonical Dataset V1", sub_style))
story.append(Paragraph("Final evaluation after training on the canonical stratified 80/10/10 dataset.", desc_style))
story.append(Spacer(1, 10))

# 1. Experiment Configuration
story.append(Paragraph("1. Experiment Configuration", h2_style))
config_data = [
    ["Parameter", "Value"],
    ["Model Architecture", "CoAtNet-0"],
    ["Model Tag (timm)", MODEL_NAME],
    ["Image Size", f"{IMG_SIZE} x {IMG_SIZE}"],
    ["Batch Size", str(BATCH_SIZE)],
    ["Epochs", str(NUM_EPOCHS)],
    ["Learning Rate", str(LEARNING_RATE)],
    ["Weight Decay", str(WEIGHT_DECAY)],
    ["Optimizer", "AdamW"],
    ["Scheduler", "CosineAnnealingLR"],
    ["Random Seed", str(SEED)],
    ["Device", str(DEVICE)],
    ["GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"]
]
t1 = Table(config_data, colWidths=[200, 320])
t1.setStyle(get_standard_table_style())
story.append(t1)
story.append(Spacer(1, 10))

# 2. Canonical Dataset
story.append(Paragraph("2. Canonical Dataset", h2_style))
dataset_summary = [
    ["Split", "Images", "Usage"],
    ["Train", str(len(train_dataset)), "Model training"],
    ["Validation", str(len(val_dataset)), "Checkpoint selection / evaluation"],
    ["Test", str(len(test_dataset)), "Final holdout benchmark"]
]
t2 = Table(dataset_summary, colWidths=[120, 100, 300])
t2.setStyle(get_standard_table_style())
story.append(t2)
story.append(Spacer(1, 4))
story.append(Paragraph(f"<b>Canonical dataset path:</b> {DATASET_DIR}", body_style))
story.append(Spacer(1, 10))

# 3. Final Metrics
best_epoch_idx = int(np.argmax(history["val_acc"])) + 1
story.append(Paragraph("3. Final Validation & Test Metrics", h2_style))
metrics_summary = [
    ["Metric", "Result"],
    ["Best Validation Accuracy", f"{best_val_acc:.2f}%"],
    ["Best Epoch", str(best_epoch_idx)],
    ["Final Test Accuracy", f"{overall_acc:.2f}%"],
    ["Macro Precision", f"{macro_prec*100:.2f}%"],
    ["Macro Recall", f"{macro_rec*100:.2f}%"],
    ["Macro F1-Score", f"{macro_f1*100:.2f}%"]
]
t3 = Table(metrics_summary, colWidths=[220, 300])
t3.setStyle(get_standard_table_style())
story.append(t3)
story.append(Spacer(1, 10))

# 4. Classification Report
story.append(Paragraph("4. Classification Report", h2_style))
clf_table_data = [["Class", "precision", "recall", "f1-score", "support"]]
for c in class_names:
    clf_table_data.append([
        c,
        f"{rep_dict[c]['precision']:.4f}",
        f"{rep_dict[c]['recall']:.4f}",
        f"{rep_dict[c]['f1-score']:.4f}",
        str(int(rep_dict[c]['support']))
    ])
clf_table_data.append([
    "accuracy", "", "", f"{rep_dict['accuracy']:.4f}", str(len(all_targets))
])
clf_table_data.append([
    "macro avg",
    f"{rep_dict['macro avg']['precision']:.4f}",
    f"{rep_dict['macro avg']['recall']:.4f}",
    f"{rep_dict['macro avg']['f1-score']:.4f}",
    str(len(all_targets))
])
clf_table_data.append([
    "weighted avg",
    f"{rep_dict['weighted avg']['precision']:.4f}",
    f"{rep_dict['weighted avg']['recall']:.4f}",
    f"{rep_dict['weighted avg']['f1-score']:.4f}",
    str(len(all_targets))
])
t4 = Table(clf_table_data, colWidths=[180, 85, 85, 85, 85])
t4.setStyle(get_standard_table_style())
story.append(t4)
story.append(Spacer(1, 10))

# 5. Confusion Matrix (Numeric Table)
story.append(Paragraph("5. Confusion Matrix", h2_style))
cm_table_data = [["True \\ Pred"] + [str(i) for i in range(len(class_names))]]
for row_idx, row in enumerate(cm):
    cm_table_data.append([str(row_idx)] + [str(val) for val in row])
t5 = Table(cm_table_data, colWidths=[90] + [70] * len(class_names))
t5.setStyle(get_standard_table_style())
story.append(t5)
story.append(Spacer(1, 4))

mapping_lines = "<br/>".join([f"<b>{i}</b> = {cls}" for i, cls in enumerate(class_names)])
story.append(Paragraph(f"<b>Class index mapping:</b><br/>{mapping_lines}", body_style))
story.append(Spacer(1, 12))

# 6. Training Curves
story.append(PageBreak())
story.append(Paragraph("6. Training Curves", h2_style))
story.append(RLImage(loss_path, width=480, height=220))
story.append(Spacer(1, 8))
story.append(RLImage(acc_path, width=480, height=220))
story.append(Spacer(1, 12))

# 7 & 8: Visual Confusion Matrix & Class-wise Acc
story.append(Paragraph("7. Confusion Matrix Visualization", h2_style))
story.append(RLImage(cm_path, width=400, height=310))
story.append(Spacer(1, 10))

story.append(Paragraph("8. Class-wise Accuracy", h2_style))
story.append(RLImage(bar_path, width=480, height=210))
story.append(Spacer(1, 12))

# 9. Complete Epoch History Table
story.append(PageBreak())
story.append(Paragraph("9. Complete Epoch History", h2_style))
hist_table_data = [["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR"]]

lr_schedule = [
    1e-6 + 0.5 * (LEARNING_RATE - 1e-6) * (1 + np.cos(np.pi * ep / NUM_EPOCHS))
    for ep in range(NUM_EPOCHS)
]

for ep in range(NUM_EPOCHS):
    hist_table_data.append([
        str(ep + 1),
        f"{history['train_loss'][ep]:.4f}",
        f"{history['train_acc'][ep]:.2f}%",
        f"{history['val_loss'][ep]:.4f}",
        f"{history['val_acc'][ep]:.2f}%",
        f"{lr_schedule[ep]:.2e}"
    ])

t9 = Table(hist_table_data, colWidths=[60, 90, 90, 90, 90, 100])
t9.setStyle(get_standard_table_style())
story.append(t9)
story.append(Spacer(1, 14))

# 10. Experiment Notes
story.append(Paragraph("10. Experiment Notes", h2_style))
notes = [
    "• The dataset uses the canonical stratified 80/10/10 split.",
    "• Random seed: 42 ensures reproducibility across runs.",
    "• Model uses a CoAtNet-0 (coatnet_0_rw_224) architecture, bridging the gap between Convolutional Neural Networks and Transformers.",
    "• Model initialized and configured using the timm (PyTorch Image Models) ecosystem."
]

for note in notes:
    story.append(Paragraph(note, body_style))
    story.append(Spacer(1, 4))

# Build PDF
doc = SimpleDocTemplate(REPORT_PDF_PATH, pagesize=letter)
doc.build(story)
print(f"\nPDF Report successfully generated at: {REPORT_PDF_PATH}")