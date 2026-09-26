# ================================================================
# RICE LEAF DISEASE CLASSIFICATION
# DeiT-Small | Canonical Dataset V1
#
# CANONICAL DATASET:
# /kaggle/input/datasets/varun2ks05/newdata
#
# Split:
#   Train      : 3063
#   Validation : 383
#   Test       : 383
#
# IMPORTANT:
# Test set is intentionally NOT used during this experiment.
# It will remain locked for final evaluation / ensemble evaluation.
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Install timm if necessary
# !pip install -q timm

import timm  

warnings.filterwarnings("ignore") 


# ================================================================
# 2. CONFIGURATION
# ================================================================

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 25

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

IMG_SIZE = 224
NUM_CLASSES = 6

MODEL_NAME = "deit_small_patch16_224"

NUM_WORKERS = 2

# Mixed precision for NVIDIA GPU
USE_AMP = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# 3. REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ================================================================
# 4. OUTPUT DIRECTORIES
# ================================================================

EXPERIMENT_DIR = Path(
    "/kaggle/working/DeiT_Canonical_V1"
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR = EXPERIMENT_DIR / "models"
PLOT_DIR = EXPERIMENT_DIR / "plots"
REPORT_DIR = EXPERIMENT_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ================================================================
# 5. CANONICAL DATASET PATH
# ================================================================

DATASET_DIR = Path(
    "/kaggle/input/datasets/varun2ks05/newdata"
)


# ================================================================
# 6. HEADER
# ================================================================

print()
print("=" * 80)
print("RICE LEAF DISEASE CLASSIFICATION")
print("DeiT-Small | Canonical V1 Dataset")
print("=" * 80)

print()
print("CONFIGURATION")
print("-" * 80)

print(f"Device          : {DEVICE}")
print(f"Model           : {MODEL_NAME}")
print(f"Image size      : {IMG_SIZE} x {IMG_SIZE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {NUM_EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Weight decay    : {WEIGHT_DECAY}")
print(f"Seed            : {SEED}")
print(f"Num classes     : {NUM_CLASSES}")
print(f"Workers         : {NUM_WORKERS}")
print(f"Mixed precision : {USE_AMP}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()
print(f"Dataset root    : {DATASET_DIR}")


# ================================================================
# 7. VERIFY DATASET
# ================================================================

print()
print("=" * 80)
print("VERIFYING CANONICAL DATASET")
print("=" * 80)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nCanonical dataset was not found at:\n{DATASET_DIR}"
    )

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

for required_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not required_dir.exists():

        raise FileNotFoundError(
            f"\nRequired directory not found:\n{required_dir}"
        )

print("Canonical dataset found successfully.")

print()
print("Dataset structure:")
print(f"  Train : {TRAIN_DIR}")
print(f"  Val   : {VAL_DIR}")
print(f"  Test  : {TEST_DIR}")


# ================================================================
# 8. LOAD CANONICAL CLASS MAPPING
# ================================================================

CLASS_MAPPING_FILE = DATASET_DIR / "class_mapping.json"

if CLASS_MAPPING_FILE.exists():

    with open(CLASS_MAPPING_FILE, "r") as f:
        mapping_data = json.load(f)

    print()
    print("Canonical class mapping loaded.")

else:

    print()
    print(
        "WARNING: class_mapping.json not found."
    )

    mapping_data = None


# ================================================================
# 9. FIXED CLASS ORDER
# ================================================================

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
]

CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: class_name
    for idx, class_name in enumerate(CLASS_NAMES)
}


print()
print("Class mapping:")
for idx, class_name in enumerate(CLASS_NAMES):
    print(f"  {idx} -> {class_name}")


# ================================================================
# 10. VERIFY DIRECTORY CLASSES
# ================================================================

print()
print("Checking class directories...")

for split_name, split_dir in [
    ("Train", TRAIN_DIR),
    ("Validation", VAL_DIR),
    ("Test", TEST_DIR)
]:

    actual_classes = sorted(
        [
            p.name
            for p in split_dir.iterdir()
            if p.is_dir()
        ]
    )

    expected_classes = sorted(CLASS_NAMES)

    if actual_classes != expected_classes:

        raise RuntimeError(
            f"\nClass mismatch in {split_name} directory.\n"
            f"Expected:\n{expected_classes}\n"
            f"Found:\n{actual_classes}"
        )

    print(
        f"  {split_name:<12}: "
        f"{len(actual_classes)} classes verified"
    )


# ================================================================
# 11. DATASET TRANSFORMS
# ================================================================

train_transforms = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transforms = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# 12. LOAD CANONICAL TRAIN / VAL DATASETS
# ================================================================

print()
print("=" * 80)
print("LOADING CANONICAL DATASET")
print("=" * 80)


train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transforms
)


val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=val_transforms
)


# ------------------------------------------------
# Verify ImageFolder class mapping
# ------------------------------------------------

print()
print("ImageFolder class mapping:")

print(train_dataset.class_to_idx)


expected_mapping = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

if train_dataset.class_to_idx != expected_mapping:

    raise RuntimeError(
        "\nImageFolder class mapping does not match "
        "the canonical mapping.\n"
        f"Expected: {expected_mapping}\n"
        f"Found: {train_dataset.class_to_idx}"
    )


if val_dataset.class_to_idx != expected_mapping:

    raise RuntimeError(
        "\nValidation ImageFolder class mapping mismatch."
    )


print()
print("Class mapping verified successfully.")


# ================================================================
# 13. DATASET COUNTS
# ================================================================

print()
print("Dataset sizes:")
print(f"  Training   : {len(train_dataset)}")
print(f"  Validation : {len(val_dataset)}")

# Test count is inspected only, NOT loaded for training/evaluation.
test_image_count = sum(
    1
    for class_dir in TEST_DIR.iterdir()
    if class_dir.is_dir()
    for file in class_dir.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

print(f"  Test       : {test_image_count}")
print()
print("TEST SET STATUS: LOCKED / NOT USED")


# ================================================================
# 14. CLASS DISTRIBUTION
# ================================================================

def count_images_per_class(split_dir):

    counts = {}

    for class_name in CLASS_NAMES:

        class_dir = split_dir / class_name

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        )

        counts[class_name] = count

    return counts


train_counts = count_images_per_class(TRAIN_DIR)
val_counts = count_images_per_class(VAL_DIR)
test_counts = count_images_per_class(TEST_DIR)


distribution_df = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Train": [train_counts[c] for c in CLASS_NAMES],
    "Validation": [val_counts[c] for c in CLASS_NAMES],
    "Test": [test_counts[c] for c in CLASS_NAMES]
})


print()
print("=" * 80)
print("CANONICAL DATASET DISTRIBUTION")
print("=" * 80)

print(
    distribution_df.to_string(
        index=False
    )
)


# ================================================================
# 15. DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ================================================================
# 16. MODEL INITIALIZATION
# ================================================================

print()
print("=" * 80)
print("INITIALIZING DeiT-SMALL")
print("=" * 80)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


print(f"Model initialized: {MODEL_NAME}")


# ================================================================
# 17. LOSS / OPTIMIZER / SCHEDULER
# ================================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6
)


# ================================================================
# 18. MIXED PRECISION
# ================================================================

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ================================================================
# 19. TRAINING VARIABLES
# ================================================================

best_val_acc = 0.0
best_epoch = 0

history = {

    "epoch": [],

    "train_loss": [],
    "val_loss": [],

    "train_acc": [],
    "val_acc": [],

    "learning_rate": []
}


BEST_MODEL_PATH = (
    MODEL_DIR /
    "best_deit_small_canonical_v1.pth"
)


# ================================================================
# 20. TRAINING
# ================================================================

print()
print("=" * 80)
print("STARTING DeiT TRAINING")
print("=" * 80)

print()
print(f"Model       : {MODEL_NAME}")
print(f"Epochs      : {NUM_EPOCHS}")
print(f"Train       : {len(train_dataset)} images")
print(f"Validation  : {len(val_dataset)} images")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Device      : {DEVICE}")

print()
print("-" * 80)


start_time = time.time()


for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()


    # ============================================================
    # TRAINING
    # ============================================================

    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_train = 0


    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )


            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()


        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        running_corrects += (
            predictions == targets
        ).sum().item()


        total_train += targets.size(0)


    scheduler.step()


    epoch_train_loss = (
        running_loss /
        total_train
    )

    epoch_train_acc = (
        running_corrects /
        total_train
    )


    # ============================================================
    # VALIDATION
    # ============================================================

    model.eval()

    val_loss = 0.0
    val_corrects = 0
    total_val = 0


    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )


            if USE_AMP:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        targets
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )


            val_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            val_corrects += (
                predictions == targets
            ).sum().item()


            total_val += targets.size(0)


    epoch_val_loss = (
        val_loss /
        total_val
    )

    epoch_val_acc = (
        val_corrects /
        total_val
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ============================================================
    # SAVE HISTORY
    # ============================================================

    history["epoch"].append(
        epoch + 1
    )

    history["train_loss"].append(
        epoch_train_loss
    )

    history["val_loss"].append(
        epoch_val_loss
    )

    history["train_acc"].append(
        epoch_train_acc
    )

    history["val_acc"].append(
        epoch_val_acc
    )

    history["learning_rate"].append(
        current_lr
    )


    # ============================================================
    # PRINT EPOCH
    # ============================================================

    epoch_time = (
        time.time() -
        epoch_start
    )


    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] "
        f"| Train Loss: {epoch_train_loss:.4f} "
        f"Acc: {epoch_train_acc*100:.2f}% "
        f"| Val Loss: {epoch_val_loss:.4f} "
        f"Acc: {epoch_val_acc*100:.2f}% "
        f"| LR: {current_lr:.2e} "
        f"| Time: {epoch_time:.1f}s"
    )


    # ============================================================
    # BEST MODEL
    # ============================================================

    if epoch_val_acc > best_val_acc:

        best_val_acc = epoch_val_acc

        best_epoch = epoch + 1


        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_accuracy":
                    best_val_acc,

                "class_names":
                    CLASS_NAMES,

                "model_name":
                    MODEL_NAME,

                "seed":
                    SEED
            },
            BEST_MODEL_PATH
        )


        print(
            f"  --> NEW BEST MODEL SAVED "
            f"({best_val_acc*100:.2f}%)"
        )


    print("-" * 80)


# ================================================================
# 21. TRAINING COMPLETE
# ================================================================

elapsed = time.time() - start_time

print()
print("=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)

print(
    f"Total training time : "
    f"{elapsed // 60:.0f}m {elapsed % 60:.0f}s"
)

print(
    f"Best validation accuracy : "
    f"{best_val_acc*100:.2f}%"
)

print(
    f"Best epoch : {best_epoch}"
)

print(
    f"Best model : {BEST_MODEL_PATH}"
)


# ================================================================
# 22. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(history)

HISTORY_PATH = (
    EXPERIMENT_DIR /
    "training_history.csv"
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ================================================================
# 23. LOAD BEST MODEL
# ================================================================

print()
print("=" * 80)
print("LOADING BEST MODEL FOR FINAL VALIDATION")
print("=" * 80)


checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    f"Loaded best checkpoint from epoch "
    f"{checkpoint['epoch']}"
)


# ================================================================
# 24. FINAL VALIDATION PREDICTIONS
# ================================================================

all_preds = []
all_targets = []


with torch.no_grad():

    for images, targets in val_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(images)

        else:

            outputs = model(images)


        predictions = outputs.argmax(
            dim=1
        )


        all_preds.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            targets.numpy()
        )


all_preds = np.array(
    all_preds
)

all_targets = np.array(
    all_targets
)


# ================================================================
# 25. FINAL METRICS
# ================================================================

accuracy = accuracy_score(
    all_targets,
    all_preds
)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0
    )
)


# ================================================================
# 26. FINAL METRICS OUTPUT
# ================================================================

print()
print("=" * 80)
print("FINAL VALIDATION METRICS")
print("=" * 80)

print()
print(
    f"Overall Accuracy : "
    f"{accuracy*100:.2f}%"
)

print(
    f"Macro Precision  : "
    f"{precision*100:.2f}%"
)

print(
    f"Macro Recall     : "
    f"{recall*100:.2f}%"
)

print(
    f"Macro F1-Score   : "
    f"{f1*100:.2f}%"
)


# ================================================================
# 27. CLASSIFICATION REPORT
# ================================================================

print()
print("-" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("-" * 80)


classification_report_text = classification_report(
    all_targets,
    all_preds,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print(
    classification_report_text
)


# ================================================================
# 28. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    all_targets,
    all_preds
)


print()
print("-" * 80)
print("CONFUSION MATRIX")
print("-" * 80)

print(cm)


# ================================================================
# 29. CLASS-WISE ACCURACY
# ================================================================

class_accuracies = (
    cm.diagonal() /
    cm.sum(axis=1)
)


print()
print("-" * 80)
print("CLASS-WISE ACCURACY")
print("-" * 80)


for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name.rjust(25)} : "
        f"{class_accuracies[i]*100:.2f}%"
    )


# ================================================================
# 30. SAVE METRICS JSON
# ================================================================

metrics = {

    "model": MODEL_NAME,

    "dataset": "Rice Leaf AUG Canonical V1",

    "dataset_path":
        str(DATASET_DIR),

    "seed": SEED,

    "epochs": NUM_EPOCHS,

    "batch_size": BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "image_size":
        IMG_SIZE,

    "best_epoch":
        best_epoch,

    "best_validation_accuracy":
        float(best_val_acc),

    "final_validation_accuracy":
        float(accuracy),

    "macro_precision":
        float(precision),

    "macro_recall":
        float(recall),

    "macro_f1":
        float(f1),

    "train_images":
        len(train_dataset),

    "validation_images":
        len(val_dataset),

    "test_images":
        test_image_count,

    "test_evaluated":
        False,

    "confusion_matrix":
        cm.tolist(),

    "class_wise_accuracy":
        {
            CLASS_NAMES[i]:
                float(class_accuracies[i])
            for i in range(NUM_CLASSES)
        }
}


METRICS_PATH = (
    EXPERIMENT_DIR /
    "final_metrics.json"
)


with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ================================================================
# 31. TRAINING CURVES
# ================================================================

print()
print("=" * 80)
print("GENERATING TRAINING CURVES")
print("=" * 80)


sns.set_theme(
    style="whitegrid",
    palette="muted"
)


# ------------------------------------------------
# Loss
# ------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Train Loss",
    linewidth=2
)


plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss",
    linewidth=2
)


plt.title(
    "DeiT-Small Training and Validation Loss",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Epoch",
    fontsize=12
)

plt.ylabel(
    "Cross-Entropy Loss",
    fontsize=12
)


plt.legend(
    frameon=True,
    shadow=True
)


plt.tight_layout()


LOSS_PLOT = (
    PLOT_DIR /
    "deit_training_validation_loss.png"
)


plt.savefig(
    LOSS_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ------------------------------------------------
# Accuracy
# ------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history["epoch"],
    np.array(history["train_acc"]) * 100,
    label="Train Accuracy",
    linewidth=2
)


plt.plot(
    history["epoch"],
    np.array(history["val_acc"]) * 100,
    label="Validation Accuracy",
    linewidth=2
)


plt.axhline(
    best_val_acc * 100,
    linestyle="--",
    linewidth=1.5,
    label=f"Best Val Accuracy: {best_val_acc*100:.2f}%"
)


plt.title(
    "DeiT-Small Training and Validation Accuracy",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Epoch",
    fontsize=12
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=12
)


plt.legend(
    frameon=True,
    shadow=True
)


plt.tight_layout()


ACC_PLOT = (
    PLOT_DIR /
    "deit_training_validation_accuracy.png"
)


plt.savefig(
    ACC_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ================================================================
# 32. CONFUSION MATRIX PLOT
# ================================================================

plt.figure(
    figsize=(10, 8)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 12},
    linewidths=0.5,
    linecolor="gray"
)


plt.title(
    "DeiT-Small Confusion Matrix",
    fontsize=16,
    fontweight="bold",
    pad=15
)


plt.ylabel(
    "True Class",
    fontsize=14,
    fontweight="bold"
)


plt.xlabel(
    "Predicted Class",
    fontsize=14,
    fontweight="bold"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.yticks(
    rotation=0
)


plt.tight_layout()


CM_PLOT = (
    PLOT_DIR /
    "deit_confusion_matrix.png"
)


plt.savefig(
    CM_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ================================================================
# 33. CLASS-WISE ACCURACY PLOT
# ================================================================

plt.figure(
    figsize=(11, 6)
)


plt.bar(
    CLASS_NAMES,
    class_accuracies * 100
)


plt.title(
    "DeiT-Small Class-wise Validation Accuracy",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Rice Leaf Disease Class",
    fontsize=12
)


plt.ylabel(
    "Accuracy (%)",
    fontsize=12
)


plt.xticks(
    rotation=35,
    ha="right"
)


plt.ylim(
    0,
    105
)


for i, value in enumerate(
    class_accuracies * 100
):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=10
    )


plt.tight_layout()


CLASS_ACC_PLOT = (
    PLOT_DIR /
    "deit_class_wise_accuracy.png"
)


plt.savefig(
    CLASS_ACC_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ================================================================
# 34. SAVE CLASSIFICATION REPORT
# ================================================================

REPORT_TEXT_PATH = (
    REPORT_DIR /
    "classification_report.txt"
)


with open(
    REPORT_TEXT_PATH,
    "w"
) as f:

    f.write(
        "DeiT-Small | Rice Leaf Disease Classification\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        classification_report_text
    )


# ================================================================
# 35. GENERATE PDF REPORT
# ================================================================

print()
print("=" * 80)
print("GENERATING PDF REPORT")
print("=" * 80)


# Install reportlab if necessary
!pip install -q reportlab


from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
    PageBreak
)


PDF_PATH = (
    EXPERIMENT_DIR /
    "DeiT_Canonical_V1_Experiment_Report.pdf"
)


doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)


styles = getSampleStyleSheet()


title_style = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=20,
    leading=24,
    spaceAfter=15
)


subtitle_style = ParagraphStyle(
    "SubtitleCustom",
    parent=styles["Normal"],
    alignment=TA_CENTER,
    fontSize=11,
    leading=15,
    spaceAfter=20
)


heading_style = ParagraphStyle(
    "HeadingCustom",
    parent=styles["Heading2"],
    fontSize=14,
    leading=18,
    spaceBefore=12,
    spaceAfter=8
)


body_style = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13
)


story = []


# ------------------------------------------------
# Title
# ------------------------------------------------

story.append(
    Paragraph(
        "Rice Leaf Disease Classification",
        title_style
    )
)


story.append(
    Paragraph(
        "DeiT-Small | Canonical Dataset V1",
        subtitle_style
    )
)


story.append(
    Paragraph(
        "Final validation evaluation after training on the "
        "canonical stratified 80/10/10 dataset.",
        body_style
    )
)


story.append(
    Spacer(
        1,
        15
    )
)


# ------------------------------------------------
# Experiment configuration
# ------------------------------------------------

story.append(
    Paragraph(
        "1. Experiment Configuration",
        heading_style
    )
)


config_data = [

    ["Parameter", "Value"],

    ["Model", MODEL_NAME],

    ["Image Size", f"{IMG_SIZE} × {IMG_SIZE}"],

    ["Batch Size", str(BATCH_SIZE)],

    ["Epochs", str(NUM_EPOCHS)],

    ["Learning Rate", str(LEARNING_RATE)],

    ["Weight Decay", str(WEIGHT_DECAY)],

    ["Optimizer", "AdamW"],

    ["Scheduler", "CosineAnnealingLR"],

    ["Random Seed", str(SEED)],

    ["Device", str(DEVICE)],

    ["GPU",
     torch.cuda.get_device_name(0)
     if torch.cuda.is_available()
     else "CPU"],

    ["Mixed Precision", str(USE_AMP)]
]


config_table = Table(
    config_data,
    colWidths=[2.1 * inch, 4.2 * inch]
)


config_table.setStyle(
    TableStyle([

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            5
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            5
        )
    ])
)


story.append(config_table)


# ------------------------------------------------
# Dataset
# ------------------------------------------------

story.append(
    Paragraph(
        "2. Canonical Dataset",
        heading_style
    )
)


dataset_data = [

    ["Split", "Images", "Usage"],

    ["Train", str(len(train_dataset)),
     "Model training"],

    ["Validation", str(len(val_dataset)),
     "Checkpoint selection / evaluation"],

    ["Test", str(test_image_count),
     "LOCKED. Not used in this experiment."]
]


dataset_table = Table(
    dataset_data,
    colWidths=[
        1.4 * inch,
        1.2 * inch,
        3.7 * inch
    ]
)


dataset_table.setStyle(
    TableStyle([

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        )
    ])
)


story.append(dataset_table)


story.append(
    Spacer(
        1,
        10
    )
)


story.append(
    Paragraph(
        "Canonical dataset path: "
        + str(DATASET_DIR),
        body_style
    )
)


# ------------------------------------------------
# Final metrics
# ------------------------------------------------

story.append(
    Paragraph(
        "3. Final Validation Metrics",
        heading_style
    )
)


metric_data = [

    ["Metric", "Result"],

    ["Best Validation Accuracy",
     f"{best_val_acc*100:.2f}%"],

    ["Best Epoch",
     str(best_epoch)],

    ["Final Validation Accuracy",
     f"{accuracy*100:.2f}%"],

    ["Macro Precision",
     f"{precision*100:.2f}%"],

    ["Macro Recall",
     f"{recall*100:.2f}%"],

    ["Macro F1-Score",
     f"{f1*100:.2f}%"]
]


metric_table = Table(
    metric_data,
    colWidths=[
        3.5 * inch,
        2.8 * inch
    ]
)


metric_table.setStyle(
    TableStyle([

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            9
        ),

        (
            "ALIGN",
            (1, 1),
            (1, -1),
            "CENTER"
        )
    ])
)


story.append(metric_table)


# ------------------------------------------------
# Classification report
# ------------------------------------------------

story.append(
    Paragraph(
        "4. Classification Report",
        heading_style
    )
)


report_lines = classification_report_text.splitlines()


report_table_data = []


for line in report_lines:

    parts = line.split()

    if len(parts) >= 5:

        if parts[0] in CLASS_NAMES:

            report_table_data.append([
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4]
            ])

        elif parts[0] in ["accuracy", "macro", "weighted"]:

            report_table_data.append(
                [line, "", "", "", ""]
            )


# Simpler monospaced report block
for line in report_lines:

    safe_line = (
        line
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            f"<font name='Courier'>{safe_line}</font>",
            ParagraphStyle(
                "ReportLine",
                parent=body_style,
                fontSize=7.5,
                leading=9
            )
        )
    )


# ------------------------------------------------
# Confusion matrix
# ------------------------------------------------

story.append(
    PageBreak()
)


story.append(
    Paragraph(
        "5. Confusion Matrix",
        heading_style
    )
)


cm_table_data = [
    ["True \\ Pred"] + [
        str(i)
        for i in range(NUM_CLASSES)
    ]
]


for i in range(NUM_CLASSES):

    cm_table_data.append(
        [str(i)] +
        [str(x) for x in cm[i]]
    )


cm_table = Table(
    cm_table_data,
    colWidths=[0.8 * inch] * 7
)


cm_table.setStyle(
    TableStyle([

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),

        (
            "BACKGROUND",
            (0, 0),
            (0, -1),
            colors.lightgrey
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),

        (
            "FONTNAME",
            (0, 0),
            (0, -1),
            "Helvetica-Bold"
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "ALIGN",
            (0, 0),
            (-1, -1),
            "CENTER"
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8
        )
    ])
)


story.append(cm_table)


story.append(
    Spacer(
        1,
        15
    )
)


story.append(
    Paragraph(
        "Class index mapping:",
        body_style
    )
)


for i, class_name in enumerate(CLASS_NAMES):

    story.append(
        Paragraph(
            f"{i} = {class_name}",
            body_style
        )
    )


# ------------------------------------------------
# Plots
# ------------------------------------------------

story.append(
    Spacer(
        1,
        15
    )
)


story.append(
    Paragraph(
        "6. Training Curves",
        heading_style
    )
)


story.append(
    ReportLabImage(
        str(LOSS_PLOT),
        width=6.5 * inch,
        height=3.9 * inch
    )
)


story.append(
    Spacer(
        1,
        10
    )
)


story.append(
    ReportLabImage(
        str(ACC_PLOT),
        width=6.5 * inch,
        height=3.9 * inch
    )
)


story.append(
    PageBreak()
)


story.append(
    Paragraph(
        "7. Confusion Matrix Visualization",
        heading_style
    )
)


story.append(
    ReportLabImage(
        str(CM_PLOT),
        width=6.5 * inch,
        height=5.2 * inch
    )
)


story.append(
    Spacer(
        1,
        15
    )
)


story.append(
    Paragraph(
        "8. Class-wise Accuracy",
        heading_style
    )
)


story.append(
    ReportLabImage(
        str(CLASS_ACC_PLOT),
        width=6.5 * inch,
        height=3.8 * inch
    )
)


# ------------------------------------------------
# Epoch history
# ------------------------------------------------

story.append(
    PageBreak()
)


story.append(
    Paragraph(
        "9. Complete Epoch History",
        heading_style
    )
)


epoch_table_data = [
    [
        "Epoch",
        "Train Loss",
        "Train Acc",
        "Val Loss",
        "Val Acc",
        "LR"
    ]
]


for i in range(len(history["epoch"])):

    epoch_table_data.append([

        str(history["epoch"][i]),

        f"{history['train_loss'][i]:.4f}",

        f"{history['train_acc'][i]*100:.2f}%",

        f"{history['val_loss'][i]:.4f}",

        f"{history['val_acc'][i]*100:.2f}%",

        f"{history['learning_rate'][i]:.2e}"
    ])


epoch_table = Table(
    epoch_table_data,
    repeatRows=1,
    colWidths=[
        0.55 * inch,
        1.1 * inch,
        1.1 * inch,
        1.1 * inch,
        1.1 * inch,
        1.1 * inch
    ]
)


epoch_table.setStyle(
    TableStyle([

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.3,
            colors.grey
        ),

        (
            "ALIGN",
            (0, 0),
            (-1, -1),
            "CENTER"
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            7
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            4
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            4
        )
    ])
)


story.append(
    epoch_table
)


# ------------------------------------------------
# Final notes
# ------------------------------------------------

story.append(
    Spacer(
        1,
        20
    )
)


story.append(
    Paragraph(
        "10. Experiment Notes",
        heading_style
    )
)


notes = [

    "The dataset uses the canonical stratified 80/10/10 split.",

    "Random seed: 42.",

    "The training and validation sets were used for model development.",

    "The test set was intentionally kept completely untouched.",

    "The best model was selected using validation accuracy.",

    "The saved checkpoint corresponds to the epoch with the highest validation accuracy.",

    "The test set should be evaluated only during the final locked evaluation stage, "
    "after all individual models and ensemble decisions have been finalized."
]


for note in notes:

    story.append(
        Paragraph(
            "• " + note,
            body_style
        )
    )


# ------------------------------------------------
# Build PDF
# ------------------------------------------------

doc.build(
    story
)


print()
print("PDF REPORT CREATED:")
print(PDF_PATH)


# ================================================================
# 36. FINAL OUTPUT SUMMARY
# ================================================================

print()
print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print()
print("BEST MODEL")
print("-" * 80)
print(BEST_MODEL_PATH)

print()
print("TRAINING HISTORY")
print("-" * 80)
print(HISTORY_PATH)

print()
print("METRICS")
print("-" * 80)
print(METRICS_PATH)

print()
print("PLOTS")
print("-" * 80)
print(LOSS_PLOT)
print(ACC_PLOT)
print(CM_PLOT)
print(CLASS_ACC_PLOT)

print()
print("CLASSIFICATION REPORT")
print("-" * 80)
print(REPORT_TEXT_PATH)

print()
print("PDF REPORT")
print("-" * 80)
print(PDF_PATH)

print()
print("=" * 80)
print("FINAL VALIDATION RESULT")
print("=" * 80)

print(
    f"Accuracy       : {accuracy*100:.2f}%"
)

print(
    f"Precision      : {precision*100:.2f}%"
)

print(
    f"Recall         : {recall*100:.2f}%"
)

print(
    f"F1 Score       : {f1*100:.2f}%"
)

print(
    f"Best Epoch     : {best_epoch}"
)

print(
    f"Best Val Acc   : {best_val_acc*100:.2f}%"
)

print()
print("TEST SET: NOT EVALUATED / LOCKED")

print()
print("=" * 80)
print("DONE")
print("=" * 80)