# ==========================================================
# CoAtNet - Rice Leaf Disease Classification (Kaggle)
# Improved version with:
# - Reproducibility
# - Proper ImageNet normalization
# - Data augmentation
# - Stratified 80/20 split
# - Class distribution check
# - Mixed precision
# - Cosine learning-rate scheduler
# - Label smoothing
# - Best model restoration
# - Detailed metrics
# - Accuracy/Loss graphs
# - Confusion matrix
# - PDF report
# ==========================================================


# ==========================================================
# 1. IMPORTS
# ==========================================================

import os
import time
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import timm

import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages


# ==========================================================
# 2. REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# 3. CONFIGURATION
# ==========================================================

# IMPORTANT:
# Use the SAME dataset path as your ViT experiment.

DATASET_PATH = "/kaggle/input/datasets/varun2ks05/rice-leaf-aug/Rice_Leaf_AUG"

IMAGE_SIZE = 224

# Safer for Tesla T4
BATCH_SIZE = 16

TOTAL_EPOCHS = 25

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

LABEL_SMOOTHING = 0.1

OUTPUT_DIR = "CoAtNet"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 4. DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CoAtNet RICE LEAF DISEASE CLASSIFICATION")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================================
# 5. DATA TRANSFORMS
# ==========================================================

# Training augmentation
train_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    # ImageNet normalization for pretrained CoAtNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation transform
val_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# 6. LOAD DATASET
# ==========================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found at:\n"
        f"{DATASET_PATH}\n\n"
        "Make sure the Rice_Leaf_AUG dataset is attached "
        "to your Kaggle notebook."
    )


print(f"\nDataset path:")
print(DATASET_PATH)


# Temporary dataset only to obtain labels/classes
base_dataset = datasets.ImageFolder(
    DATASET_PATH
)

class_names = base_dataset.classes

num_classes = len(class_names)

targets = np.array(
    base_dataset.targets
)


print("\n" + "-" * 60)
print("DATASET INFORMATION")
print("-" * 60)

print(f"Total images: {len(base_dataset)}")

print(f"Number of classes: {num_classes}")

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        f"  {i}: {name}"
    )


# ==========================================================
# 7. CLASS DISTRIBUTION CHECK
# ==========================================================

print("\nClass distribution:")

unique_classes, class_counts = np.unique(
    targets,
    return_counts=True
)

for class_id, count in zip(
    unique_classes,
    class_counts
):

    print(
        f"  {class_names[class_id]:<25} : {count}"
    )


# ==========================================================
# 8. STRATIFIED 80/20 TRAIN-VALIDATION SPLIT
# ==========================================================

indices = np.arange(
    len(base_dataset)
)

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=targets
)


print("\n" + "-" * 60)
print("DATA SPLIT")
print("-" * 60)

print(
    f"Training images:   {len(train_indices)}"
)

print(
    f"Validation images: {len(val_indices)}"
)


# ==========================================================
# 9. CREATE TRAINING AND VALIDATION DATASETS
# ==========================================================

# Separate ImageFolder datasets are used so that
# augmentation is applied ONLY to training images.

train_full_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transforms
)

val_full_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=val_transforms
)


train_dataset = Subset(
    train_full_dataset,
    train_indices
)

val_dataset = Subset(
    val_full_dataset,
    val_indices
)


# ==========================================================
# 10. DATA LOADERS
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


print("\nDataLoaders created successfully.")


# ==========================================================
# 11. CREATE PRETRAINED CoAtNet
# ==========================================================

print("\n" + "-" * 60)
print("LOADING PRETRAINED CoAtNet")
print("-" * 60)


coatnet_model = timm.create_model(
    "coatnet_0_rw_224.sw_in1k",
    pretrained=True,
    num_classes=num_classes
)


coatnet_model = coatnet_model.to(device)


print(
    "CoAtNet model loaded successfully."
)

print(
    f"Number of classes: {num_classes}"
)


# ==========================================================
# 12. TRAINING FUNCTION
# ==========================================================

def train_coatnet(
    model,
    train_loader,
    val_loader
):

    # ------------------------------------------------------
    # Loss function
    # ------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )


    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ------------------------------------------------------
    # Cosine learning-rate scheduler
    # ------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS
    )


    # ------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------

    use_amp = device.type == "cuda"


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # ------------------------------------------------------
    # History
    # ------------------------------------------------------

    history = {

        "train_loss": [],
        "train_acc": [],

        "val_loss": [],
        "val_acc": [],

        "learning_rate": []
    }


    # ------------------------------------------------------
    # Best model tracking
    # ------------------------------------------------------

    best_val_acc = 0.0

    best_state = None


    # ------------------------------------------------------
    # Training timer
    # ------------------------------------------------------

    start_time = time.time()


    print(
        f"\nStarting CoAtNet training "
        f"for {TOTAL_EPOCHS} epochs...\n"
    )


    # ======================================================
    # EPOCH LOOP
    # ======================================================

    for epoch in range(TOTAL_EPOCHS):


        # ==================================================
        # TRAINING
        # ==================================================

        model.train()

        train_loss = 0.0

        train_correct = 0

        train_total = 0


        train_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1:02d}/"
                f"{TOTAL_EPOCHS} [Train]"
            )
        )


        for images, labels in train_bar:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            # ----------------------------------------------
            # Forward pass
            # ----------------------------------------------

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            # ----------------------------------------------
            # Backpropagation
            # ----------------------------------------------

            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            # ----------------------------------------------
            # Statistics
            # ----------------------------------------------

            train_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            train_correct += (
                predictions == labels
            ).sum().item()


            train_total += (
                labels.size(0)
            )


            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        # ==================================================
        # TRAINING METRICS
        # ==================================================

        epoch_train_loss = (
            train_loss / train_total
        )

        epoch_train_acc = (
            train_correct / train_total
        )


        # ==================================================
        # VALIDATION
        # ==================================================

        model.eval()

        val_loss = 0.0

        val_correct = 0

        val_total = 0


        epoch_preds = []

        epoch_labels = []


        with torch.no_grad():

            val_bar = tqdm(
                val_loader,
                desc=(
                    f"Epoch {epoch + 1:02d}/"
                    f"{TOTAL_EPOCHS} [Val]  "
                )
            )


            for images, labels in val_bar:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )


                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )


                val_loss += (
                    loss.item()
                    * images.size(0)
                )


                predictions = outputs.argmax(
                    dim=1
                )


                val_correct += (
                    predictions == labels
                ).sum().item()


                val_total += (
                    labels.size(0)
                )


                epoch_preds.extend(
                    predictions.cpu().numpy()
                )


                epoch_labels.extend(
                    labels.cpu().numpy()
                )


        # ==================================================
        # VALIDATION METRICS
        # ==================================================

        epoch_val_loss = (
            val_loss / val_total
        )

        epoch_val_acc = (
            val_correct / val_total
        )


        # ==================================================
        # SAVE HISTORY
        # ==================================================

        history["train_loss"].append(
            epoch_train_loss
        )

        history["train_acc"].append(
            epoch_train_acc
        )

        history["val_loss"].append(
            epoch_val_loss
        )

        history["val_acc"].append(
            epoch_val_acc
        )

        history["learning_rate"].append(
            optimizer.param_groups[0]["lr"]
        )


        # ==================================================
        # PRINT SUMMARY
        # ==================================================

        print(
            f"\nEpoch {epoch + 1:02d}/"
            f"{TOTAL_EPOCHS} Summary -> "

            f"Train Loss: "
            f"{epoch_train_loss:.4f} | "

            f"Train Acc: "
            f"{epoch_train_acc:.4f} | "

            f"Val Loss: "
            f"{epoch_val_loss:.4f} | "

            f"Val Acc: "
            f"{epoch_val_acc:.4f} | "

            f"LR: "
            f"{optimizer.param_groups[0]['lr']:.2e}\n"
        )


        # ==================================================
        # SAVE BEST MODEL
        # ==================================================

        if epoch_val_acc > best_val_acc:

            best_val_acc = epoch_val_acc

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            print(
                f"[✓] New best validation "
                f"accuracy: "
                f"{best_val_acc * 100:.2f}%"
            )


        # ==================================================
        # UPDATE LEARNING RATE
        # ==================================================

        scheduler.step()


    # ======================================================
    # TOTAL TRAINING TIME
    # ======================================================

    total_training_time = (
        time.time() - start_time
    )


    # ======================================================
    # RESTORE BEST MODEL
    # ======================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    # ======================================================
    # FINAL VALIDATION
    # ======================================================

    model.eval()

    final_preds = []

    final_labels = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                outputs = model(images)


            predictions = outputs.argmax(
                dim=1
            )


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
# 13. TRAIN CoAtNet
# ==========================================================

(
    trained_model,
    history,
    y_true,
    y_pred,
    total_training_time,
    best_val_acc
) = train_coatnet(
    coatnet_model,
    train_loader,
    val_loader
)


# ==========================================================
# 14. CALCULATE METRICS
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


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


class_accuracy = (
    cm.diagonal()
    /
    np.where(
        cm.sum(axis=1) == 0,
        1,
        cm.sum(axis=1)
    )
)


hours, remainder = divmod(
    total_training_time,
    3600
)

minutes, seconds = divmod(
    remainder,
    60
)


# ==========================================================
# 15. FINAL RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("FINAL CoAtNet EVALUATION METRICS")
print("=" * 60)


print(
    f"Total Training Time : "
    f"{int(hours):02d}h "
    f"{int(minutes):02d}m "
    f"{seconds:05.2f}s"
)


print(
    f"Overall Accuracy    : "
    f"{accuracy * 100:.2f}% "
    f"({accuracy:.4f})"
)


print(
    f"Weighted Precision  : "
    f"{precision:.4f}"
)


print(
    f"Weighted Recall     : "
    f"{recall:.4f}"
)


print(
    f"Weighted F1 Score   : "
    f"{f1:.4f}"
)


print(
    f"Best Validation Acc : "
    f"{best_val_acc * 100:.2f}%"
)


# ==========================================================
# 16. CLASS-WISE ACCURACY
# ==========================================================

print("\nCLASS-WISE ACCURACY")
print("-" * 40)


for i, name in enumerate(
    class_names
):

    print(
        f"{name:<30}: "
        f"{class_accuracy[i] * 100:.2f}% "
        f"({class_accuracy[i]:.4f})"
    )


# ==========================================================
# 17. SAVE MAIN METRICS TXT
# ==========================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_metrics.txt"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL CoAtNet EVALUATION METRICS\n"
    )

    f.write("=" * 50 + "\n")

    f.write(
        "Model: coatnet_0_rw_224.sw_in1k\n"
    )

    f.write(
        "Dataset: Rice_Leaf_AUG\n"
    )

    f.write(
        f"Total Images: "
        f"{len(base_dataset)}\n"
    )

    f.write(
        f"Training Images: "
        f"{len(train_indices)}\n"
    )

    f.write(
        f"Validation Images: "
        f"{len(val_indices)}\n"
    )

    f.write(
        f"Epochs: "
        f"{TOTAL_EPOCHS}\n"
    )

    f.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Learning Rate: "
        f"{LEARNING_RATE}\n"
    )

    f.write(
        f"Weight Decay: "
        f"{WEIGHT_DECAY}\n"
    )

    f.write(
        f"Label Smoothing: "
        f"{LABEL_SMOOTHING}\n"
    )

    f.write(
        f"Total Training Time: "
        f"{int(hours):02d}h "
        f"{int(minutes):02d}m "
        f"{seconds:05.2f}s\n"
    )

    f.write(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}% "
        f"({accuracy:.4f})\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{recall:.4f}\n"
    )

    f.write(
        f"Weighted F1 Score: "
        f"{f1:.4f}\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}% "
        f"({best_val_acc:.4f})\n"
    )


print(
    f"\n[✓] Saved metrics to: "
    f"{metrics_path}"
)


# ==========================================================
# 18. SAVE CLASS-WISE ACCURACY
# ==========================================================

classwise_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_classwise_accuracy.txt"
)


with open(
    classwise_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CoAtNet CLASS-WISE ACCURACY\n"
    )

    f.write("=" * 50 + "\n\n")


    for i, name in enumerate(
        class_names
    ):

        f.write(
            f"{name}: "
            f"{class_accuracy[i] * 100:.2f}% "
            f"({class_accuracy[i]:.4f})\n"
        )


print(
    f"[✓] Saved class-wise accuracy to: "
    f"{classwise_path}"
)


# ==========================================================
# 19. CLASSIFICATION REPORT
# ==========================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_classification_report.txt"
)


report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CoAtNet CLASSIFICATION REPORT\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(report)


print(
    f"[✓] Saved classification report to: "
    f"{report_path}"
)


# ==========================================================
# 20. ACCURACY GRAPH
# ==========================================================

accuracy_plot_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_accuracy.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    range(
        1,
        TOTAL_EPOCHS + 1
    ),
    history["train_acc"],
    label="Train Accuracy",
    marker="o"
)


plt.plot(
    range(
        1,
        TOTAL_EPOCHS + 1
    ),
    history["val_acc"],
    label="Validation Accuracy",
    marker="o"
)


plt.title(
    "CoAtNet Accuracy Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

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


print(
    f"[✓] Saved accuracy graph to: "
    f"{accuracy_plot_path}"
)


# ==========================================================
# 21. LOSS GRAPH
# ==========================================================

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_loss.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    range(
        1,
        TOTAL_EPOCHS + 1
    ),
    history["train_loss"],
    label="Train Loss",
    marker="o"
)


plt.plot(
    range(
        1,
        TOTAL_EPOCHS + 1
    ),
    history["val_loss"],
    label="Validation Loss",
    marker="o"
)


plt.title(
    "CoAtNet Loss Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

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


print(
    f"[✓] Saved loss graph to: "
    f"{loss_plot_path}"
)


# ==========================================================
# 22. CONFUSION MATRIX
# ==========================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_confusion_matrix.png"
)


plt.figure(
    figsize=(10, 8)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)


plt.title(
    "CoAtNet Confusion Matrix"
)

plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


print(
    f"[✓] Saved confusion matrix to: "
    f"{confusion_path}"
)


# ==========================================================
# 23. SAVE MODEL
# ==========================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "coatnet_rice_leaf_final.pth"
)


torch.save(
    trained_model.state_dict(),
    model_path
)


print(
    f"\n[✓] Model saved to: "
    f"{model_path}"
)


# ==========================================================
# 24. CREATE PDF REPORT
# ==========================================================

pdf_path = os.path.join(
    OUTPUT_DIR,
    "CoAtNet Output.pdf"
)


with PdfPages(pdf_path) as pdf:


    # ------------------------------------------------------
    # Page 1: Metrics
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(8.27, 11.69)
    )

    plt.axis("off")


    metric_text = (

        "CoAtNet RICE LEAF DISEASE CLASSIFICATION\n"

        + "=" * 45

        + "\n\n"

        f"Model: coatnet_0_rw_224.sw_in1k\n"

        f"Dataset: Rice_Leaf_AUG\n"

        f"Total Images: {len(base_dataset)}\n"

        f"Training Images: {len(train_indices)}\n"

        f"Validation Images: {len(val_indices)}\n"

        f"Epochs: {TOTAL_EPOCHS}\n"

        f"Batch Size: {BATCH_SIZE}\n"

        f"Learning Rate: {LEARNING_RATE}\n"

        f"Weight Decay: {WEIGHT_DECAY}\n"

        f"Label Smoothing: {LABEL_SMOOTHING}\n\n"

        f"Training Time: "
        f"{int(hours):02d}h "
        f"{int(minutes):02d}m "
        f"{seconds:05.2f}s\n\n"

        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n"

        f"Precision: "
        f"{precision:.4f}\n"

        f"Recall: "
        f"{recall:.4f}\n"

        f"F1 Score: "
        f"{f1:.4f}\n"

        f"Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}%\n"
    )


    plt.text(
        0.08,
        0.92,
        metric_text,
        fontsize=13,
        verticalalignment="top",
        family="monospace"
    )


    pdf.savefig(
        fig,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ------------------------------------------------------
    # Page 2: Accuracy
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        range(
            1,
            TOTAL_EPOCHS + 1
        ),
        history["train_acc"],
        label="Train Accuracy",
        marker="o"
    )


    plt.plot(
        range(
            1,
            TOTAL_EPOCHS + 1
        ),
        history["val_acc"],
        label="Validation Accuracy",
        marker="o"
    )


    plt.title(
        "CoAtNet Accuracy Curve"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()


    pdf.savefig(fig)

    plt.close(fig)


    # ------------------------------------------------------
    # Page 3: Loss
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        range(
            1,
            TOTAL_EPOCHS + 1
        ),
        history["train_loss"],
        label="Train Loss",
        marker="o"
    )


    plt.plot(
        range(
            1,
            TOTAL_EPOCHS + 1
        ),
        history["val_loss"],
        label="Validation Loss",
        marker="o"
    )


    plt.title(
        "CoAtNet Loss Curve"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()


    pdf.savefig(fig)

    plt.close(fig)


    # ------------------------------------------------------
    # Page 4: Confusion Matrix
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )


    plt.title(
        "CoAtNet Confusion Matrix"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()


    pdf.savefig(fig)

    plt.close(fig)


print(
    f"[✓] PDF report saved to: "
    f"{pdf_path}"
)


# ==========================================================
# 25. FINAL OUTPUT SUMMARY
# ==========================================================

print("\n" + "=" * 60)

print(
    "CoAtNet TRAINING COMPLETED SUCCESSFULLY"
)

print("=" * 60)


print("\nOutput files:")


for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"  ✓ {os.path.join(OUTPUT_DIR, filename)}"
    )


print("\nFinal Metrics:")

print(
    f"  Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"  Precision: "
    f"{precision:.4f}"
)

print(
    f"  Recall   : "
    f"{recall:.4f}"
)

print(
    f"  F1 Score : "
    f"{f1:.4f}"
)

print(
    f"  Best Val : "
    f"{best_val_acc * 100:.2f}%"
)


print("\nDone.")