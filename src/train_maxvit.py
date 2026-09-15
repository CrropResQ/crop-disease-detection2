# ============================================================
# MAXVIT
# Rice Leaf Disease Classification
# ============================================================

import os
import time
import random
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import timm

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. SETTINGS
# ============================================================

EPOCHS = 25

# IMPORTANT:
# Physical batch that actually goes through the GPU
BATCH_SIZE = 4

# We accumulate 8 batches:
# 4 x 8 = effective batch size 32
GRADIENT_ACCUMULATION = 8

EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE * GRADIENT_ACCUMULATION
)

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.01

TRAIN_RATIO = 0.80

RANDOM_SEED = 42

IMAGE_SIZE = 224

NUM_WORKERS = 0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 3. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "Rice_Leaf_AUG"
)


# ============================================================
# 4. OUTPUT PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "best_maxvit_rice_model.pth"
)

ACCURACY_PATH = (
    PROJECT_ROOT
    / "maxvit_accuracy_curve.png"
)

LOSS_PATH = (
    PROJECT_ROOT
    / "maxvit_loss_curve.png"
)

CONFUSION_PATH = (
    PROJECT_ROOT
    / "maxvit_confusion_matrix.png"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "maxvit_classification_report.txt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "maxvit_metrics.txt"
)

CLASSWISE_PATH = (
    PROJECT_ROOT
    / "maxvit_classwise_accuracy.txt"
)


# ============================================================
# 5. HEADER
# ============================================================

print("=" * 70)
print("MAXVIT")
print("RICE LEAF DISEASE CLASSIFICATION")
print("=" * 70)

print("\nProject folder:")
print(PROJECT_ROOT)

print("\nDataset folder:")
print(DATASET_DIR)


# ============================================================
# 6. CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nDataset not found!\n"
        f"Expected location:\n{DATASET_DIR}\n"
    )


# ============================================================
# 7. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:")
print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ============================================================
# 8. IMAGE TRANSFORMS
# ============================================================
#
# IMPORTANT:
# MaxViT model is configured for 224x224.
# DO NOT change this to 128x128.
#
# 224 / 4 = 56
# 56 -> 28 -> 14 -> 7
#
# These dimensions work correctly with MaxViT.
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 9. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_full_dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=train_transform
)

val_full_dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=val_transform
)


# ============================================================
# 10. CLASS INFORMATION
# ============================================================

class_names = train_full_dataset.classes

num_classes = len(class_names)

print("\nClasses:")

for index, class_name in enumerate(class_names):

    print(
        f"{index}: {class_name}"
    )

print(
    "\nNumber of classes:",
    num_classes
)

print(
    "Total images:",
    len(train_full_dataset)
)


# ============================================================
# 11. CHECK CLASS MAPPING
# ============================================================

if (
    train_full_dataset.class_to_idx
    !=
    val_full_dataset.class_to_idx
):

    raise ValueError(
        "Training and validation class mappings are different!"
    )


# ============================================================
# 12. TRAIN / VALIDATION SPLIT
# ============================================================

dataset_size = len(
    train_full_dataset
)

train_size = int(
    TRAIN_RATIO * dataset_size
)

val_size = (
    dataset_size - train_size
)


generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)


indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()


train_indices = indices[
    :train_size
]

val_indices = indices[
    train_size:
]


train_dataset = Subset(
    train_full_dataset,
    train_indices
)

val_dataset = Subset(
    val_full_dataset,
    val_indices
)


print("\nDataset split:")

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# 13. DATALOADERS
# ============================================================

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


print(
    "\nDataLoaders created successfully!"
)


# ============================================================
# 14. LOAD MAXVIT
# ============================================================

print("\nLoading MaxViT...")


model = timm.create_model(

    "maxvit_tiny_tf_224",

    pretrained=True,

    num_classes=num_classes
)


model = model.to(device)


print(
    "MaxViT loaded successfully!"
)


# ============================================================
# 15. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 16. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 17. MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# 18. HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_acc": [],

    "val_loss": [],

    "val_acc": []
}


best_val_accuracy = 0.0


# ============================================================
# 19. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    optimizer.zero_grad(
        set_to_none=True
    )


    progress_bar = tqdm(

        train_loader,

        desc="Training"
    )


    for batch_index, (images, labels) in enumerate(
        progress_bar
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

                # Gradient accumulation
                loss_for_backward = (
                    loss
                    /
                    GRADIENT_ACCUMULATION
                )

        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss_for_backward = (
                loss
                /
                GRADIENT_ACCUMULATION
            )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()


        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        if (
            (batch_index + 1)
            %
            GRADIENT_ACCUMULATION
            == 0
        ):

            if use_amp:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()


            optimizer.zero_grad(
                set_to_none=True
            )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predicted = outputs.argmax(
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        current_accuracy = (
            correct / total
        )


        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{current_accuracy:.4f}"
        )


        # Free temporary tensors

        del images
        del labels
        del outputs
        del loss


    # --------------------------------------------------------
    # Remaining gradients
    # --------------------------------------------------------

    if (
        len(train_loader)
        %
        GRADIENT_ACCUMULATION
        != 0
    ):

        if use_amp:

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()


        optimizer.zero_grad(
            set_to_none=True
        )


    epoch_loss = (
        running_loss
        /
        total
    )

    epoch_accuracy = (
        correct
        /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 20. VALIDATION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        progress_bar = tqdm(

            val_loader,

            desc="Validating"
        )


        for images, labels in progress_bar:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            predicted = outputs.argmax(
                dim=1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )


            del images
            del labels
            del outputs
            del loss


    epoch_loss = (
        running_loss
        /
        total
    )

    epoch_accuracy = (
        correct
        /
        total
    )


    return (

        epoch_loss,

        epoch_accuracy,

        all_labels,

        all_predictions
    )


# ============================================================
# 21. START TRAINING
# ============================================================

print("\n")

print("=" * 70)

print(
    "STARTING MAXVIT TRAINING"
)

print("=" * 70)


print(
    f"\nEpochs: {EPOCHS}"
)

print(
    f"Physical GPU batch size: {BATCH_SIZE}"
)

print(
    f"Gradient accumulation: {GRADIENT_ACCUMULATION}"
)

print(
    f"Effective batch size: {EFFECTIVE_BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}"
)


start_time = time.time()


# ============================================================
# 22. TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n")

    print("-" * 70)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        all_labels,
        all_predictions
    ) = validate()


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_acc"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_accuracy
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nEpoch Results:")

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "num_classes":
                    num_classes,

                "epoch":
                    epoch + 1,

                "val_accuracy":
                    val_accuracy,

                "image_size":
                    IMAGE_SIZE
            },

            MODEL_PATH
        )


        print(
            f"\nBest model saved!"
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )


    # --------------------------------------------------------
    # CUDA cleanup
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    gc.collect()


# ============================================================
# 23. TRAINING TIME
# ============================================================

end_time = time.time()

training_seconds = (
    end_time - start_time
)

training_minutes = (
    training_seconds / 60
)


print("\n")

print("=" * 70)

print(
    "TRAINING COMPLETED"
)

print("=" * 70)

print(
    f"Training time: "
    f"{training_minutes:.2f} minutes"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 24. LOAD BEST MODEL
# ============================================================

print("\nLoading best MaxViT model...")


checkpoint = torch.load(

    MODEL_PATH,

    map_location=device
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print(
    "Best model loaded successfully!"
)


# ============================================================
# 25. FINAL VALIDATION
# ============================================================

(
    final_val_loss,

    final_val_accuracy,

    all_labels,

    all_predictions

) = validate()


# ============================================================
# 26. FINAL METRICS
# ============================================================

accuracy = accuracy_score(

    all_labels,

    all_predictions
)


precision = precision_score(

    all_labels,

    all_predictions,

    average="weighted",

    zero_division=0
)


recall = recall_score(

    all_labels,

    all_predictions,

    average="weighted",

    zero_division=0
)


f1 = f1_score(

    all_labels,

    all_predictions,

    average="weighted",

    zero_division=0
)


print("\n")

print("=" * 70)

print(
    "FINAL MAXVIT RESULTS"
)

print("=" * 70)


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 27. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    zero_division=0
)


print("\n")

print("=" * 70)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 70)

print(report)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions
)


fig, ax = plt.subplots(

    figsize=(9, 9)
)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)


display.plot(

    ax=ax,

    xticks_rotation=45,

    cmap="Blues",

    colorbar=True
)


plt.title(
    "MaxViT - Rice Leaf Disease Confusion Matrix"
)

plt.tight_layout()


plt.savefig(

    CONFUSION_PATH,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    CONFUSION_PATH
)


# ============================================================
# 29. CLASS-WISE ACCURACY
# ============================================================

classwise_text = ""

for i, class_name in enumerate(
    class_names
):

    class_total = cm[i].sum()

    class_correct = cm[i, i]


    if class_total > 0:

        class_accuracy = (
            class_correct
            /
            class_total
            *
            100
        )

    else:

        class_accuracy = 0


    line = (
        f"{class_name}: "
        f"{class_accuracy:.2f}%"
    )


    print(line)

    classwise_text += (
        line + "\n"
    )


with open(
    CLASSWISE_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        classwise_text
    )


# ============================================================
# 30. METRICS FILE
# ============================================================

metrics_text = f"""

MAXVIT RICE LEAF DISEASE CLASSIFICATION
========================================

Epochs: {EPOCHS}

Physical Batch Size: {BATCH_SIZE}

Gradient Accumulation:
{GRADIENT_ACCUMULATION}

Effective Batch Size:
{EFFECTIVE_BATCH_SIZE}

Learning Rate:
{LEARNING_RATE}

Image Size:
{IMAGE_SIZE} x {IMAGE_SIZE}

Training Images:
{len(train_dataset)}

Validation Images:
{len(val_dataset)}

Best Validation Accuracy:
{best_val_accuracy * 100:.2f}%

Final Accuracy:
{accuracy * 100:.2f}%

Precision:
{precision * 100:.2f}%

Recall:
{recall * 100:.2f}%

F1 Score:
{f1 * 100:.2f}%

Training Time:
{training_minutes:.2f} minutes
"""


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        metrics_text
    )


# ============================================================
# 31. ACCURACY CURVE
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    epochs_range,

    [
        x * 100
        for x in history["train_acc"]
    ],

    label="Training Accuracy"
)


plt.plot(

    epochs_range,

    [
        x * 100
        for x in history["val_acc"]
    ],

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "MaxViT Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(

    ACCURACY_PATH,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 32. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    epochs_range,

    history["train_loss"],

    label="Training Loss"
)


plt.plot(

    epochs_range,

    history["val_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "MaxViT Training and Validation Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(

    LOSS_PATH,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 33. FINAL OUTPUT
# ============================================================

print("\n")

print("=" * 70)

print(
    "ALL MAXVIT RESULTS SAVED"
)

print("=" * 70)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nAccuracy graph:"
)

print(
    ACCURACY_PATH
)

print(
    "\nLoss graph:"
)

print(
    LOSS_PATH
)

print(
    "\nConfusion matrix:"
)

print(
    CONFUSION_PATH
)

print(
    "\nClassification report:"
)

print(
    REPORT_PATH
)

print(
    "\nMetrics:"
)

print(
    METRICS_PATH
)

print("\n")

print("=" * 70)

print(
    "DONE!"
)

print("=" * 70)