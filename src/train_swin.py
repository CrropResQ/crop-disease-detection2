# ============================================================
# SWIN TRANSFORMER - RICE LEAF DISEASE CLASSIFICATION
# ============================================================

import os

# Helps reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
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

# RTX 3050 4 GB
BATCH_SIZE = 32

# Gradient accumulation
# Effective batch size = 2 x 4 = 8
ACCUMULATION_STEPS = 8

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.01

TRAIN_RATIO = 0.80

RANDOM_SEED = 42

IMAGE_SIZE = 224


# ============================================================
# 2. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "Rice_Leaf_AUG"
)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("SWIN TRANSFORMER")
print("RICE LEAF DISEASE CLASSIFICATION")
print("=" * 70)

print("\nProject folder:")
print(PROJECT_ROOT)

print("\nDataset folder:")
print(DATASET_DIR)

print("\nDevice:")
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
# 4. CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_DIR}"
    )


# ============================================================
# 5. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
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
        (IMAGE_SIZE, IMAGE_SIZE)
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
# 6. LOAD DATASET
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
# 7. CLASS INFORMATION
# ============================================================

class_names = train_full_dataset.classes

num_classes = len(class_names)


print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        f"{i}: {name}"
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
# 8. TRAIN / VALIDATION SPLIT
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
# 9. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


print(
    "\nDataLoaders created successfully!"
)


# ============================================================
# 10. LOAD SWIN TRANSFORMER
# ============================================================

print("\nLoading Swin Transformer...")


model = timm.create_model(

    "swin_tiny_patch4_window7_224",

    pretrained=True,

    num_classes=num_classes

)


model = model.to(device)


print(
    "Swin Transformer loaded successfully!"
)


# ============================================================
# 11. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 13. MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# 14. HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_acc": [],

    "val_loss": [],

    "val_acc": []

}


best_val_accuracy = 0.0


# ============================================================
# 15. TRAIN FUNCTION
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
        # Forward pass
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
                    loss / ACCUMULATION_STEPS
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
                loss / ACCUMULATION_STEPS
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

        should_update = (

            (batch_index + 1)
            % ACCUMULATION_STEPS == 0

            or

            (batch_index + 1)
            == len(train_loader)

        )


        if should_update:

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
        # Statistics
        # ----------------------------------------------------

        running_loss += (

            loss.item()
            * images.size(0)

        )


        _, predicted = torch.max(

            outputs,

            1

        )


        total += labels.size(0)


        correct += (

            predicted == labels

        ).sum().item()


        current_acc = (
            correct / total
        )


        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{current_acc:.4f}"

        )


        # Release temporary references
        del outputs
        del loss
        del loss_for_backward


    epoch_loss = (

        running_loss
        / total

    )


    epoch_accuracy = (

        correct
        / total

    )


    return (

        epoch_loss,

        epoch_accuracy

    )


# ============================================================
# 16. VALIDATION FUNCTION
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


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            running_loss += (

                loss.item()
                * images.size(0)

            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            _, predicted = torch.max(

                outputs,

                1

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


            del outputs
            del loss


    epoch_loss = (

        running_loss
        / total

    )


    epoch_accuracy = (

        correct
        / total

    )


    return (

        epoch_loss,

        epoch_accuracy,

        all_labels,

        all_predictions

    )


# ============================================================
# 17. TRAINING START
# ============================================================

print("\n")

print("=" * 70)

print(
    "STARTING SWIN TRANSFORMER TRAINING"
)

print("=" * 70)


print(
    f"\nEpochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Gradient accumulation: {ACCUMULATION_STEPS}"
)

print(
    f"Effective batch size: "
    f"{BATCH_SIZE * ACCUMULATION_STEPS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)


start_time = time.time()


# ============================================================
# 18. EPOCH LOOP
# ============================================================

for epoch in range(EPOCHS):


    print("\n")

    print("-" * 70)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_acc = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    (
        val_loss,

        val_acc,

        all_labels,

        all_predictions

    ) = validate()


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_acc"].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_acc
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nEpoch Results:")

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_acc:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_acc:.4f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc


        model_path = (

            PROJECT_ROOT
            / "best_swin_transformer.pth"

        )


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "num_classes":
                    num_classes,

                "best_val_accuracy":
                    best_val_accuracy,

                "epoch":
                    epoch + 1

            },

            model_path

        )


        print(
            "\n✓ BEST MODEL SAVED"
        )


        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.4f}"
        )


    # --------------------------------------------------------
    # Clear unused GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 19. TRAINING TIME
# ============================================================

end_time = time.time()


training_seconds = (

    end_time
    - start_time

)


training_minutes = (

    training_seconds
    / 60

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
    f"{best_val_accuracy:.4f}"
)


# ============================================================
# 20. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")


model_path = (

    PROJECT_ROOT
    / "best_swin_transformer.pth"

)


checkpoint = torch.load(

    model_path,

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
# 21. FINAL VALIDATION
# ============================================================

(
    final_val_loss,

    final_val_accuracy,

    all_labels,

    all_predictions

) = validate()


# ============================================================
# 22. FINAL METRICS
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
    "FINAL SWIN TRANSFORMER RESULTS"
)

print("=" * 70)


print(
    f"Accuracy : {accuracy:.4f}"
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
# 23. CLASSIFICATION REPORT
# ============================================================

print("\n")

print("=" * 70)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 70)


print(

    classification_report(

        all_labels,

        all_predictions,

        target_names=class_names,

        zero_division=0

    )

)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")


cm = confusion_matrix(

    all_labels,

    all_predictions

)


fig, ax = plt.subplots(

    figsize=(10, 10)

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

    "Swin Transformer - Confusion Matrix"

)


plt.tight_layout()


confusion_path = (

    PROJECT_ROOT
    / "swin_confusion_matrix.png"

)


plt.savefig(

    confusion_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ============================================================
# 25. LOSS CURVE
# ============================================================

epochs_range = range(

    1,

    EPOCHS + 1

)


plt.figure(

    figsize=(9, 5)

)


plt.plot(

    epochs_range,

    history["train_loss"],

    marker="o",

    label="Training Loss"

)


plt.plot(

    epochs_range,

    history["val_loss"],

    marker="o",

    label="Validation Loss"

)


plt.xlabel("Epoch")

plt.ylabel("Loss")


plt.title(

    "Swin Transformer - Loss Curve"

)


plt.legend()

plt.grid(True)


plt.tight_layout()


loss_path = (

    PROJECT_ROOT
    / "swin_loss_curve.png"

)


plt.savefig(

    loss_path,

    dpi=300

)


plt.show()


# ============================================================
# 26. ACCURACY CURVE
# ============================================================

plt.figure(

    figsize=(9, 5)

)


plt.plot(

    epochs_range,

    history["train_acc"],

    marker="o",

    label="Training Accuracy"

)


plt.plot(

    epochs_range,

    history["val_acc"],

    marker="o",

    label="Validation Accuracy"

)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")


plt.title(

    "Swin Transformer - Accuracy Curve"

)


plt.legend()

plt.grid(True)


plt.tight_layout()


accuracy_path = (

    PROJECT_ROOT
    / "swin_accuracy_curve.png"

)


plt.savefig(

    accuracy_path,

    dpi=300

)


plt.show()


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print(
    "FILES CREATED"
)

print("=" * 70)


print(
    "\n1. Best Model:"
)

print(
    model_path
)


print(
    "\n2. Confusion Matrix:"
)

print(
    confusion_path
)


print(
    "\n3. Loss Curve:"
)

print(
    loss_path
)


print(
    "\n4. Accuracy Curve:"
)

print(
    accuracy_path
)


print("\n")

print("=" * 70)

print(
    "SWIN TRANSFORMER TRAINING COMPLETED ✓"
)

print("=" * 70)