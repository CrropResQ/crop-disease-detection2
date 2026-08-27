import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, 
classification_report
def calculate_and_print_metrics(y_true, y_pred, class_names):
    overall_acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print("-" * 30)
    print(f"Overall Accuracy:  {overall_acc:.4f}")
    print(f"Weighted Precision:{precision:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("-" * 30)
    # Class-wise Accuracy
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, 
zero_division=0)
    print("Class-wise Accuracy (Recall):")
    for name in class_names:
        if name in report:
            print(f" - {name}: {report[name]['recall']:.4f}")
    return overall_acc, precision, f1
def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_learning_curves(history, save_path="learning_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Loss Curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    # Accuracy / Gain Curve
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_title("Accuracy (Gain) Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()