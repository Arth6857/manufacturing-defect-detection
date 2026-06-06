"""
Manufacturing Defect Detection - Training Script
Dataset: MVTec Anomaly Detection Dataset
Model: MobileNetV2 (Transfer Learning)
Author: Arth Vichpuria
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 10     # Train only top layers first
EPOCHS_FINETUNE = 20   # Unfreeze last N layers of MobileNetV2
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5
FINE_TUNE_AT  = 100    # Unfreeze from this layer onward
NUM_CLASSES   = 2      # defect / no_defect  (binary for general use)

DATA_DIR    = "data/mvtec/bottle"          # Set to your MVTec dataset root
MODEL_DIR   = "models"
LOG_DIR     = "logs"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── DATA GENERATORS ──────────────────────────────────────────────────────────

def build_generators(data_dir: str):
    """
    Expects directory structure:
        data/mvtec/
            train/
                good/       ← defect-free images
                defective/  ← all anomaly sub-classes merged
            val/
                good/
                defective/
            test/
                good/
                defective/
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ─── MODEL ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2) -> Model:
    """
    MobileNetV2 backbone + custom classification head.
    Binary output (good vs defective) — swap to softmax for multi-class.
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze backbone initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        output = Dense(1, activation="sigmoid")(x)
    else:
        output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def compile_model(model, lr, binary=True):
    loss = "binary_crossentropy" if binary else "categorical_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"),
               tf.keras.metrics.Precision(name="precision"),
               tf.keras.metrics.Recall(name="recall")]
    model.compile(optimizer=Adam(lr), loss=loss, metrics=metrics)


def get_callbacks(phase: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"best_model_{phase}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(LOG_DIR, f"{phase}_{timestamp}")),
    ]


def train(data_dir: str = DATA_DIR):
    print("=" * 60)
    print("  Manufacturing Defect Detection — Training Pipeline")
    print("=" * 60)

    # ── Data
    train_gen, val_gen, test_gen = build_generators(data_dir)
    print(f"\nClass indices: {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Val   samples : {val_gen.samples}")
    print(f"Test  samples : {test_gen.samples}")

    # ── Phase 1: Train head only
    print("\n[Phase 1] Training classification head (backbone frozen)...")
    model, base_model = build_model(NUM_CLASSES)
    compile_model(model, LEARNING_RATE)
    model.summary()

    history_phase1 = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=get_callbacks("phase1"),
    )

    # ── Phase 2: Fine-tune top layers of MobileNetV2
    print(f"\n[Phase 2] Fine-tuning from layer {FINE_TUNE_AT} onward...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    compile_model(model, FINE_TUNE_LR)

    history_phase2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=get_callbacks("phase2"),
    )

    # ── Evaluation
    print("\n[Evaluation] Running on test set...")
    results = model.evaluate(test_gen, verbose=1)
    metrics_names = model.metrics_names
    eval_dict = dict(zip(metrics_names, results))
    print("\nTest Metrics:")
    for k, v in eval_dict.items():
        print(f"  {k:12s}: {v:.4f}")

    # ── Classification report
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    report = classification_report(
        y_true, y_pred,
        target_names=list(test_gen.class_indices.keys()),
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=list(test_gen.class_indices.keys())
    ))

    # ── Save artefacts
    model.save(os.path.join(MODEL_DIR, "defect_detection_mobilenetv2.keras"))
    with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f)

    # ── Plots
    _plot_history(history_phase1, history_phase2)
    _plot_confusion_matrix(y_true, y_pred, list(test_gen.class_indices.keys()))

    print(f"\n✓ Model saved  → {MODEL_DIR}/defect_detection_mobilenetv2.keras")
    print(f"✓ Results saved → {RESULTS_DIR}/")
    return model


def _plot_history(h1, h2):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"]  + h2.history["loss"]
    vloss= h1.history["val_loss"] + h2.history["val_loss"]
    sep  = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, title in zip(
        axes,
        [acc, loss],
        [vacc, vloss],
        ["Accuracy", "Loss"]
    ):
        ax.plot(train_vals, label="Train", linewidth=2)
        ax.plot(val_vals,   label="Val",   linewidth=2)
        ax.axvline(sep - 1, color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Training History — Defect Detection (MobileNetV2)", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150)
    plt.close()
    print("✓ Training history plot saved.")


def _plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("✓ Confusion matrix saved.")


if __name__ == "__main__":
    train()
