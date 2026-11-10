
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import sys
sys.path.insert(0, MODEL_DIR)
from preprocessing import load_data

class_names = ["Mild", "Moderate", "No DR", "Proliferate DR", "Severe"]

print("Loading dataset...")
X, y = load_data()
X = X.astype("float32")
y = y.astype("int32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading saved model...")
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "diabetic_retinopathy_model.keras")
)

print("Predicting...")
predictions = model.predict(X_test, batch_size=32)
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\nMODEL EVALUATION")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

report = classification_report(
    y_test, y_pred, labels=[0, 1, 2, 3, 4],
    target_names=class_names, zero_division=0
)
print("\nClassification Report:")
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
display = ConfusionMatrixDisplay(cm, display_labels=class_names)
display.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

metrics = [accuracy, precision, recall, f1]
names = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure()
plt.bar(names, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Evaluation Metrics")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
plt.close()

# Generate sample predictions
sample_dir = os.path.join(OUTPUT_DIR, "sample_predictions")
os.makedirs(sample_dir, exist_ok=True)

for i in range(5):
    image = X_test[i]
    actual = class_names[int(y_test[i])]
    predicted = class_names[int(y_pred[i])]
    confidence = float(np.max(predictions[i])) * 100

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}\n"
        f"Confidence: {confidence:.2f}%"
    )
    plt.savefig(os.path.join(sample_dir, f"prediction_{i+1}.png"))
    plt.close()

print("\nEvaluation completed!")
print("Results saved in output folder.")
print("Sample predictions saved successfully!")
