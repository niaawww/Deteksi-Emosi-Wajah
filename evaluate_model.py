from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Model
model = load_model("model/emotion_model.h5")

# Dataset Test
test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# Prediksi
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

# Classification Report
print("\n=== CLASSIFICATION REPORT ===\n")

print(
    classification_report(
        test_generator.classes,
        y_pred,
        target_names=list(test_generator.class_indices.keys())
    )
)

# Confusion Matrix
cm = confusion_matrix(
    test_generator.classes,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(test_generator.class_indices.keys()),
    yticklabels=list(test_generator.class_indices.keys())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()