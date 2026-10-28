
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=8,
    validation_split=0.2,
    subset="training",
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=8,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_ds.class_names
n = len(class_names)

# Use a small subset first to test the code
train_ds = train_ds.take(10)
test_ds = test_ds.take(3)

# 2. Load pretrained VGG16

cnn = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
cnn.trainable = False


# 3. Extract training features
X_train, y_train = [], []
X_test, y_test = [], []


for i, (images, labels) in enumerate(train_ds):
    images = preprocess_input(images.numpy())
    features = cnn.predict(images, verbose=0)

    X_train.extend(features)
    y_train.extend(labels.numpy())



for i, (images, labels) in enumerate(test_ds):
    images = preprocess_input(images.numpy())
    features = cnn.predict(images, verbose=0)

    X_test.extend(features)
    y_test.extend(labels.numpy())


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Feature vector shape:", X_train.shape)

# 4. Build classifier
model = models.Sequential([
    layers.Input(shape=(512,)),
    layers.Dense(n, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train classifier
print("Training classifier...", flush=True)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=8,
    verbose=1
)

# 6. Evaluate

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Test accuracy:", accuracy)

# 7. Predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=range(n))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
).plot(xticks_rotation=45)

plt.tight_layout()
plt.show()

# 9. Precision, recall and F1-score
print(classification_report(
    y_test,
    y_pred,
    labels=range(n),
    target_names=class_names,
    zero_division=0
))
