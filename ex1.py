import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    image_size=(224, 224),
    batch_size=5,
    shuffle=True
)

print("Classes:", dataset.class_names)

images, labels = next(iter(dataset))

print("Original shape:", images.shape)
print("Preprocessed shape: (5, 224, 224, 3)")


plt.figure(figsize=(12, 5))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(dataset.class_names[labels[i]])
    plt.axis("off")

plt.show()


model = VGG16(weights="imagenet")

x = preprocess_input(images)

print("Pixel value before preprocessing:", images[0][0][0])
print("Pixel value after preprocessing :", x[0][0][0])

predictions = model.predict(x)

results = decode_predictions(predictions, top=5)

for i in range(5):
    print("\nImage", i + 1)
    print("Original label:", dataset.class_names[labels[i]])
    print("Top-5 predictions:")

    for _, label, probability in results[i]:
        print(label,_, ":", round(probability * 100, 2), "%")

    print("Predicted label:", results[i][0][1])