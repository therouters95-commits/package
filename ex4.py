
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = tf.keras.utils.image_dataset_from_directory(
    "dataset", image_size=(224, 224), batch_size=8,shuffle=True
)
classes = data.class_names
ds = data.take(15)

captions = ["<start> a photo of a " + c + " <end>" for c in classes]
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(captions)

vocab_size = len(tokenizer.word_index) + 1
seqs = tokenizer.texts_to_sequences(captions)
max_len = max(len(s) for s in seqs)

cnn = VGG16(weights="imagenet", include_top=False, pooling="avg")
cnn.trainable = False

features, inputs, targets = [], [], []

for images, labels in ds:
    f = cnn.predict(preprocess_input(images.numpy()), verbose=0)

    for i, label in enumerate(labels.numpy()):
        seq = tokenizer.texts_to_sequences([captions[label]])[0]
        for j in range(1, len(seq)):
            features.append(f[i])
            inputs.append(seq[:j])
            targets.append(seq[j])

inputs = pad_sequences(inputs, maxlen=max_len, padding="post")
features = np.array(features)
targets = np.array(targets)

image_input = layers.Input(shape=(512,))
caption_input = layers.Input(shape=(max_len,))

x = layers.Embedding(vocab_size, 64)(caption_input)
x = layers.LSTM(128)(x)
x = layers.Concatenate()([image_input, x])
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(vocab_size, activation="softmax")(x)

model = Model([image_input, caption_input], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit([features, inputs], targets, epochs=5, batch_size=8)

test_ds=data.skip(15)
images, labels = next(iter(test_ds))

image = images[0:1]

feature = cnn.predict(preprocess_input(image.numpy()), verbose=0)
caption = ["<start>"]

for _ in range(max_len):
    seq = tokenizer.texts_to_sequences([" ".join(caption)])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict([feature, seq], verbose=0)
    word = tokenizer.index_word.get(np.argmax(pred[0]), "")
    if word in ("<end>", ""):
        break
    caption.append(word)

print("Original class:", classes[labels[0].numpy()])
print("Generated caption:", " ".join(caption[1:]))

plt.imshow(images[0].numpy().astype("uint8"))
plt.axis("off")
plt.show()
