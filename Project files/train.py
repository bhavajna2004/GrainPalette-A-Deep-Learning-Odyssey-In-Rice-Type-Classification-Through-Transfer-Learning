import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

train_dir = 'data/'
model_save_path = 'models/rice.h5'
img_size = (224, 224)
batch_size = 32
epochs = 10

datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    subset='training', class_mode='categorical'
)
val_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    subset='validation', class_mode='categorical'
)

model = models.Sequential([
    layers.Input(shape=(*img_size, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),

    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
model.save('models/rice.keras')

print(f"Model saved to {model_save_path}")

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig('models/training_plot.png')
