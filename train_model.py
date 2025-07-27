import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setări
img_height, img_width = 180, 180
batch_size = 32

# Încarcă imaginile din folder
train_ds = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_ds.flow_from_directory(
    "dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset="training",
    class_mode="categorical"
)

val_data = train_ds.flow_from_directory(
    "dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical"
)

# Model CNN simplu
model = Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 clase de flori
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)

# Salvează modelul
model.save("flower_model.h5")
