from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Clasele în aceeași ordine ca în folderul de antrenare
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Încarcă modelul
model = load_model("flower_model.h5")

# Încarcă poza
img_path = "test_flower.jpg"  # pune aici poza ta
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prezicere
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"Specia prezisă: {predicted_class}")
