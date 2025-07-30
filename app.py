import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Clasele în aceeași ordine ca în folderul de antrenare

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered", initial_sidebar_state="expanded")


# Load model
@st.cache_resource
def load_flower_model():
    return load_model("flower_model.h5")


model = load_flower_model()

# Sidebar
with st.sidebar:
    st.title("Flower Classifier")
    st.markdown("""
    **Supported classes:**
    - Daisy
    - Dandelion
    - Rose
    - Sunflower
    - Tulip
    """)


st.markdown("""
<h1 style='text-align: center; color: #8e44ad;'>🌸 Flower Classifier 🌸</h1>
<p style='text-align: center;'>Upload a flower image and find out its species!</p>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    
    with st.spinner('Processing image and predicting species...'):
        # Preprocess image
        img_resized = img.resize((180, 180))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
    st.success(f"🌼 Predicted species: **{predicted_class.capitalize()}** ({confidence:.2f}% confidence)")