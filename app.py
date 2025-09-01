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

# Add custom CSS for background, fonts, and upload box
st.markdown("""
    <style>
        .stApp {
            background-color: #C6A1CF;
        }
        h1, .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption, .stDataFrame, .stTable, .stSuccess, .stSpinner, .stFileUploader, .stButton, .stSidebar, .css-10trblm, .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
            color: white !important;
        }
        .stSidebar {
            background-color: #896F8F !important;
            min-width: 340px !important;
            max-width: 340px !important;
        }
        .stFileUploader > div {
            border: 2px dashed #896F8F;
            background-color: #C6A1CF;
            border-radius: 12px;
            padding: 16px;
        }
        .stFileUploader label, .stFileUploader span, .stFileUploader .css-1cpxqw2, .stFileUploader .css-1c7y2kd, .stFileUploader .css-1aehpvj, .stFileUploader .e1b2p2ww10 {
            color: #896F8F !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='margin-top: 50px; text-align:center; font-weight:bold; font-size:1.6em;'>🌸 <b>Flower Classifier</b> 🌸</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-left: 60px; margin-top: 40px; font-size:1.3em;'>
    <b>Supported classes:</b><br>
    - 🌼 Daisy<br>
    - 🌻 Dandelion<br>
    - 🌹 Rose<br>
    - 🌻 Sunflower<br>
    - 🌷 Tulip
    </div>
    """, unsafe_allow_html=True)


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

# Footer
st.markdown(
    "<hr style='margin-top:40px;margin-bottom:10px;'>"
    "<div style='text-align:center; color: white;'>Made with ❤️</div>",
    unsafe_allow_html=True
)