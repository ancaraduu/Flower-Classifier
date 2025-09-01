import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            background-color: #D6B8EC;
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
        .stButton > button {
            background-color: #896F8F !important;
            color: white !important;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #A084CA !important;
            color: white !important;
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

    # Show bar chart of probabilities for the uploaded image
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0], color="#896F8F")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # Optionally, show confusion matrix and report (for all validation data)
    if st.button("Show Confusion Matrix & Report"):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        val_data = datagen.flow_from_directory(
            'dataset',
            target_size=(180, 180),
            batch_size=32,
            subset='validation',
            class_mode='categorical',
            shuffle=False
        )
        val_pred_probs = model.predict(val_data)
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_true = val_data.classes
        cm = confusion_matrix(val_true, val_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Purples', ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Confusion Matrix')
        st.pyplot(fig2)

        # Classification report as a styled heatmap
        import pandas as pd
        report_dict = classification_report(val_true, val_pred, target_names=class_names, output_dict=True)
        # Remove 'accuracy', 'macro avg', 'weighted avg' for a cleaner table
        report_df = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        sns.heatmap(report_df.iloc[:, :3], annot=True, fmt='.2f', cmap='Purples', cbar=False, ax=ax3)
        ax3.set_title('Classification Report')
        st.pyplot(fig3)

# Footer
st.markdown(
    "<hr style='margin-top:40px;margin-bottom:10px;'>"
    "<div style='text-align:center; color: white;'>Made with ❤️</div>",
    unsafe_allow_html=True
)