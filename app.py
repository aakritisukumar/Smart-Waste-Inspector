import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

model = load_model("model/garbage_classifier_model_legacy.h5", compile=False)

bin_images = {
    "Recyclable ♻️": "assets/recycle-bin.png",
    "Organic 🍃": "assets/compost-bin.png",
    "General Waste 🗑️": "assets/garbage.png"
}


class_names = [
    "batteries", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

class_to_bin = {
    "plastic": "Recyclable ♻️",
    "metal": "Recyclable ♻️",
    "paper": "Recyclable ♻️",
    "cardboard": "Recyclable ♻️",
    "white-glass": "Recyclable ♻️",
    "green-glass": "Recyclable ♻️",
    "brown-glass": "Recyclable ♻️",
    "biological": "Organic 🍃",
    "trash": "General Waste 🗑️",
    "clothes": "General Waste 🗑️",
    "shoes": "General Waste 🗑️",
    "batteries": "General Waste 🗑️"  
}

bin_colors = {
    "Recyclable ♻️": "#1abc9c",   
    "Organic 🍃": "#8bc34a",      
    "General Waste 🗑️": "#bdc3c7"  
}

def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = np.max(preds)
    bin_type = class_to_bin.get(pred_class, "Unknown")

    return pred_class, bin_type, confidence

# Streamlit UI
st.set_page_config(page_title="Smart Waste Inspector ♻️", layout="centered")
st.title("🧠 Smart Waste Inspector")
st.markdown("Upload a photo of your trash item to see what bin it belongs to.")


uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        pred_class, bin_type, confidence = predict_image(img)

    color = bin_colors.get(bin_type, "#cccccc")

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.markdown(f"**Predicted Class:** `{pred_class}`")
    st.markdown(f"**Bin Suggestion:** `{bin_type}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
    
    if bin_type in bin_images:
        st.image(bin_images[bin_type], width=200, caption=bin_type)

    
    st.markdown(
        f"<div style='background-color:{color};padding:15px;border-radius:10px;'>"
        f"<h3 style='text-align:center;'>{bin_type}</h3></div>",
        unsafe_allow_html=True
    )
