import streamlit as st
from PIL import Image
from utils import load_model, predict_emotion

st.set_page_config(page_title="Emotion Recognition")

st.title("😊 Facial Emotion Recognition")
st.write("Upload a face image")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

uploaded_file = st.file_uploader(
    "Upload image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Predict"):
        emotion, confidence = predict_emotion(model, image)
        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence:.2f}")
