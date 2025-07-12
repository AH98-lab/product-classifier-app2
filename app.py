import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="تصنيف المنتجات", layout="centered")
st.title("🛍️ أداة تصنيف صور المنتجات")

model = tf.keras.models.load_model("keras_model.h5")

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

uploaded_file = st.file_uploader("📤 ارفع صورة المنتج", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="الصورة التي تم رفعها", use_column_width=True)

    image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"الفئة المتوقعة: **{predicted_class}**")