import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_gesture_modelv2.keras")


gesture_labels = {0: "Call", 1: "Dislike", 2: "Like", 3: "Peace", 4: "Rock", 5: "Stop"}

model = load_model()

def plot_predictions(predictions):
    plt.figure(figsize=(8, 5))
    plt.bar(gesture_labels.values(), predictions[0], color='skyblue')
    plt.xlabel("Gesty")
    plt.ylabel("Prawdopodobieństwo")
    plt.title("Prawdopodobieństwa dla poszczególnych gestów")
    st.pyplot(plt)

def predict_gesture(img, use_augmentation=False):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if use_augmentation:
        predictions = []
        for _ in range(5):
            augmented_img = tf.image.random_flip_left_right(img_array)
            augmented_img = tf.image.random_brightness(augmented_img, max_delta=0.1)
            predictions.append(model.predict(augmented_img))
        predictions = np.mean(predictions, axis=0)
    else:
        predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return gesture_labels.get(predicted_class, "Nie rozpoznano gestu"), confidence, predictions

st.title("Rozpoznawanie gestów")
st.write("Wgraj zdjęcie z gestem")

uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Wgrany obraz", use_container_width=True)

    use_augmentation = st.checkbox("Użyj augmentacji obrazu dla lepszej pewności", value=False)

    result, confidence, predictions = predict_gesture(img, use_augmentation)
    st.write(f"Rozpoznano gest: {result} (pewność: {confidence:.2f})")

    plot_predictions(predictions)