import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_gesture_modelv3.keras")


gesture_labels = {0: "Call", 1: "Dislike", 2: "Fist", 3: "Like", 4: "Mute", 5: "Ok", 6: "One", 7: "Peace", 8: "Rock", 9: "Stop"}

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


st.markdown("""
    <style>
    .stApp {
        background-color: #b3d9ff;
    }
    .main-title {
        color: #003366;
        font-size: 2.8em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px #99ccff;
    }
    .subtitle {
        color: #003366;
        font-size: 1.3em;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton > button:hover {
        background-color: #004d99;
    }
    .stFileUploader label {
        color: #003366;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-title'>🖐 Rozpoznawanie gestów</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>✨ Wgraj zdjęcie z gestem i poznaj wynik predykcji ✨</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)

    st.markdown("<div class='uploaded-image'>", unsafe_allow_html=True)
    st.image(img, caption="Wgrany obraz", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    use_augmentation = st.checkbox("🔄 Użyj augmentacji obrazu dla lepszej pewności", value=False)

    result, confidence, predictions = predict_gesture(img, use_augmentation)

    if confidence > 0.8:
        st.success(f"🎉 Rozpoznano gest: **{result}** z wysoką pewnością ({confidence:.2f})!")
    elif confidence > 0.5:
        st.warning(f"⚠️ Rozpoznano gest: **{result}** średnia pewność: ({confidence:.2f})")
    else:
        st.error(f"❌ Nie udało się rozpoznać gestu z wystarczającą pewnością, przypuszczalny gest to: (**{result}** ; {confidence:.2f})")

    plot_predictions(predictions)
