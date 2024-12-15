import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

# Inicjalizacja MediaPipe do wykrywania dłoni
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


# Funkcja do wycięcia dłoni z obrazu
def detect_and_crop_hand(img):
    # Konwertowanie obrazu do RGB
    img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Określenie regionu dłoni na podstawie współrzędnych wykrytych punktów
            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
            y_max = max([landmark.y for landmark in hand_landmarks.landmark])

            # Skala obrazu w stosunku do współrzędnych wykrytej dłoni
            h, w, _ = img.shape
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            # Wycięcie regionu dłoni
            hand_region = img[y_min:y_max, x_min:x_max]
            return hand_region
    return None


# Parametry modelu
img_height, img_width = 128, 128
batch_size = 32

# Generatory danych z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizacja pikseli
    rotation_range=40,  # Zwiększenie zakresu rotacji
    width_shift_range=0.4,  # Zwiększenie zakresu przesunięcia
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.4,  # Zwiększenie zoomu
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # Zmiana jasności
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Tylko normalizacja

# Generatory treningowe i testowe
train_dir = 'dataset_det/train'
test_dir = 'dataset_det/test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Wieloklasowy model
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

# Liczba klas
num_classes = len(train_generator.class_indices)

# Wczytanie pretrenowanego modelu MobileNetV2
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = True  # Odblokowanie warstw MobileNetV2

# Zamrożenie tylko początkowych warstw (fine-tuning od ostatnich warstw)
fine_tune_at = 100  # np. fine-tune od 100-tej warstwy
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Dodanie własnych warstw klasyfikujących
model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Zmniejszenie dropout do 0.3
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Zmniejszenie początkowego współczynnika uczenia
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_gesture_modelv2.keras', monitor='val_accuracy', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Funkcja do przewidywania gestu z detekcją dłoni
def predict_gesture_with_hand_detection(img, use_augmentation=False):
    # Konwertowanie obrazu do formatu RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Detekcja i przycięcie dłoni
    hand_region = detect_and_crop_hand(np.array(img))
    if hand_region is not None:
        # Resize obrazu dłoni do wymagań modelu
        hand_region = cv2.resize(hand_region, (img_width, img_height))
        hand_image = Image.fromarray(hand_region)

        # Normalizacja
        img_array = np.array(hand_image) / 255.0  # Normalizacja
        img_array = np.expand_dims(img_array, axis=0)  # Dodanie wymiaru batcha

        if use_augmentation:
            # Augmentacja obrazu i uśrednianie wyników
            predictions = []
            for _ in range(5):  # 5 wersji obrazu
                augmented_img = tf.image.random_flip_left_right(img_array)
                augmented_img = tf.image.random_brightness(augmented_img, max_delta=0.1)
                predictions.append(model.predict(augmented_img))
            predictions = np.mean(predictions, axis=0)
        else:
            # Standardowa predykcja
            predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        return gesture_labels.get(predicted_class, "Nie rozpoznano gestu"), confidence, predictions
    return "Brak wykrytej dłoni", 0, None


# Trenowanie modelu
history = model.fit(
    train_generator,
    epochs=20,  # Zwiększenie liczby epok
    validation_data=test_generator,
    callbacks=[checkpoint, lr_scheduler],  # Dodanie scheduler
    verbose=1
)

# Ewaluacja na danych testowych
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")
