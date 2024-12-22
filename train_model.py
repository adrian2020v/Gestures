#model
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

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()


# def detect_and_crop_hand(img):
#     img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)
#
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             x_min = min([landmark.x for landmark in hand_landmarks.landmark])
#             y_min = min([landmark.y for landmark in hand_landmarks.landmark])
#             x_max = max([landmark.x for landmark in hand_landmarks.landmark])
#             y_max = max([landmark.y for landmark in hand_landmarks.landmark])
#
#             h, w, _ = img.shape
#             x_min, x_max = int(x_min * w), int(x_max * w)
#             y_min, y_max = int(y_min * h), int(y_max * h)
#
#             hand_region = img[y_min:y_max, x_min:x_max]
#             return hand_region
#     return None


img_height, img_width = 128, 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_dir = 'dataset_det/train'
test_dir = 'dataset_det/test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

num_classes = len(train_generator.class_indices)

base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_gesture_modelv3.keras', monitor='val_accuracy', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# def predict_gesture_with_hand_detection(img, use_augmentation=False):
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#
#     hand_region = detect_and_crop_hand(np.array(img))
#     if hand_region is not None:
#
#         hand_region = cv2.resize(hand_region, (img_width, img_height))
#         hand_image = Image.fromarray(hand_region)
#
#         img_array = np.array(hand_image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#
#         if use_augmentation:
#             predictions = []
#             for _ in range(5):
#                 augmented_img = tf.image.random_flip_left_right(img_array)
#                 augmented_img = tf.image.random_brightness(augmented_img, max_delta=0.1)
#                 predictions.append(model.predict(augmented_img))
#             predictions = np.mean(predictions, axis=0)
#         else:
#             predictions = model.predict(img_array)
#
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         confidence = np.max(predictions)
#         return gesture_labels.get(predicted_class, "Nie rozpoznano gestu"), confidence, predictions
#     return "Brak wykrytej dłoni", 0, None


history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[checkpoint, lr_scheduler],
    verbose=1
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")
