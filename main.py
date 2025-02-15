import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import cv2


def load_and_prepare_data(filepath='fer2013.csv'):
    data = pd.read_csv(filepath)
    X = np.array([np.fromstring(pixels, sep=' ').reshape(48, 48, 1) for pixels in data['pixels']])
    X = X / 255.0
    y = np.array(data['emotion'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_train_cat = to_categorical(y_train, 7)
    y_test_cat = to_categorical(y_test, 7)
    return X_train, X_test, y_train_cat, y_test_cat, y_test


def build_emotion_detection_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model(model, X_train, y_train_cat, X_test, y_test_cat, model_path='emotion_model.h5'):
    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=20, batch_size=64)
    model.save(model_path)
    print(f'Model saved to {model_path}')


def load_trained_model(model_path='emotion_model.h5'):
    return load_model(model_path)


def predict_emotion(model, image):
    image = image.reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(image)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotion_labels[np.argmax(prediction)]


def real_time_emotion_detection(model):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            emotion = predict_emotion(model, face_resized)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load or train the model
    try:
        model = load_trained_model()
        print("Loaded trained model.")
    except:
        print("Training new model...")
        X_train, X_test, y_train_cat, y_test_cat, y_test = load_and_prepare_data()
        model = build_emotion_detection_model()
        train_and_save_model(model, X_train, y_train_cat, X_test, y_test_cat)

    # Start real-time emotion detection
    real_time_emotion_detection(model)