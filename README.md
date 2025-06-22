# **Real-Time Emotion Detection using FER2013**
This project implements real-time emotion detection using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. It utilizes OpenCV for face detection and TensorFlow/Keras for emotion classification.  
*_Frontend is under development._

## 📌 Features
### ✔️ Trained on FER2013 dataset (7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
### ✔️ Real-time face detection using OpenCV
### ✔️ Deep Learning-based Emotion Classification
### ✔️ Pretrained Model Support (Load saved .h5 model or train from scratch)
### ✔️ Live Webcam Feed for Predictions

## 🚀 How It Works
Load a trained model (emotion_model.h5) if available. If not, train a new one using FER2013.
Capture webcam video, detect faces using Haar cascades.
Preprocess the detected face, resize it to 48x48 pixels, normalize the values.
Predict the emotion using a CNN and display it on the live video feed.
## 📂 Installation
```
pip install pandas numpy matplotlib scikit-learn tensorflow opencv-python
```
## ▶️ Run the Project
```
python emotion_detection.py
```
_Press 'q' to quit the webcam window._
