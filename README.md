## 🧠 MNIST Edge Deployment Pipeline
Deploying a digit recognition model from training to real-time inference on Raspberry Pi and Jetson Nano using ONNX and TensorRT.

## 🔍 Overview
This project showcases an end-to-end deep learning deployment pipeline:

Train a CNN on the MNIST dataset

Convert the model from Keras → ONNX → TensorRT

Deploy to edge devices (Raspberry Pi and Jetson Nano)

The goal is fast, lightweight inference on resource-constrained hardware.

## 📁 File Structure
File Name	Description
mnist_digits.csv	Cleaned MNIST digit dataset (optional CSV format)

bestmodel (1).keras	Trained Keras model (HDF5)

bestmodel (1).onnx	Converted ONNX version of the Keras model

model (1).trt	TensorRT-optimized model

model (1).tflite	TensorFlow Lite version (optional alt. deployment)

onnx2trt (1).py	Script for converting ONNX → TensorRT

pi.py	Raspberry Pi inference script

Power (1).py	Utility or benchmarking script (can rename for clarity)

## 🚀 Getting Started
🔧 Train the Model
Train a simple CNN on MNIST using Keras.

Save the model (in terminal):

```model.save("bestmodel.keras")```


🔁 Convert to ONNX

Use tf2onnx or keras2onnx to convert:

```python -m tf2onnx.convert --keras bestmodel.keras --output bestmodel.onnx```


⚡ Convert to TensorRT

Use the onnx2trt (1).py script to convert:

```python onnx2trt\ \(1\).py```


🤖 Deploy to Raspberry Pi / Jetson Nano

Upload the .trt model and run pi.py to perform live inference on a camera feed or test set.

## 🛠️ Requirements
Python 3.x

TensorFlow, Keras

ONNX, tf2onnx

TensorRT (for Jetson)

OpenCV (for camera inference)

## 🌐 Deployment Notes
Raspberry Pi will need TensorFlow Lite or pre-converted .trt models.

Jetson Nano runs best with TensorRT + ONNX.

The .tflite model is optional for experiments or fallback deployment.

## ✍️ Author
Benjamin Kaliope
Automation Engineering Intern | Undergraduate Researcher | — University of South Carolina
Focused on Edge AI Robotics, and Real-Time ML Deployment
