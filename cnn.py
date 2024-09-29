import os
import tensorflow as tf
import numpy as np
import cv2
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import Model
from numpy import rollaxis
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# Model configuration
output_dims = 68 * 3  # for facial landmarks
reconstruction_dims = 224 * 224  # for 2D facial reconstruction

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Facial landmarks prediction
x1 = Dense(1024, activation='relu')(x)
x1 = Dropout(.2)(x1)
x1 = Dense(512, activation='relu')(x1)
x1 = Dropout(.2)(x1)
x1 = Dense(256, activation='relu')(x1)
x1 = Dropout(.2)(x1)
landmark_preds = Dense(output_dims)(x1)
landmark_preds = Reshape((68, 3))(landmark_preds)

# 2D facial reconstruction (depth map)
x2 = Dense(1024, activation='relu')(x)
x2 = Dropout(.2)(x2)
x2 = Dense(512, activation='relu')(x2)
x2 = Dropout(.2)(x2)
reconstruction_preds = Dense(reconstruction_dims)(x2)
reconstruction_preds = Reshape((224, 224))(reconstruction_preds)

# Combine into a single model
model = Model(inputs=base_model.input, outputs=[landmark_preds, reconstruction_preds])

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Function to simulate conditions: motion blur, poor lighting, low resolution
def simulate_conditions(img, condition="low_resolution"):
    if condition == "low_resolution":
        # Simulate low resolution by downscaling and then upscaling the image
        low_res = cv2.resize(img, (56, 56))  # Downscale to 1/4 resolution
        return cv2.resize(low_res, (224, 224))  # Upscale back to original resolution

    elif condition == "poor_lighting":
        # Simulate poor lighting by reducing brightness
        return cv2.convertScaleAbs(img, alpha=0.5, beta=-50)  # Reduce brightness

    elif condition == "motion_blur":
        # Simulate motion blur
        size = 15  # Length of the blur effect
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        return cv2.filter2D(img, -1, kernel_motion_blur)

# Function to train the model on individual images
def train_on_image(img_path, condition=None):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    if condition:
        img = simulate_conditions(img, condition)

    # Prepare the input
    img_input = np.expand_dims(img, axis=0)

    # Generate dummy outputs (targets) for training (replace with actual target data in real cases)
    y_landmarks = np.random.random((1, 68, 3))  # Random landmarks (replace with actual data)
    y_reconstruction = np.random.random((1, 224, 224))  # Random reconstruction (replace with actual data)

    # Train on this image
    model.train_on_batch(img_input, [y_landmarks, y_reconstruction])

    print("Training completed on image:", img_path)

# Function to load and train on an image using the GUI
def load_image_for_training():
    file_path = filedialog.askopenfilename()
    if file_path:
        condition = "motion_blur"  # Set to "low_resolution", "poor_lighting", or "motion_blur"
        train_on_image(file_path, condition)

# GUI for loading videos and displaying results
def load_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        condition = "motion_blur"  # Set to "low_resolution", "poor_lighting", or "motion_blur"
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (224, 224))

            # Simulate low-quality conditions (set condition to the desired test case)
            frame_resized = simulate_conditions(frame_resized, condition)

            # Prepare the input
            frame_input = np.expand_dims(frame_resized, axis=0)
            preds = model.predict(frame_input)

            # Show the prediction output (landmarks & reconstruction)
            landmark_output = preds[0]
            reconstruction_output = preds[1]

            # Visualization of landmarks
            img = cv2.resize(frame, (224, 224))
            for landmark in landmark_output[0]:
                cv2.circle(img, (int(landmark[0]), int(landmark[1])), 2, (255, 0, 0), -1)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            lbl_image.config(image=imgtk)
            lbl_image.image = imgtk

            # Visualization of 2D reconstruction (depth map)
            reconstruction_img = np.uint8(reconstruction_output[0] * 255)
            imgtk_recon = ImageTk.PhotoImage(image=Image.fromarray(reconstruction_img))
            lbl_reconstruction.config(image=imgtk_recon)
            lbl_reconstruction.image = imgtk_recon

        cap.release()

# GUI setup
root = tk.Tk()
root.title("Facial Recognition & 2D Reconstruction")

lbl_image = tk.Label(root)
lbl_image.pack()

lbl_reconstruction = tk.Label(root)
lbl_reconstruction.pack()

btn_load_video = tk.Button(root, text="Load Video", command=lambda: threading.Thread(target=load_video).start())
btn_load_video.pack()

btn_load_image = tk.Button(root, text="Train on Image", command=lambda: threading.Thread(target=load_image_for_training).start())
btn_load_image.pack()

root.mainloop()