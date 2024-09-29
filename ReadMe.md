#Facial Recognition and 2D Reconstruction

This repository contains a model for performing 2D facial recognition and facial reconstruction using TensorFlow, Keras, and MobileNetV2. It is designed to handle low-quality video footage, focusing on challenges like motion blur, poor lighting, and low resolution.

|Table of Contents      |
|-----------------------|
|Requirements           |
|Installation           |
|Dataset                |
|How to Train           |
|Running the Application|
|Test Cases             |
|License                |
|Requirements           |


*Ensure you have Python 3.7+ installed on your system.*

The main dependencies for this project include:

**TensorFlow (2.x)**
**Keras**
**NumPy**
**OpenCV**
**scikit-image**
**Pillow**
**Tkinter (for GUI)**

---
##Installation

Follow these steps to install the necessary dependencies:

Clone the repository:

bash
`git clone https://github.com/your-name/ETHOS`
`cd ETHOS`

*Install the required packages:*

bash
`pip install -r requirements.txt`

*(Optional) Install GPU version of TensorFlow if you plan to use GPU acceleration:*

bash
`pip install tensorflow-gpu==2.9.0`

---
##Dataset

This model expects a dataset that includes:

Facial Images: Low-quality images/videos of faces (e.g., from CCTV footage).

Landmark Coordinates: 68 facial key points in 3D.

Depth Maps (2D Reconstruction Targets): Grayscale images for pseudo-depth reconstruction.

Make sure your dataset follows this structure:


>/dataset_directory
>    /subfolder_1
>       data_1.npz
>        data_2.npz
>       ...
>    /subfolder_2
>        data_1.npz
>        data_2.npz
>        ...

Each .npz file contains:

    colorImages: The facial images.
    landmarks3D: The corresponding 3D landmarks.

You can modify the data loading part in the code to fit your specific dataset format.

---
##How to Train

Once the dataset is ready, follow these steps to train the model:

Set the dataset path in the code. For example:

python
`data_path = "../input/youtube-faces-with-facial-keypoints/"  # Update this to your dataset path`

Modify the hyperparameters in the train_model() function if necessary:

```batch_size = 16
epochs = 50
steps_per_epoch = 100```

**Start training the model:**

bash
`python train.py`


The train.py script will begin training the model for both facial landmark prediction and 2D facial reconstruction simultaneously. It uses mean squared error as the loss function for both outputs.

During training, the script will output metrics like mean absolute error (MAE) and loss for each batch.

Key Points:
**Training takes place on both facial landmarks and 2D reconstruction at the same time.**
**The dataset should contain both RGB images for facial recognition and pseudo-depth (grayscale) targets for reconstruction.**
**Running the Application**
**Once the model is trained, you can load a video and get real-time predictions (facial landmarks and 2D reconstruction):**

##Run the GUI application:

bash
`python gui.py`

Use the GUI to load a video file with low-quality footage:

Click the **"Load Video"** button and select your video.
The model will predict facial landmarks and 2D reconstruction, displaying both in the GUI.

---
##Test Cases

The model has been tested under the following conditions:

Low Resolution: Handles videos with lower pixel density and fewer details.
Poor Lighting: Robust against dark or dim environments where facial features are hard to distinguish.
Motion Blur: Effective in handling blurred frames caused by fast motion in videos.
Make sure your dataset includes a mix of these conditions to train the model effectively for real-world applications.


[-1]:This project is licensed under the MIT License.