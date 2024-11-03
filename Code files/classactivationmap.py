import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import matplotlib.pyplot as plt
import cv2

# Load your saved model
model = load_model('ResNet50V2_model.h5')

# Function to compute CAM
def compute_cam(model, img_array):
    # Get the feature maps from the last convolutional layer
    last_conv_layer = model.get_layer('conv5_block3_out')  # Adjust this based on your model architecture
    last_conv_output = last_conv_layer.output

    # Define a model that outputs the CAM given the input image
    cam_model = Model(inputs=model.input, outputs=[last_conv_output, model.output])

    # Get the CAM result
    with tf.GradientTape() as tape:
        # Compute the output of the last convolutional layer given the input image
        conv_outputs, predictions = cam_model(img_array)
        class_idx = tf.argmax(predictions[0])
        output = predictions[:, class_idx]

    # Calculate the gradients of the class output value with respect to the feature maps
    grads = tape.gradient(output, conv_outputs)

    # Vectorize the gradients to obtain the importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map array by its importance weight
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, :, i] = tf.multiply(conv_outputs[:, :, :, i], pooled_grads[i])

    # Compute the CAM by summing the weighted feature maps along the channel dimension
    cam = tf.reduce_sum(conv_outputs, axis=-1)

    # Normalize CAM
    cam = np.maximum(cam, 0)  # ReLU activation
    cam /= tf.reduce_max(cam)  # Normalize

    return cam.numpy()

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Folder containing subfolders 'Control' and 'PD'
testing_folder = 'TESTING'

# List of subfolders
subfolders = ['Control', 'PD']

# Iterate over subfolders
for subfolder in subfolders:
    # Get the list of image files in the subfolder
    img_files = os.listdir(os.path.join(testing_folder, subfolder))

    # Take the first image from the subfolder
    if img_files:
        img_path = os.path.join(testing_folder, subfolder, img_files[0])
        img_array = load_and_preprocess_image(img_path)

        # Compute CAM
        cam = compute_cam(model, img_array)

        # Resize CAM to match the input image size
        cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))

        # Convert to uint8 before using cv2.addWeighted
        img_bgr = cv2.cvtColor(img_array[0], cv2.COLOR_RGB2BGR).astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # Combine the heatmap with the original image
        cam_overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)

        # Display the results with subfolder information
        plt.imshow(cam_overlay)
        plt.title(f'{subfolder} - Class Activation Map')
        plt.axis('off')
        plt.show()
