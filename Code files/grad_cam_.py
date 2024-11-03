import os
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import cv2

# Load your saved model
model = load_model('ResNet50V2_model.h5')

# Define the target layer for Grad-CAM
target_layer = 'conv4_block1_1_conv'  # Replace with the actual name of the target layer

# Define the Grad-CAM function
def grad_cam(model, img_array, layer_name):
    # Define the gradient function for the target layer
    grad_model = Model(inputs=[model.input], outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Calculate gradients of the target class with respect to the output feature map
    grads = tape.gradient(loss, conv_output)[0]

    # Compute the guided gradients
    guided_grads = (tf.cast(conv_output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads)

    # Compute the weights using global average pooling
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

    # Compute the weighted sum of the output feature map
    cam = np.dot(conv_output[0], weights)

    return cam

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

        # Generate Grad-CAM
        cam = grad_cam(model, img_array, target_layer)

        # Resize CAM to match the input image size
        cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))

        # Normalize CAM values to be in the range [0, 1]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        # Convert to uint8 before using cv2.addWeighted
        img_bgr = cv2.cvtColor(img_array[0], cv2.COLOR_RGB2BGR).astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # Combine the heatmap with the original image
        superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)

        # Display the results with subfolder information
        plt.imshow(superimposed_img)
        plt.title(f'{subfolder} - Grad-CAM')
        plt.show()
