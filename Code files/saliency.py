import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\91866\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\xai\files\resnet_feature\ResNet50V2_model.h5')

# Define paths to testing data for both "PD" and "Control" classes
test_data_dir_pd = r'C:\Users\91866\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\xai\files\resnet_feature\TESTING\PD'
test_data_dir_control = r'C:\Users\91866\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\xai\files\resnet_feature\TESTING\Control'

# Define image size
img_size = (224, 224)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet_v2.preprocess_input(image_array)
    return image_array

# Function to compute the saliency map using GradientTape
def compute_saliency_map(model, image):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        prediction = predictions[0]
    gradient = tape.gradient(prediction, image_tensor)
    saliency_map = tf.reduce_max(tf.abs(gradient), axis=-1)
    return saliency_map.numpy()

# Function to visualize the original image and saliency map with inverted colormap
def visualize_saliency(image_path, saliency_map):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(tf.keras.preprocessing.image.load_img(image_path, target_size=img_size))
    plt.title('Original Image')
    plt.axis('off')
    
    # Set the background color of the saliency map to black
    saliency_map_black_bg = np.where(saliency_map > 0, saliency_map, 0)
    
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map_black_bg, cmap='hsv', alpha=0.5, vmin=0, vmax=1)
    plt.title('Saliency Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Choose random images from both classes and visualize their saliency maps
random.seed(42)  # Set seed for reproducibility

# Random image from "PD" class
pd_images = os.listdir(test_data_dir_pd)
pd_image_path = os.path.join(test_data_dir_pd, random.choice(pd_images))
pd_image = load_and_preprocess_image(pd_image_path)
pd_saliency_map = compute_saliency_map(model, pd_image)
pd_saliency_map_normalized = (pd_saliency_map - np.min(pd_saliency_map)) / (np.max(pd_saliency_map) - np.min(pd_saliency_map))
print("PD Saliency Map Values:")
print(pd_saliency_map_normalized)
visualize_saliency(pd_image_path, pd_saliency_map_normalized[0])
print("PD")

# Random image from "Control" class
control_images = os.listdir(test_data_dir_control)
control_image_path = os.path.join(test_data_dir_control, random.choice(control_images))
control_image = load_and_preprocess_image(control_image_path)
control_saliency_map = compute_saliency_map(model, control_image)
control_saliency_map_normalized = (control_saliency_map - np.min(control_saliency_map)) / (np.max(control_saliency_map) - np.min(control_saliency_map))
print("Control Saliency Map Values:")
print(control_saliency_map_normalized)
visualize_saliency(control_image_path, control_saliency_map_normalized[0])
print("Control")