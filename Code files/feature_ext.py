import os
import random
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Specify the path to your testing folder
testing_folder = r'TESTING'

# Subfolders for control and pd
control_folder = os.path.join(testing_folder, 'control')
pd_folder = os.path.join(testing_folder, 'pd')

# Load your ResNet50V2 model without the final classification layers
model_path = r'ResNet50V2_model.h5'
model = load_model(model_path)

# Get the names of specific layers to visualize
layer_names_to_visualize = ['conv2_block1_1_conv', 'conv3_block1_1_conv', 'conv4_block1_1_conv']

# Create a model that extracts features from the specified layers
feature_extraction_models = [Model(inputs=model.input, outputs=model.get_layer(name).output) for name in layer_names_to_visualize]

# Function to display feature maps with layer name
def display_feature_maps(feature_maps, layer_name):
    num_feature_maps = min(feature_maps.shape[-1], 2)  # Display up to 2 feature maps
    square = int(np.ceil(np.sqrt(num_feature_maps)))
    fig, axs = plt.subplots(square, square, figsize=(4, 4))  # Adjust the figsize to make the images larger

    fig.suptitle(f'Feature Maps for Layer: {layer_name}', fontsize=10, y=.94)

    for i in range(num_feature_maps):
        ax = axs[i // square, i % square]
        ax.imshow(feature_maps[0, :, :, i], cmap='gray')  # Use cmap='gray'
        ax.axis('off')

    # Remove any unused subplots
    for i in range(num_feature_maps, square * square):
        fig.delaxes(axs.flatten()[i])

    plt.subplots_adjust(wspace=0.02, hspace=0.02)  # Adjust spacing between subplots
    plt.show()

# Choose one random image from each subfolder
control_image = random.choice(os.listdir(control_folder))
pd_image = random.choice(os.listdir(pd_folder))

# Process the control image and visualize feature maps from selected layers
control_image_path = os.path.join(control_folder, control_image)
control_img = image.load_img(control_image_path, target_size=(224, 224))
control_img_array = image.img_to_array(control_img)
control_img_array = preprocess_input(control_img_array.reshape((1,) + control_img_array.shape))

for i, model in enumerate(feature_extraction_models):
    control_features = model.predict(control_img_array)
    display_feature_maps(control_features, layer_names_to_visualize[i])

# Process the pd image and visualize feature maps from selected layers
pd_image_path = os.path.join(pd_folder, pd_image)
pd_img = image.load_img(pd_image_path, target_size=(224, 224))
pd_img_array = image.img_to_array(pd_img)
pd_img_array = preprocess_input(pd_img_array.reshape((1,) + pd_img_array.shape))

for i, model in enumerate(feature_extraction_models):
    pd_features = model.predict(pd_img_array)
    display_feature_maps(pd_features, layer_names_to_visualize[i])
