# test.py
import os
from matplotlib import pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from projet import extract_features, svm 

# Directory containing the images
image_directory = "test_images"

# Get a list of all .jpg files in the directory
test_image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(".jpg")]

fig, axs = plt.subplots(1, len(test_image_paths), figsize=(20, 20))

for i, image_path in enumerate(test_image_paths):
    image = load_and_preprocess_image(image_path)
    features = extract_features(image)
    features = np.reshape(features, (1, -1))
    predicted_label = svm.predict(features)

    img = imread(image_path)
    axs[i].imshow(img)
    axs[i].set_title(f"Predicted label: {predicted_label[0]}")
    axs[i].axis("off")

plt.show()
