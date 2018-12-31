import time
import os, os.path
import random
import cv2
import glob
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import cv2
import os
import csv

# directory where images are stored
DIR = "DB_Clustering"


def dataset_stats():
    # This is an array with the letters available.
    # If you add another animal later, you will need to structure its images in the same way
    # and add its letter to this array
    subfolder = ['2']

    # dictionary where we will store the stats
    stats = []

    for animal in subfolder:
        # get a list of subdirectories that start with this character
        directory_list = sorted(glob.glob("{}/[{}]*".format(DIR, animal)))

        for sub_directory in directory_list:
            file_names = [file for file in os.listdir(sub_directory)]
            file_count = len(file_names)
            sub_directory_name = os.path.basename(sub_directory)
            stats.append({"Code": sub_directory_name[:sub_directory_name.find('-')],
                          "Image count": file_count,
                          "Folder name": os.path.basename(sub_directory),
                          "File names": file_names})

    df = pd.DataFrame(stats)

    return df

# Show codes with their folder names and image counts
dataset = dataset_stats().set_index("Code")
#print(dataset[["Folder name", "Image count"]])


# Function returns an array of images whoose filenames start with a given set of characters
# after resizing them to 224 x 224

def load_images(codes):
    # Define empty arrays where we will store our images and labels
    images = []
    labels = []

    for code in codes:
        # get the folder name for this code
        folder_name = dataset.loc[code]["Folder name"]

        for file in dataset.loc[code]["File names"]:
            # build file path
            file_path = os.path.join(DIR, folder_name, file)

            # Read the image
            image = cv2.imread(file_path)

            # Resize it to 224 x 224
            image = cv2.resize(image, (64, 64))

            # Convert it from BGR to RGB so we can plot them later (because openCV reads images as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Now we add it to our array
            images.append(image)
            labels.append(code)

    return images, labels


codes = ["2"]
images, labels = load_images(codes)


def show_random_images(images, labels, number_of_images_to_show=2):

    for code in list(set(labels)):

        indicies = [i for i, label in enumerate(labels) if label == code]
        random_indicies = [random.choice(indicies) for i in range(number_of_images_to_show)]
        figure, axis = plt.subplots(1, number_of_images_to_show)

        print("{} random images for code {}".format(number_of_images_to_show, code))

        for image in range(number_of_images_to_show):
            axis[image].imshow(images[random_indicies[image]])
        plt.show()


#show_random_images(images, labels)


def normalise_images(images, labels):
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Normalise the images
    images /= 255

    return images, labels


images, labels = normalise_images(images, labels)


def shuffle_data(images, labels):
    # Set aside the testing data. We won't touch these until the very end.
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0, random_state=728)

    return X_train, y_train


#X_train, y_train = shuffle_data(images, labels)
X_train = images

# Load the models with ImageNet weights
vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(64, 64, 3))

# vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(64, 64, 3))

# resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(64, 64, 3))


def covnet_transform(covnet_model, raw_images):
    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)

    return flat


vgg16_output = covnet_transform(vgg16_model, X_train)
print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))

# vgg19_output = covnet_transform(vgg19_model, X_train)
# print("VGG19 flattened output has {} features".format(vgg19_output.shape[1]))

# resnet50_output = covnet_transform(resnet50_model, X_train)
# print("ResNet50 flattened output has {} features".format(resnet50_output.shape[1]))


# Function that creates a PCA instance, fits it to the data and returns the instance
def create_fit_PCA(data, n_components=None):
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)

    return p


# Create PCA instances for each covnet output
vgg16_pca = create_fit_PCA(vgg16_output)
# vgg19_pca = create_fit_PCA(vgg19_output)
# resnet50_pca = create_fit_PCA(resnet50_output)

# PCA transformations of covnet outputs
vgg16_output_pca = vgg16_pca.transform(vgg16_output)
# vgg19_output_pca = vgg19_pca.transform(vgg19_output)
# resnet50_output_pca = resnet50_pca.transform(resnet50_output)


def create_train_kmeans(data, number_of_clusters=2):
    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters

    k = KMeans(n_clusters=number_of_clusters, n_jobs=-1, random_state=728)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing
    end = time.time()

    # And see how long that took
    print("Training took {} seconds".format(end - start))

    return k


# Let's pass the data into the algorithm and predict who lies in which cluster.
# Since we're using the same data that we trained it on, this should give us the training results.

# Here we create and fit a KMeans model with the PCA outputs
print("KMeans (PCA): \n")

print("VGG16")
K_vgg16_pca = create_train_kmeans(vgg16_output_pca)

# print("\nVGG19")
# K_vgg19_pca = create_train_kmeans(vgg19_output_pca)

# print("\nResNet50")
# K_resnet50_pca = create_train_kmeans(resnet50_output_pca)


k_vgg16_pred_pca = K_vgg16_pca.predict(vgg16_output_pca)
# k_vgg19_pred_pca = K_vgg19_pca.predict(vgg19_output_pca)
# k_resnet50_pred_pca = K_resnet50_pca.predict(resnet50_output_pca)

print(k_vgg16_pred_pca)

i = 0
with open('file.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['image_id', 'classe'])
    for file in glob.glob(os.path.join("DB_Clustering/2-Classes", '*.jpg')):
        # read the image
        image = cv2.imread(file)
        # extract image name from the path & delete jpg extension
        image_name = os.path.basename(file)
        filewriter.writerow([image_name, k_vgg16_pred_pca[i]+1])
        i += 1

