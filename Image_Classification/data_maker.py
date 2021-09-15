#!/usr/bin/env python3
"""
1. extracts data from a location
2. Labels them (dependant on folder structure)
3. returns training and testing data

"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def initialize_configs():
    config_dict = {
        "IMG_SIZE": 256,
        "train_data_path": "\\\\TBD.com\\us\\shared\\rwe\\CS_DAV\\DataFolder\\cone_stain_images_WK2133\\train\\*",
        "test_data_path": "\\\\TBD.com\\us\\shared\\wer\\CS_DAV\\DataFolder\\cone_stain_images_WK2133\\test\\*",
    }

    return config_dict


def pre_process_image(img_path, config_dict):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (config_dict["IMG_SIZE"], config_dict["IMG_SIZE"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def convert_images_to_array(images):
    array_images = np.array(images)

    return array_images


def get_data(config_dict):
    # Training Data
    train_images = []
    train_labels = []

    for directory_path in glob.glob(config_dict["train_data_path"]):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = pre_process_image(img_path, config_dict)
            train_images.append(img)
            train_labels.append(label)

    # Convert lists to arrays
    train_images = convert_images_to_array(train_images)
    train_labels = np.array(train_labels)

    # Testing Data
    test_images = []
    test_labels = []

    for directory_path in glob.glob(config_dict["test_data_path"]):
        test_label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = pre_process_image(img_path, config_dict)
            test_images.append(img)
            test_labels.append(test_label)

    # Convert lists to arrays
    test_images = convert_images_to_array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


def main():
    # Create configuration dictionary
    config_dict = initialize_configs()

    # Get data
    train_images, train_labels, test_images, test_labels = get_data(config_dict)

    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    main()
