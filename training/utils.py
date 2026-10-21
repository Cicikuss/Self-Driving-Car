import ntpath
import os
import random
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.utils import shuffle
import random


NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "others": [bool, object, str, bytes]
    }


from imgaug import augmenters as iaa


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def load_driving_log(data_dir):
    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    log_path = os.path.join(data_dir, "driving_log.csv")
    data = pd.read_csv(log_path, names=columns)
    for column in ("center", "left", "right"):
        data[column] = data[column].apply(path_leaf)
    return data


def balance_steering_data(data, num_bins=25, samples_per_bin=400, seed=None):
    hist, bins = np.histogram(data["steering"], num_bins)
    remove_list = []
    for i in range(num_bins):
        list_ = []
        for j in range(len(data["steering"])):
            if data["steering"][j] >= bins[i] and data["steering"][j] <= bins[i + 1]:
                list_.append(j)
        list_ = shuffle(list_, random_state=seed)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    data.drop(data.index[remove_list], inplace=True)
    return data, bins, hist

def load_img_steering(data_dir, data):
    image_paths = [
        os.path.join(data_dir, row["center"].strip())
        for _, row in data.iterrows()
    ]
    steerings = data["steering"].astype(float).to_numpy()
    return np.asarray(image_paths), steerings


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image


def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


def random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image


def random_flip(image, steering_angle):
    return cv2.flip(image, 1), -steering_angle


def random_augment(image_path, steering_angle):
    image = mpimg.imread(image_path)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = random_flip(image, steering_angle)
    return image, steering_angle


def preprocess_image(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    return image / 255.0


def batch_generator(image_paths,steering_ang,batch_size,is_training):
  while True:
    batch_img = []
    batch_steering = []
    for i in range(batch_size):
        random_index = random.randint(0,len(image_paths)-1)
        if is_training:
            im,steer = random_augment(image_paths[random_index],steering_ang[random_index])
        else:
            im = mpimg.imread(image_paths[random_index])
            steer = steering_ang[random_index]
        im = preprocess_image(im)
        batch_img.append(im)
        batch_steering.append(steer)
    yield (np.asarray(batch_img),np.asarray(batch_steering))


def nvidia_model():
    model = Sequential(
        [
            Conv2D(24, (5, 5), strides=(2, 2), activation="elu", input_shape=(66, 200, 3)),
            Conv2D(36, (5, 5), strides=(2, 2), activation="elu"),
            Conv2D(48, (5, 5), strides=(2, 2), activation="elu"),
            Conv2D(64, (3, 3), activation="elu"),
            Conv2D(64, (3, 3), activation="elu"),
            Flatten(),
            Dense(100, activation="elu"),
            Dense(50, activation="elu"),
            Dense(10, activation="elu"),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4))
    return model
