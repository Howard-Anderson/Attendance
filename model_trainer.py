"""
            Attendance Maintainer: Face Detection Using CNN:

    Author: Howard Anderson.

    Date: 06/02/2024.

    Description: Module to train and save the model.

    Filename: model_trainer.py
"""

# Importing Modules.
import pandas as pd

# Imports from Tensorflow.
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


IMG_DIR = "tf_dataset"

name_count = int(input("\nEnter the Number of People in the Dataset: "))

# Read the Dataset.
dataset = image_dataset_from_directory(
        IMG_DIR,
        labels = "inferred",
        label_mode = "int",
        image_size = (480,480),
        batch_size = 32,
        shuffle = True,
        validation_split = 0.2,
        subset = "training"
)

classnames = dataset.class_names
classnames = {x:y for x,y in enumerate(class_names)}
df_classnames = pd.Dataframe(class_names)
df_classnames.to_csv("lookup.csv")

# Building the Model.
classifier = Sequential([
        Conv2D(32, (3,3), activation = "relu", input_shape = (480,480,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation = "relu"),
        MaxPooling2D((2,2)),
        Flatten()
        Dense(128, activation = "relu"),
        Dense(name_count, activation = "softmax")
])

# Training the Model.
classifer.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
)

print(classifier.summary())

classifier.fit(dataset, epochs = 10)

# Saving the Model.
classifier.save("Classifier.h5")
