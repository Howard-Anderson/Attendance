"""
            Attendance Maintainer: Face Detection Using CNN:

    Author: Howard Anderson.

    Date: 06/02/2024.

    Description: Module to capture images and save them to use it in the dataset.

    Filename: dataset_generator.py
"""

import os
import cv2

output_dir = "tf_dataset"

# Function to capture images from Webcam.
def capture_images(name, count = 500):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for img_count in range(500):
        ret, frame = cap.read()
        cv2.imshow(f"{name}", frame)
        # Save the image in the directory.
        os.makedir(os.path.join(output_dir, name), exist_ok = True)

        if(cv2.waitKey(0) & 0xFF == ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()



capture_images(name = input("\nEnter the Name: "))
