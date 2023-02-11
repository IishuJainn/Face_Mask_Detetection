# Face_Mask_Detetection
A model trained to detect if is person is wearing Mask or not.
A VGG16 model is used for training and then open cv is used to detect the mask.
![Mask_Detection](https://user-images.githubusercontent.com/102272183/213639332-0ca33ff2-d42e-47dc-8b86-34203e052e2b.png)

# Face Mask Detection
This project is a simple face mask detection system using OpenCV and a deep learning model built using Keras. The model is trained on a dataset of images of people wearing masks and people without masks and it is capable of detecting the presence of masks in real-time using a webcam feed.

## Requirements
Python 3.x

Keras

OpenCV

Numpy

Pickle

Sklearn

Haarcascade_frontalface_default.xml file

## Running the code

Clone the repository to your local machine

![image](https://user-images.githubusercontent.com/102272183/218272441-89a29d2d-05fd-4a37-a492-f049424d1233.png)

Install the required packages mentioned in the Requirements section.

Run the Face_Mask_Model.py file to train the model and save it as a model1.pkl file.

Run the Face_Mask_Webcam.py file to start the webcam feed and detect masks in real-time.

## Files in the Repository

Face_Mask_Model.py: This file trains the deep learning model and saves the trained model as a pickle file.

Face_Mask_Webcam.py: This file uses the trained model to detect masks in real-time using a webcam feed.

haarcascade_frontalface_default.xml: Haar cascade classifier used to detect faces in images.

Dataset: A folder containing the training images for the model.

## Conclusion

This project provides a simple implementation of a face mask detection system and can be used in real-world applications to enforce mask-wearing policies. The model can be further improved by fine-tuning on a larger dataset and incorporating other computer vision techniques to improve accuracy.
