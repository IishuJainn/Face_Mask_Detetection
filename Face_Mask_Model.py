import pickle
from keras_preprocessing import image
import os
category=['with_mask','without_mask']
import cv2
data = []
for categories in category:
    path = os.path.join("Dataset", categories)
    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        label = category.index(categories)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        data.append([img, label])

print(len(data))
import random
random.shuffle(data)
X = []
y = []
for features, labels in data:
    X.append(features)
    y.append(labels)
print(len(X))
print(len(y))
import numpy as np
X=np.array(X)
y=np.array(y)
print(X.shape)
X=X/255
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)
print(X_train.shape)
X_test.shape
from keras.applications.vgg16 import VGG16
vgg=VGG16()
from keras import Sequential
model=Sequential()
for layers in vgg.layers[:-1]:
    model.add(layers)
model.summary()
for layer in model.layers:
    layer.trainable=False
from keras.layers import Dense
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))
pickle.dump(model, open('model1.pkl', 'wb'))
