import csv
import cv2
import numpy as np
import os
import sklearn

#Import driving_log data which captures the image location, steering angle, speed, throttle & brake
lines = []
with open('/home/carnd/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
for line in lines:
    for i in range(3): #looping in 3 times to include in left, center & right camera images
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/home/carnd/CarND-Behavioral-Cloning-P3/data/IMG/'+ filename
        image = cv2.imread(current_path)
        if image is None:
            print("Image path incorrect: ", current_path)
            continue  # Included this for debugging
        images.append(image)
        # if center camera image is used then use the steering angle as captured by simulator. Otherwise
        # adjust it by a factor of 0.18
        if i==0: 
            measurement = float(line[3])
        elif i==1:
            measurement = float(line[3])+0.18
        else:
            measurement = float(line[3])-0.18
        measurements.append(measurement)


X_train=np.array(images)
y_train=np.array(measurements)

#Model architecture  
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Normalizing/mean centering data through lambda function
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
#Cropping top section (70 pixels) to remove the seciton which contains sky
#Cropping bottom section (25 pixels) which displays the hood of the car
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

model.save('model.h5')