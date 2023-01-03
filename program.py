import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#import databases
x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

#view no. of values of each class in y dataset
print(pd.Series(y).value_counts())

#assign classes as each letter
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)

#split data for training and testing, scale x values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=9, train_size=7500, test_size=2500)
xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0

#create a model, make a prediction and check accuracy
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(xtrainscaled, ytrain)
ypred = clf.predict(xtestscaled)
accuracy = accuracy_score(ytest, ypred)
print("The accuracy is : ", accuracy)

#open camera
cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #define a box in which to detect the alphabet
        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(height / 2 - 56))
        bottom_right = (int(width/2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        #take the image and convert as required
        im_pil = Image.fromarray(roi)

        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixelfilter = 20
        minpixel = np.percentile(image_bw_resized_inverted, pixelfilter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - minpixel, 0, 255)
        maxpixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/maxpixel

        #make a prediction on the test sample
        testsample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        testpred = clf.predict(testsample)
        print("Predicted class is : ", testpred)

        cv2.imshow('frame', gray)
        
        #if 'q' key is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #if an error is present, exit
    except Exception as e:
        pass

#close camera
cap.release()
cv2.destroyAllWindows()