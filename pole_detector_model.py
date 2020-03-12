import pickle
import os
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import pickle

def featureize(raw_data):
    X = []
    y = []
    
    for d in raw_data:
        features = []
        hull, label = d

        MA, ma, angle = ellispe_features(hull)
        hull_area, rect_area, aspect_ratio = contour_features(hull)

        axis_ratio = float(MA)/ma
        extent = float(hull_area)/rect_area
        angle = np.abs(np.sin(angle *np.pi/180))

        features.append(axis_ratio)
        features.append(extent)
        features.append(angle)
        features.append(aspect_ratio)

        X.append(features)
        y.append(label)
    
    return np.asarray(X).astype(float), np.asarray(y).astype(float)


def ellispe_features(hull):
    angle = 0
    MA = 1
    ma = 1
    try:
        (x,y),(MA,ma),angle = cv.fitEllipse(hull)
    except:
        pass
    return MA, ma, angle

def contour_features(hull):
    hull_area = cv.contourArea(hull)
    x,y,w,h = cv.boundingRect(hull)
    rect_area = w*h
    aspect_ratio =  float(w)/h
    return hull_area, rect_area, aspect_ratio


def plot(X,y):
    pole = X[y == 1]
    not_pole = X[y == 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pole.T[0], pole.T[1], pole.T[2], c='r')
    ax.scatter(not_pole.T[0], not_pole.T[1], not_pole.T[2] ,c='b')
    fig.show()
    plt.show()


def main():

    # Load pickle data
    directory = os.getcwd()
    with open(os.path.join(directory, 'pickle/pole_data.pkl'), 'rb') as file:
        data = pickle.load(file)

        # Featurize raw data
        X, y = featureize(data)

        # Train Model
        model = svm.SVC(kernel='linear')
        model.fit(X, y)

        # Get report on training
        y_hat = model.predict(X)
        print(classification_report(y, y_hat))

        # Pickle model
        with open(os.path.join(os.getcwd(), 'pickle/model.pkl'), 'wb') as file:
            pickle.dump(model, file)

        # Plot the model
        # plot(X,y)


if __name__ == '__main__':
    main()