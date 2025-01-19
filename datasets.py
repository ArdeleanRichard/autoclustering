import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff

def transform_arff_data(data):
    X = []
    y = []
    for sample in data:
        x = []
        for id, value in enumerate(sample):
            if id == len(sample) - 1:
                y.append(value)
            else:
                x.append(value)
        X.append(x)


    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)



def create_2d4c():
    data, meta = arff.loadarff('./datasets/2d-4c.arff')
    return transform_arff_data(data)

def create_2d10c():
    data, meta = arff.loadarff('./datasets/2d-10c.arff')
    return transform_arff_data(data)




