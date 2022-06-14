import numpy as np
import tensorflow as tf
import pickle

def predict(arr):
    # Load the model
    with open('/home/app/dd_model.sav', 'rb') as f:
        model = pickle.load(f)
    classes = {0:'a deepfake', 1:'real'}
    # return prediction as well as class probabilities
    preds = model.predict([arr])[0]
    return (classes[np.argmax(preds)], preds)