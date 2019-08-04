import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request

with open('./model.pkl', 'rb') as model_pkl:
    knn = pickle.load(model_pkl)


app = Flask(__name__)

@app.route('/predict')
def predict_iris():
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    pl = request.args.get('pl')
    pw = request.args.get('pw')

    unseen = np.array([[sl, sw, pl, pw]])
    result = knn.predict(unseen)

    return 'Predicted result for observation ' + str(unseen) + ' is: ' + str(result)


if __name__ == '__main__':
    app.run()