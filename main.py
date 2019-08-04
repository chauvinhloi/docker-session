import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request
from flasgger import Swagger

with open('./model.pkl', 'rb') as model_pkl:
    knn = pickle.load(model_pkl)


app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """
    API endpoint for predicting iris species
    ---
    parameters:
        - name: sl
          description: sepal length
          in: query
          type: number
          required: true
        - name: sw
          description: sepal width
          in: query
          type: number
          required: true
        - name: pl
          description: petal length
          in: query
          type: number
          required: true
        - name: pw
          description: petal width
          in: query
          type: number
          required: true
    responses:
        '200':
          description: OK

    """
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    pl = request.args.get('pl')
    pw = request.args.get('pw')

    unseen = np.array([[sl, sw, pl, pw]]).astype(np.float64)
    result = knn.predict(unseen)

    return 'Predicted result for observation ' + str(unseen) + ' is: ' + str(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)