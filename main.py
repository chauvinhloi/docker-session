import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

with open('./model.pkl', 'rb') as model_pkl:
    knn = pickle.load(model_pkl)

unseen = np.array([[3.2, 1.1, 1.5, 2.1]])
result = knn.predict(unseen)

print('Predicted result for observation ' + str(unseen) + ' is: ' + str(result))