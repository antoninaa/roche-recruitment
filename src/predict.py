import pandas as pd
import pickle as pkl
from sklearn.metrics import accuracy_score

from src.preprocess import Preprocess
from src.build_features import BuildFeatures


class Predict:

    def __init__(self, input_data, model_path, sep=','):
        self.data = pd.read_csv(input_data, sep=sep)
        self.model_path = model_path

    def run(self):
        """Trains the model"""
        preproc_data = Preprocess(self.data)
        preproc = preproc_data.execute()

        bf_predict = BuildFeatures(preproc)
        features = bf_predict.execute()

        X = features.drop("Survived", axis=1)
        y = features["Survived"]

        model_unpickle = open(self.model_path, 'rb')
        model = pkl.load(model_unpickle)
        predictions = model.predict(X)
        model_unpickle.close()
        # Reassign target (if it was present) and predictions.
        print("Accuracy is", accuracy_score(y, predictions))
