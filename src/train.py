# import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocess import Preprocess
from src.build_features import BuildFeatures


class Train:

    def __init__(self, input_data, model_path, sep=','):
        self.data = pd.read_csv(input_data, sep=sep)
        self.model_path = model_path

    def run(self):
        """Trains the model"""
        preproc_train_data = Preprocess(self.data)
        preproc = preproc_train_data.execute()

        bf_train = BuildFeatures(preproc)
        features = bf_train.execute()

        # Split the data for training.
        X = features.drop("Survived", axis=1)
        y = features["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create a classifier and select scoring methods.
        clf = RandomForestClassifier(n_estimators=10)

        # Fit full model and predict on both train and test.
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)

        metric_name = "train_accuracy"
        metric_result = accuracy_score(y_test, y_preds)

        model_pickle = open(self.model_path, 'wb')
        pkl.dump(clf, model_pickle)
        model_pickle.close()

        # Return metrics and model.
        print(f"{metric_name} for the model is {metric_result}.")
