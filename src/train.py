# import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from src.preprocess import Preprocess
from src.build_features import BuildFeatures


class Train:
    """
    The Predict class predicts the survival of Titanic crash.
    
    Parameters
    ----------
    input_data : pandas.DataFrame
        A path to file with data to use for model training.
    model_path : str
        A path where trained model will be stored.
    sep : str
        A separator sign of files that are being uploaded to DataFrame. Default = ','
    
    Attributes
    ----------
    data : pandas.DataFrame
        This is where we store input data for model training.
    model_path : str
        This is where we store path of the model that we will train.
    """
    
    def __init__(self, input_data, model_path, sep=','):
        self.data = pd.read_csv(input_data, sep=sep)
        self.model_path = model_path

    def run(self):
        """
        Run the steps to train the model using RandomForestClassifier, save trained model and print the accuracy score for this model.
        """
        
        # Preprocess data and build features.
        preproc_train_data = Preprocess(self.data)
        preproc = preproc_train_data.execute()

        bf_train = BuildFeatures(preproc)
        features = bf_train.execute()

        # Split the data for training and testing.
        X = features.drop("Survived", axis=1)
        y = features["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        # Create a classifier.
        clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                     max_depth=None, max_features='auto', max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=2, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=140,
                                     n_jobs=None, oob_score=False, random_state=42, verbose=0,
                                     warm_start=False)

        # Fit model and predict on test data.
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)
           
        # Select scoring metrics.
        metric_name = "Train accuracy"
        metric_result = accuracy_score(y_test, y_preds)
        
        # Save model.
        model_pickle = open(self.model_path, 'wb')
        pkl.dump(clf, model_pickle)
        model_pickle.close()

        # Print metrics.
        print(f"{metric_name} for the model is {metric_result}.")
