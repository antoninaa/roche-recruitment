import pandas as pd
import pickle as pkl
from sklearn.metrics import accuracy_score

from src.preprocess import Preprocess
from src.build_features import BuildFeatures


class Predict:
    """
    The Predict class predicts the survival of Titanic crash.
    
    Parameters
    ----------
    input_data : pandas.DataFrame
        A path to file with data to use for prediction.
    model_path : str
        A path where trained model is stored.

    Attributes
    ----------
    data : pandas.DataFrame
        This is where we store input data for prediction.
    model_path : str
        This is where we store path of the trained model.
    """
    
    def __init__(self, input_data, model_path):
        self.data = input_data
        self.model_path = model_path

    def run(self):
        """
        Run the steps to predict the survival and print the accuracy score of the prediction.
        """
        data = self.data.copy()
        # Preprocess data and build features.
        preproc_data = Preprocess(data)
        preproc = preproc_data.execute()

        bf_predict = BuildFeatures(preproc)
        features = bf_predict.execute()
        features_list = bf_predict.get_feature_columns()

        X = features[features_list]

        # Open model and make prediction.
        model_unpickle = open(self.model_path, 'rb')
        model = pkl.load(model_unpickle)
        predictions = model.predict(X)
        model_unpickle.close()

        # Save predictions back to original data
        self.data["Predictions"] = predictions

        return predictions

