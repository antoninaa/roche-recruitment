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
    input_data : str
        A path to file with data to use for prediction.
    model_path : str
        A path where trained model is stored.
    sep : str
        A separator sign of files that are being uploaded to DataFrame. Default = ','

    Attributes
    ----------
    data : pandas.DataFrame
        This is where we store input data for prediction.
    model_path : str
        This is where we store path of the trained model.
    """
    
    def __init__(self, input_data, model_path, sep=','):
        self.data = pd.read_csv(input_data, sep=sep)
        self.model_path = model_path

    def run(self):
        """
        Run the steps to predict the survival and print the accuracy score of the prediction.
        """
        
        # Preprocess data and build features.
        preproc_data = Preprocess(self.data)
        preproc = preproc_data.execute()

        bf_predict = BuildFeatures(preproc)
        features = bf_predict.execute()

        X = features.drop("Survived", axis=1)
        y = features["Survived"]

        # Open model and make prediction.
        model_unpickle = open(self.model_path, 'rb')
        model = pkl.load(model_unpickle)
        predictions = model.predict(X)
        model_unpickle.close()
        
        # Print metrics for prediction.
        print("Accuracy is", accuracy_score(y, predictions))
