import pandas as pd
# import numpy as np


class Preprocess:
    """
    The Preprocess class handles preprocessing of input data.
    
    Parameters
    ----------
    input_data : pandas.DataFrame
        Input data that will be preprocessed
    
    Attributes
    ----------
    data : pandas.DataFrame
        This is where we store input data
    """

    def __init__(self, input_data):
        self.data = input_data

    def execute(self):
        """
        Execute steps to preprocess data.
        
        Return
        ------
        pandas.DataFrame
            Preprocessed data.
        """
        
        self.data.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
        self.data.dropna(inplace=True)
        return self.data
