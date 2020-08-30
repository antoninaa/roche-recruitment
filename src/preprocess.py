import pandas as pd
# import numpy as np


class Preprocess:

    def __init__(self, input_data):
        self.data = input_data

    def execute(self):

        self.data.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
        self.data.dropna(inplace=True)
        return self.data
