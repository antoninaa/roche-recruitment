import pandas as pd
import numpy as np


class BuildFeatures:
    """
    The BuildFeatures class prepares the proper input dataset for machine learning algorithms.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Input data for building features.

    Attributes
    ----------
    data : pandas.DataFrame
        This is where we store input data.
    """
    
    def __init__(self, input_data):
        self.data = input_data
        self.feature_columns = []

    def execute(self):
        """
        Build features

        Return
        ------
        pandas.DataFrame
            Dataset with proper format for ML algorithms.
        """

        # Feature 1: title
        self.extract_title("Name")
        self.data["Title"] = self.data["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3})
        self.data.drop(columns=["Name"], inplace=True)
        self.feature_columns.append("Title")

        # Feature 2: embarked
        self.data["Embarked"] = self.data["Embarked"].map({'C': 0, 'Q': 1, 'S': 2})
        self.feature_columns.append("Embarked")

        # Feature 3: familySize
        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1
        self.data["FamilySize"] = self.family_size_frequency_encoding("FamilySize")
        self.data.drop(columns=["SibSp", "Parch"], inplace=True)
        self.feature_columns.append("FamilySize")

        # Feature 4: fare
        self.data["Fare"] = self.fare_frequency_encoding("Fare")
        self.feature_columns.append("Fare")

        # Feature 5: age
        self.data["Age"] = pd.qcut(self.data["Age"], 5, duplicates="drop").cat.codes
        self.feature_columns.append("Age")

        # Feature 6: sex
        self.data["Sex"] = self.data["Sex"].map({'male': 0, 'female': 1})
        self.feature_columns.append("Sex")

        # Feature 7: pclass (no changes required)
        self.feature_columns.append("Pclass")

        return self.data

    def get_feature_columns(self):
        """
        Return list of feature columns.

        Return:
        ______
        list
            List of column names

        """
        return self.feature_columns

    def family_size_frequency_encoding(self, column):
        """
        Encode column based on frequency (hardcoded)

        Parameters:
        ----------
        column: str
            A column name to be processed.
        Return:
        ______
        pandas.Series
            A series with codes
        """
        conditions = [
            (self.data[column] == 1),
            (self.data[column] >= 2) & (self.data[column] <= 4),
            (self.data[column] >= 5) & (self.data[column] <= 7),
            (self.data[column] >= 8)]

        choices = [0, 1, 2, 3]

        return np.select(conditions, choices, default=-1)

    def fare_frequency_encoding(self, column):
        """
        Encode column based on frequency (hardcoded)

        Parameters:
        ----------
        column: str
            A column name to be processed.
        Return:
        ______
        pandas.Series
            A series with codes
        """
        conditions = [
            (self.data[column] <= 7.854),
            (self.data[column] > 7.854) & (self.data[column] <= 10.5),
            (self.data[column] > 10.5) & (self.data[column] <= 22.225),
            (self.data[column] > 22.225) & (self.data[column] <= 39.688),
            (self.data[column] > 39.688)]

        choices = [0, 1, 2, 3, 4]

        return np.select(conditions, choices, default=-1)

    def age_frequency_encoding(self, column):
        """
        Encode column based on frequency (hardcoded)

        Parameters:
        ----------
        column: str
            A column name to be processed.
        Return:
        ______
        pandas.Series
            A series with codes
        """
        conditions = [
            (self.data[column] <= 19),
            (self.data[column] > 19) & (self.data[column] <= 25),
            (self.data[column] > 25) & (self.data[column] <= 32),
            (self.data[column] > 32) & (self.data[column] <= 42),
            (self.data[column] > 42)]

        choices = [0, 1, 2, 3, 4]

        return np.select(conditions, choices, default=-1)

    def extract_title(self, column):
        """
        Extract title from name and fixes the wrong entries

        Parameters:
        ----------
        column: str
            A column name to be processed.
        """
        self.data["Title"] = self.data[column].str.split(r'. |, ', expand=True)[1]
        title_list = ["Mr", "Miss", "Mrs", "Master"]
        self.data["Title"] = self.data.loc[self.data["Sex"] == 'female', "Title"].apply(lambda x: x if x in title_list else "Miss")
        self.data.loc[(self.data["Title"].isna()) & (self.data["Sex"] == "female"), "Title"] = "Miss"
        self.data.loc[(self.data["Title"].isna()) & (self.data["Sex"] == "male"), "Title"] = "Mr"


