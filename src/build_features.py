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
        self.data["Title"] = self.data["Title"].astype('category').cat.codes
        self.data.drop(columns=["Name"], inplace=True)

        # Feature 2: embarked
        self.data["Embarked"] = self.fillna_with_most_common("Embarked")
        self.data["Embarked"] = self.data["Embarked"].astype('category').cat.codes

        # Feature 3: familySize
        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1
        self.data["FamilySize"] = self.frequency_encoding("FamilySize")
        self.data.drop(columns=["SibSp", "Parch"], inplace=True)

        # Feature 4: fare
        self.data["Fare"] = pd.qcut(self.data["Fare"], 5).cat.codes

        # Feature 5: age
        self.data["Age"] = self.fillna_with_mean("Age", "Title")
        self.data["Age"] = pd.qcut(self.data["Age"], 5, duplicates="drop").cat.codes

        # Feature 6: sex
        self.data["Sex"] = self.data["Sex"].map({'male': 0, 'female': 1})

        # Feature 7: pclass (no changes required)

        return self.data

    def fillna_with_most_common(self, column):
        """
        Fill missing values with the most common one in the column.

        Parameters:
        ----------
        column: str
            A column name to be processed.
        Return:
        ______
        pandas.Series
            A series with filled missing values
        """
        most_common = self.data[column].value_counts().idxmax()
        return self.data[column].fillna(most_common)

    def frequency_encoding(self, column):
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

    def fillna_with_mean(self, column2fill, column_cat):
        """
        Fill missing values with the mean values of given category.

        Parameters:
        ----------
        column2fill: str
            A column name that is to be filled.
        column_cat: str
            A column name that we group by

        Return:
        ______
        pandas.Series
            A series with filled missing values
        """
        subset_df = self.data[[column2fill, column_cat]]
        means = subset_df.groupby([column_cat])[column2fill].mean()
        subset_df.set_index([column_cat], inplace=True)
        subset_df = subset_df[column2fill].fillna(means)
        subset_df = subset_df.reset_index()
        return subset_df[column2fill]
