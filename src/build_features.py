import pandas as pd
# import sklearn


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

        self.data["Sex"] = self.data["Sex"].map({'male': 0, 'female': 1})
        self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes

        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1
        self.data["IsAlone"] = 0
        self.data.loc[self.data["FamilySize"] == 1, "IsAlone"] = 1

        return self.data
