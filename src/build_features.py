import pandas as pd
# import sklearn


class BuildFeatures:

    def __init__(self, input_data):
        self.data = input_data

    def execute(self):
        """Builds features"""

        self.data["Sex"] = self.data["Sex"].map({'male': 0, 'female': 1})
        self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes

        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1
        self.data["IsAlone"] = 0
        self.data.loc[self.data["FamilySize"] == 1, "IsAlone"] = 1

        return self.data
