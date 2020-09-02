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

        self.data = self.data.astype({'PassengerId':'int32',
                   'Pclass':'int',
                   'Name':'str',
                   'Sex':'str',
                   'Age':'float',
                   'SibSp':'int',
                   'Parch':'int',
                   'Ticket':'str',
                   'Fare':'float',
                   'Cabin':'str',
                   'Embarked':'str'})

        # Drop not relevant columns
        self.data.drop(columns=["PassengerId", "Ticket", "Cabin"], inplace=True)

        self.data["Embarked"] = self.fillna_with_most_common("Embarked")
        self.data["Age"] = self.fillna_with_mean("Age", "Pclass")
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
        if self.data[column].isna().sum() == 0:
            return self.data[column]
        most_common = self.data[column].value_counts().idxmax()
        return self.data[column].fillna(most_common)

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
        if self.data[column2fill].isna().sum() == 0:
            return self.data[column2fill]
        if self.data[column2fill].count() == 1:
            return self.data[column2fill]
        if self.data[column2fill].count() == 0:
            return self.data[column2fill].fillna(30)

        subset_df = self.data[[column2fill, column_cat]]
        means = subset_df.groupby([column_cat])[column2fill].mean()
        subset_df.set_index([column_cat], inplace=True)
        subset_df = subset_df[column2fill].fillna(means)
        subset_df = subset_df.reset_index()
        return subset_df[column2fill]