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

        # Drop not relevant columns
        self.data.drop(columns=["PassengerId", "Ticket", "Cabin"], inplace=True)
        return self.data
