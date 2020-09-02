def predict():
    # use parser and find the user's query
    args = request.json['data']
    # user_text = args
    #
    # headers = ['PassengerId',
    #            'Pclass',
    #            'Name',
    #            'Sex',
    #            'Age',
    #            'SibSp',
    #            'Parch',
    #            'Ticket',
    #            'Fare',
    #            'Cabin',
    #            'Embarked']
    # data = pd.DataFrame(user_text, columns=headers,
    #                                index=[0])
    #
    # predict_obj = Predict(data, '..data/model.pkl')
    # prediction = predict_obj.run()
    #
    # # Output either 'Negative' or 'Positive' along with the score
    # if prediction[0] == 0:
    #     pred_text = 'Not survived'
    # else:
    #     pred_text = 'Survived'

    # create JSON object
    output = {'prediction'}

    return output