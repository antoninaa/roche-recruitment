from flask import Flask, json, request
from flask_cors import CORS
from flask_restful import reqparse, Api, Resource
import pandas as pd
from src.predict import Predict

app = Flask(__name__)
CORS(app)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('PassengerId')
parser.add_argument('Pclass')
parser.add_argument('Name')
parser.add_argument('Sex')
parser.add_argument('Age')
parser.add_argument('SibSp')
parser.add_argument('Parch')
parser.add_argument('Ticket')
parser.add_argument('Fare')
parser.add_argument('Cabin')
parser.add_argument('Embarked')


class Predict(Resource):
    def post(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_text = args['PassengerId']
        sex = args['Pclass']

        headers = ['PassengerId',
                   'Pclass',
                   'Name',
                   'Sex',
                   'Age',
                   'SibSp',
                   'Parch',
                   'Ticket',
                   'Fare',
                   'Cabin',
                   'Embarked']
        param_dict ={}
        for column in headers:
            param_dict[column] = args[column]

        data = pd.DataFrame(param_dict, index=[0], columns=headers)

        predict_obj = Predict(data, 'data/model.pkl')
        prediction = predict_obj.run()

        if prediction[0] == 0:
            pred_text = 'Not survived'
        else:
            pred_text = 'Survived'
        # create JSON object
        output = {'prediction': pred_text}

        return output

    def get(self):
        return {'prediction': 'none'}


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(debug=False)
