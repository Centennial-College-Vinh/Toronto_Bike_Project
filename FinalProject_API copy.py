from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import joblib
import sys
from flask_cors import CORS

# Your API definition
app = Flask(__name__)
CORS(app)
#app.config.from_object('config')

@app.route("/predict_lr", methods=['GET','POST']) #use decorator pattern for the route
def json():
    if model1:
        try:
            json_ = request.get_json()
            print('json_ : ', json_, '\n\n')
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = list(model1.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


@app.route("/predict_dt", methods=['GET','POST']) #use decorator pattern for the route
def json2():
    if model2:
        try:
            json_ = request.get_json()
            print('json_ : ', json_, '\n\n')
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = list(model2.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model1 = joblib.load('C:/Users/Phong/Desktop/COMP309/Final Project/logistic_model_upsampled.pkl')
    model2 = joblib.load('C:/Users/Phong/Desktop/COMP309/Final Project/decision_tree_model_upsampled.pkl')
        #Load("model.pkl")
    print ('Model loaded')
    model_columns = joblib.load('C:/Users/Phong/Desktop/COMP309/Final Project/model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)