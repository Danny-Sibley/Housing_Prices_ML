import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# deserialize pickle object
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # inputted values from user
    user_features = [int(x) for x in request.form.values()]
    final_features = [np.array(user_features)]
    # creates prediciton using model
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    # outputs to html file
    # ZIPCODE, BEDS, BATHS, SQFT, LOT_SIZE, YEAR_BUILT
    return render_template('index.html', prediction_text=(f'ZIPCODE: {final_features[0][0]}, Bedrooms:{final_features[0][1]}, Bathrooms: {final_features[0][2]}, SQFT: {final_features[0][3]}, Lot Size: {final_features[0][4]}, Year Built: {final_features[0][5]} \n Predicted Home Value: {output}'))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
