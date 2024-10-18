from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

model = joblib.load('insurance_premium_model.pkl')
app = Flask(__name__)


CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    
    return jsonify({'premium': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
