from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model_joblib')  # Load trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])  # Get input from form
    input_df = pd.DataFrame({'sqft_living': [sqft]})
    prediction = model.predict(input_df)[0]
    return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
