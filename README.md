* House Price Prediction Web Application

This project is a simple machine learning-powered web application that predicts house prices based on the square footage of the property. The model is trained using Simple Linear Regression on a housing dataset and deployed using Flask.

* Project Structure

.
├── 7_may_SLR_HousePrice.py     # Script to train and serialize the ML model
├── model_joblib                # Serialized trained model using Joblib
├── myapp.py                    # Flask web application
├── index.html                  # Frontend HTML template
├── kc_house_data.csv           # (You need to include this) Dataset used for training
├── areas.csv                   # (Optional) For testing batch predictions
└── myprediction.csv            # Output file of predictions (generated)

* Features

Trains a linear regression model to predict house prices based on sqft_living.

Saves the trained model using Joblib.

Provides a web interface for predicting price based on user input.

Visualizes the regression line and data points.

* Model

Input: Square footage of the house.

Output: Predicted price.

Algorithm: Simple Linear Regression using scikit-learn.

* Installation and Running the App

1. Clone the repository or download files

2. Install dependencies

pip install flask pandas scikit-learn matplotlib joblib

3. Train the model (if not already trained)

Ensure kc_house_data.csv is in your working directory, then run:

python 7_may_SLR_HousePrice.py

This will create model_joblib (the serialized model).

4. Run the Flask App

python myapp.py

5. Visit the Application
