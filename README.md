# 🏠 House Price Prediction Web Application

This project is a simple machine learning-powered web application that predicts house prices based on the square footage of the property. The model is trained using **Simple Linear Regression** on a housing dataset and deployed using **Flask**.


## 🚀 Features

- Trains a linear regression model to predict house prices based on `sqft_living`.
- Saves the trained model using Joblib.
- Provides a web interface for predicting price based on user input.
- Visualizes the regression line and data points.

## 📊 Model

- **Input**: Square footage of the house.
- **Output**: Predicted price.
- **Algorithm**: Simple Linear Regression using `scikit-learn`.

## 💻 Installation and Running the App

### 1. Clone the repository or download files

### 2. Install dependencies

pip install flask pandas scikit-learn matplotlib joblib

### 3. Train the model
Ensure that the dataset file kc_house_data.csv is present in the project directory. Then run the following command to train the model and generate the serialized file:
python 7_may_SLR_HousePrice.py
This will train the Linear Regression model and save it as model_joblib.

### 4. Run the Flask web application
Start the development server with:
python myapp.py

