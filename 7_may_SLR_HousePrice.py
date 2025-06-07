#----"Model training, prediction, and serialization----------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('kc_house_data.csv')
#•	The dataset includes various features;
#we'll focus on sqft_living and price.
#3. Select Features and Target Variable
# Select the predictor and target variable
sqf=data['sqft_living']
print(sqf.head(5))
print("--------------------------------------")
SqareFeet = data[['sqft_living']]  # Predictor
Price = data['price']          # Target variable
print(SqareFeet)
print(Price)
#•	X: Independent variable (square footage).
#•	y: Dependent variable (house price).
#4. Create and Train the Model
# Create the linear regression model
model = LinearRegression()
# Train the model
model.fit(SqareFeet, Price)
#	model.fit(X, y): Fits the linear model to the data.
#5. Retrieve Model Parameters
# Retrieve the slope (coefficient) and intercept
slope = model.coef_[0] #m  slope in the equation
intercept = model.intercept_ # c intercept in the equation

print(f"Slope(m) (b1): {slope}")
print(f"Intercept (b): {intercept}")
#slope (b1): Indicates the change in price for each additional square foot.
#intercept (b0): The predicted price when sqft_living is zero.
#6. Make Predictions
# Predict the price for a house with 2000 sqft

predicted_price = model.predict(pd.DataFrame({'sqft_living': [2000]}))

print(f"Predicted price for 2000 sqft: ${predicted_price[0]:,.2f}")

#Predicts the price of a house with 2000 square feet using the trained model.

predicted_price = model.predict(pd.DataFrame({'sqft_living': [1000]}))

print(f"Predicted price for 1000 sqft: ${predicted_price[0]:,.2f}")

#predciting using a another csv file
area_df = pd.read_csv("areas.csv")
area_df.head(3)
newprices=model.predict(area_df)
print(newprices)

area_df['Prices']=newprices
print("new df",area_df)
area_df.to_csv("myperdiction.csv")


#Checkinh manually by substituignin teh equation y=mx+c
#Whre m is the slope and c is the inetrcept
manual_price=slope*2000+intercept
print("manual=",manual_price)
'''
Hoemprice=m*area+b
m-slope
b-itercept
'''

print("Predicted price -",predicted_price)
print("Manually calcualted price using formula -",manual_price)
print("Bingo it worked ",(predicted_price==manual_price))


#7. Visualize the Results
# Plot the data and the regression line
plt.scatter(SqareFeet, Price, color='blue', alpha=0.5, label='Actual Data')
plt.plot(SqareFeet, model.predict(SqareFeet), color='red', linewidth=2, label='Regression Line')
plt.title('House Price vs. Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#Saving the price of modulising pickle
joblib.dump(model, 'model_joblib')
house_price=joblib.load('model_joblib')
print(house_price.coef_)
print(house_price.intercept_)

newPriceObtd=house_price.predict(pd.DataFrame({'sqft_living':[2000]}))
print(f"Predicted price using pickled model 2000 sqft: ${predicted_price[0]:,.2f}")

newPriceObtd=house_price.predict(pd.DataFrame({'sqft_living':[1000]}))
print(f"Predicted price using pickled model 2000 sqft: ${predicted_price[0]:,.2f}")









      
