import pandas as pd 
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

CATEGORIES = np.array(['Status','Brand','Model','CPU','RAM','Storage','Storage type','GPU','Screen','Touch','Final Price'])

def main():
    if len(sys.argv) != 2:
        sys.exit("Please provide the data!")

    encoders, evidence, labels = load_data(sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=0.1
    )
    
    model = train_model(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'mean Absolute Error (MAE): {mae}')

    new_laptop = get_input()
    predicted_price = predict_price(model, new_laptop, encoders)
    print(f'Predicted Laptop Price: ${predicted_price:.2f}')

def load_data(filename):
    df = pd.read_csv(filename, usecols=CATEGORIES)
    df = df.fillna({'GPU':0, 'Storage type':'SSD', 'Screen':df['Screen'].mean()})
    # Dictionary to store encoders for categorical columns
    # can't use the same encoder since different types of variables
    encoders = {}

    for col in CATEGORIES[:-1]:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))

    evidence = df.iloc[:,:-1].to_numpy()
    labels = df.iloc[:,-1].to_numpy()
    return encoders, evidence, labels

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def get_input():
    user_data = []
    for col in CATEGORIES[:-1]:
        user_insert = input(f"What's the laptop's {col}: ")
        user_data.append(user_insert)
    return np.array(user_data)

def predict_price(model, new_laptop, encoders):
    """Encodes a new laptop's features and predicts its price"""
    encoded_values = []
    for i, col in enumerate(CATEGORIES[:-1]):
        encoded_values.append(encoders[col].transform([str(new_laptop[i])])[0])

    input_data = np.array([encoded_values]).astype(float)

    predicted_price = model.predict(input_data)[0]
    return predicted_price

if __name__ == "__main__":
    main()
