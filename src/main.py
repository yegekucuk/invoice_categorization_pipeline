import pickle
import pandas as pd
from Model import Model
from sklearn.preprocessing import LabelEncoder

RESULTS_PATH = "results"

model:Model
le:LabelEncoder

def load():
    # Assign global variables
    global model, le
    # Load model
    with open(f"{RESULTS_PATH}/model.pkl", 'rb') as f:
        model = pickle.load(f)
    # Load y_encoder
    with open(f"{RESULTS_PATH}/y_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
    print(model)
    print(le)

def run():
    # Item features
    description = input("Enter the name of your item: ")
    vendor = input("Enter the name of vendor: ")
    net_amount = input("Enter the price of item: ")
    
    # Create input for the model
    x = []
    x.append([vendor, net_amount, description])
    x = pd.DataFrame(x, columns=["vendor", "net_amount", "description"])
    
    # Make a prediction
    y_pred = model.predict(x)
    y_pred = le.inverse_transform(y_pred)
    print(f"Predict: {str(y_pred[0])}")

if __name__ == "__main__":
    # Load the models
    load()
    # Run
    while True:
        run()
