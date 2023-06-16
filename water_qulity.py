import os
import pickle

import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models import LGBM, ModelBase


class WaterQuality:
    def __init__(self, show_report=True):
        self.show_report = show_report
    
    
    def load_data(self, data_path='dataset.csv'):
        """
        Load training and testing data.
        Use 'preprocessed_dataset.csv' if exsits
        """
         
        if os.path.exists("preprocessed_dataset.csv"):
            print("Use existing preprocessed data")
            print("Read preprocessed data...")
            self.data = pd.read_csv("preprocessed_dataset.csv")
        else:
            print("Read raw data...")
            self.data = pd.read_csv(data_path)
            print(f"Start preprocessing (data shape: {self.data.shape})...")

            self.data["Source"].fillna("NA", inplace=True)
            self.data["Color"].fillna("NA", inplace=True)
            self.data["Month"].fillna("NA", inplace=True)

            encoder = LabelEncoder()
            self.data["Source"] = encoder.fit_transform(self.data["Source"].astype(str))
            self.data["Color"] = encoder.fit_transform(self.data["Color"].astype(str))
            self.data["Month"] = encoder.fit_transform(self.data["Month"].astype(str))

            self.data.fillna(self.data.mean(), inplace=True)

            print("Write to preprocessed_dataset.csv...")
            self.data.to_csv("preprocessed_dataset.csv", index=False)
            
        print("Split the dataset into training and testing sets...")
        self.data = self.data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
        X = self.data.drop("Target", axis=1)
        target = self.data["Target"]
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            X, target, test_size=0.33, random_state=42, stratify=target
        )
    
    def load_model(self, model):
        '''
        Load initial model or trained model saved by pickle
        '''
        if isinstance(model, ModelBase):    
            self.model = model
            self.model.load()
        elif isinstance(model, str):
            with open(model, 'rb') as model_file:
                self.model = pickle.load(model_file)
    
    def save_model(self, save_path='test.pk'):
        '''
        Save model to file
        '''
        self.model.save(save_path)
    
    def train(self):
        self.model.train(self.data_train, self.target_train,
                         self.data_test, self.target_test)
    
    def predict(self, data=[]):
        '''
        Predict water quality.
        
        If data is not provided, doing prediction for the whole testset.
        '''
        if len(data) == 0:
            result = self.model.predict(self.data_test)
            if self.show_report:
                print(
                    classification_report(
                        self.target_test,
                        result,
                        target_names=["Safe to drink", "Not safe to drink"],
                    )
                )
        else:
            result = self.model.predict(data)
        return result
    
    def __call__(self, *args):
        return self.predict(*args)
        


if __name__ == "__main__":
    system = WaterQuality()
    system.load_data("preprocessed_dataset.csv")
    model = LGBM()
    system.load_data(model)
    system.train()
    # system.save_model()
    # system.load_model('test.pk')
    system.predict()
    